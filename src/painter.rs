//! The `vello_cpu` bridge: rasterizes a display list into a region-sized
//! scratch pixmap.
//!
//! This is the only module that touches `vello_cpu` directly; its 0.0.x API
//! is expected to change and the churn must stay contained here. Region
//! rendering works by translating every command by the region origin and
//! rasterizing into a context exactly the size of the region — sparse-strip
//! rasterization only pays for covered pixels, so this is cheap even though
//! the full scene is replayed.
//!
//! Scenes reach it as recorded command lists ([`waterui_graphics::draw`]'s
//! `cherenkov_record` vocabulary), replayed band by band through
//! [`SceneReplay`] — no offscreen surface, no readback, no bitmap, so a
//! scene costs a draw exactly what its own commands cost and fractional
//! placement is as exact as any other fill's.

use kurbo::{
    Affine, BezPath, Cap, Join, PathEl, Point, Rect, RoundedRect, RoundedRectRadii, Shape, Stroke,
};
use peniko::ImageSampler;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::{
    Image, ImageSource, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings,
    Resources,
};
use waterui_graphics::draw;

use crate::color::to_peniko;
use crate::compositor::DeviceRegion;
use crate::display_list::{BEZIER_TOLERANCE, Clip, ClipRegion, DisplayList, DrawCommand, Scene};
use crate::stats::FrameWork;
use crate::views::scene::SceneTarget;

/// Image sources uploaded to `vello_cpu`, reused across frames that draw the
/// same image brush.
#[derive(Debug, Default)]
struct CpuImageCache {
    images: Vec<CachedImage>,
}

#[derive(Debug)]
struct CachedImage {
    data: peniko::ImageData,
    source: ImageSource,
}

impl CpuImageCache {
    const fn new() -> Self {
        Self { images: Vec::new() }
    }

    #[cfg(test)]
    const fn len(&self) -> usize {
        self.images.len()
    }

    fn source_for(&mut self, image: &peniko::ImageBrush) -> ImageSource {
        if let Some(cached) = self.images.iter().find(|cached| cached.data == image.image) {
            return cached.source.clone();
        }
        let source = ImageSource::from_peniko_image_data(&image.image);
        self.images.push(CachedImage {
            data: image.image.clone(),
            source: source.clone(),
        });
        source
    }

    /// Selects `brush` as the paint of `ctx`, uploading an image brush once.
    fn set_brush(&mut self, ctx: &mut RenderContext, brush: &peniko::Brush) {
        match brush {
            peniko::Brush::Solid(color) => ctx.set_paint(*color),
            peniko::Brush::Gradient(gradient) => ctx.set_paint(gradient.clone()),
            peniko::Brush::Image(image) => {
                let source = self.source_for(image);
                ctx.set_paint(Image {
                    image: source,
                    sampler: image.sampler,
                });
            }
        }
    }
}

/// The rasterizer: owns the persistent `vello_cpu` resources (glyph atlas,
/// image registry) that must survive across bands and frames.
///
/// One painter per screen; create it once and reuse it for every region.
#[derive(Debug)]
pub struct Painter {
    resources: Resources,
    profile: RenderProfile,
    images: CpuImageCache,
    scratch: Vec<ScratchSlot>,
    /// Buffers a scene replay reuses across commands, bands and frames —
    /// steady-state replay allocates nothing per band.
    scene: SceneScratch,
}

/// A reusable render context and pixmap for one region size.
///
/// A `RenderContext` is fixed-size, and steady-state dirty regions repeat the
/// same handful of sizes frame after frame (an animating progress bar dirties
/// identical bands every frame). Recreating the context and pixmap per band
/// was the single largest source of per-frame heap churn the work simulation
/// measured, and on an RTOS heap churn is fragmentation pressure — so the
/// painter keeps one slot per recent size and `reset()`s it instead.
#[derive(Debug)]
struct ScratchSlot {
    width: u16,
    height: u16,
    context: RenderContext,
    pixmap: Pixmap,
    rasterizer: RasterizerSettings,
}

/// Distinct region sizes kept alive for reuse, least recently used evicted.
///
/// Steady-state frames cycle through only a few sizes; the bound exists so a
/// pathological size storm cannot hoard band-sized buffers.
const SCRATCH_SLOTS: usize = 8;

/// How a board rasterizes: what its render contexts are created with (SIMD
/// level, worker threads) and the pipeline every region is rendered through.
#[derive(Debug, Clone, Copy)]
pub struct RenderProfile {
    /// Settings each scratch render context is created with.
    pub context: RenderSettings,
    /// Settings each region is rasterized with.
    pub rasterizer: RasterizerSettings,
}

impl Default for RenderProfile {
    fn default() -> Self {
        target_render_profile()
    }
}

impl Default for Painter {
    fn default() -> Self {
        Self::new(target_render_profile())
    }
}

impl Painter {
    /// Creates a painter with empty caches and an explicit render profile.
    #[must_use]
    pub fn new(profile: RenderProfile) -> Self {
        Self {
            resources: Resources::new(),
            profile,
            images: CpuImageCache::new(),
            scratch: Vec::new(),
            scene: SceneScratch::default(),
        }
    }

    /// The reusable scratch slot for a `width` × `height` region, creating it
    /// (evicting the least recently used) when absent. The returned slot's
    /// context is reset and ready to encode.
    fn scratch_slot(&mut self, width: u16, height: u16) -> &mut ScratchSlot {
        if let Some(index) = self
            .scratch
            .iter()
            .position(|slot| slot.width == width && slot.height == height)
        {
            // Move to the back: the back is the most recently used.
            let slot = self.scratch.remove(index);
            self.scratch.push(slot);
        } else {
            if self.scratch.len() == SCRATCH_SLOTS {
                self.scratch.remove(0);
            }
            self.scratch.push(ScratchSlot {
                width,
                height,
                context: RenderContext::new_with(width, height, self.profile.context),
                pixmap: Pixmap::new(width, height),
                rasterizer: self.profile.rasterizer,
            });
        }
        let slot = self
            .scratch
            .last_mut()
            .expect("a scratch slot was just ensured");
        slot.context.reset();
        slot
    }

    /// Rasterizes the window-coordinate `list` clipped to `region`,
    /// returning a `region.width × region.height` premultiplied-RGBA8
    /// pixmap.
    ///
    /// The pixmap borrows the painter's reusable scratch slot for this
    /// region size and is valid until the next `rasterize_region` call —
    /// callers stream it out immediately, which is also the only usage the
    /// banded flush model permits.
    ///
    /// `candidates` are indices into `list.commands()` that a spatial index
    /// has already established *may* touch this region's band row; the
    /// painter still tests each one against the exact region. Passing every
    /// index is correct but reduces the pass to the quadratic scan the index
    /// exists to avoid.
    ///
    /// # Panics
    ///
    /// Panics when the region exceeds `u16::MAX` in either dimension, far
    /// beyond any target panel, or when a candidate index is out of range.
    #[must_use]
    pub fn rasterize_region(
        &mut self,
        list: &DisplayList,
        region: DeviceRegion,
        candidates: &[u32],
        work: &mut FrameWork,
    ) -> &Pixmap {
        let width = u16::try_from(region.width).expect("region width exceeds u16::MAX");
        let height = u16::try_from(region.height).expect("region height exceeds u16::MAX");
        // Ensure the slot exists and is reset, then split borrows so the
        // brush cache and the slot can be used simultaneously.
        let _ = self.scratch_slot(width, height);
        let Self {
            resources,
            images,
            scratch,
            scene: scene_scratch,
            ..
        } = self;
        let slot = scratch
            .last_mut()
            .expect("scratch_slot just ensured a slot");
        let ctx = &mut slot.context;
        let shift = Affine::translate((-f64::from(region.x), -f64::from(region.y)));
        let region_bounds = Rect::new(
            f64::from(region.x),
            f64::from(region.y),
            f64::from(region.x + region.width),
            f64::from(region.y + region.height),
        );
        let commands = list.commands();
        work.command_band_visits += candidates.len() as u64;
        work.pixels_rasterized += region.area();
        for index in candidates {
            let placed = &commands[usize::try_from(*index)
                .expect("display-list command index must fit a pointer-sized value")];
            if !placed.intersects(region_bounds) {
                continue;
            }
            work.command_band_draws += 1;
            let command = placed.command();
            let Some(clip_depth) = push_clip_layers(ctx, command.clip(), shift) else {
                continue;
            };
            paint_command(
                ctx,
                &mut PaintTables {
                    resources,
                    images,
                    scene_scratch,
                },
                command,
                shift,
                region_bounds,
                work,
            );
            for _ in 0..clip_depth {
                ctx.pop_clip_path();
            }
        }
        ctx.flush();
        slot.render(resources);
        &slot.pixmap
    }
}

/// The painter's per-band tables: resource resolution, the image-brush
/// cache and the reusable scene-replay scratch — bundled so a per-command
/// call stays inside the argument limit.
struct PaintTables<'a> {
    resources: &'a mut Resources,
    images: &'a mut CpuImageCache,
    scene_scratch: &'a mut SceneScratch,
}

/// Rasterizes one draw command into `ctx` with `shift` applied — the region
/// translation every transform multiplies.
fn paint_command(
    ctx: &mut RenderContext,
    tables: &mut PaintTables<'_>,
    command: &DrawCommand,
    shift: Affine,
    region_bounds: Rect,
    work: &mut FrameWork,
) {
    match command {
        DrawCommand::FillPath {
            path,
            transform,
            brush,
            ..
        } => {
            ctx.set_transform(shift * *transform);
            tables.images.set_brush(ctx, brush);
            ctx.fill_path(path);
        }
        DrawCommand::StrokePath {
            path,
            transform,
            stroke,
            brush,
            ..
        } => {
            ctx.set_transform(shift * *transform);
            ctx.set_stroke(stroke.clone());
            tables.images.set_brush(ctx, brush);
            ctx.stroke_path(path);
        }
        DrawCommand::GlyphRun {
            font,
            font_size,
            glyphs,
            glyph_bounds,
            transform,
            brush,
            ..
        } => {
            ctx.set_transform(shift * *transform);
            tables.images.set_brush(ctx, brush);
            ctx.glyph_run(tables.resources, font)
                .font_size(*font_size)
                .hint(true)
                .fill_glyphs(
                    glyphs
                        .iter()
                        .zip(glyph_bounds.iter())
                        .filter(|(_, bounds)| {
                            transform
                                .transform_rect_bbox(**bounds)
                                .intersect(region_bounds)
                                .area()
                                > 0.0
                        })
                        .map(|(glyph, _)| *glyph),
                );
        }
        DrawCommand::Scene {
            scene,
            transform,
            bounds,
            ..
        } => {
            replay_scene(
                ctx,
                tables.resources,
                scene,
                *bounds,
                shift * *transform,
                work,
                tables.scene_scratch,
            );
        }
    }
}

/// What a scope opened inside a scene recording is: which stack an `End`
/// pops, and what state a transform scope restores.
#[derive(Debug)]
enum SceneScope {
    /// A `BeginClip`: popped by `pop_clip_path`.
    Clip,
    /// A `BeginGroup`: popped by `pop_layer`.
    Layer,
    /// A `BeginTransform`: restores the ambient transform the scope opened
    /// under — every draw sets its transform from `current`, so the ambient
    /// transform is the only state a transform scope carries.
    Transform(Affine),
}

/// Buffers a scene replay reuses across commands, bands and frames: the
/// scope stack, the path each shape builds into and the gradient stop list.
#[derive(Debug, Default)]
struct SceneScratch {
    /// Open scopes, outermost first.
    scopes: Vec<SceneScope>,
    /// The path a `ShapeData` converts into.
    path: BezPath,
    /// `path` with every subpath closed — the dilation stroke's copy.
    closed: BezPath,
    /// The peniko stops a gradient paint converts into.
    stops: Vec<peniko::ColorStop>,
}

/// Replays one scene recording's commands into a band's render context.
///
/// The `cherenkov_record` vocabulary maps one to one onto `vello_cpu`:
/// fills, strokes, glyph runs and images draw at the ambient transform;
/// `BeginClip`/`BeginTransform`/`BeginGroup` open clip, state and layer
/// scopes; `End` closes the innermost one; `Picture` recurses under its
/// placement. Paint transforms in the recording compose onto
/// `vello_cpu`'s `set_paint_transform`, which is applied after the scene
/// transform — the same image-space-to-content-space convention the
/// recording's `ImagePattern::transform` uses.
///
/// The replay restores the context's ambient state when it exits —
/// [`replay_scene`] resets the fill rule and paint transform — and a
/// recording that pushes a scope it never closes is a content defect, not
/// a condition to paint around.
struct SceneReplay<'a> {
    ctx: &'a mut RenderContext,
    resources: &'a mut Resources,
    target: &'a SceneTarget,
    /// The band's rectangle in canvas coordinates — the cull box per-glyph
    /// filtering intersects against.
    canvas: Rect,
    /// The frame-work counters every replayed command lands in.
    work: &'a mut FrameWork,
    /// The ambient transform: `shift * command.transform` with each open
    /// `BeginTransform` and `Picture` placement multiplied in.
    current: Affine,
    /// Open scopes, outermost first — the `Painter`'s buffer, so steady-state
    /// replay allocates nothing per band.
    scopes: &'a mut Vec<SceneScope>,
    /// The path a `ShapeData` converts into — the `Painter`'s buffer.
    scratch: &'a mut BezPath,
    /// `scratch` with every subpath closed — the `Painter`'s buffer for a
    /// shadow's dilation stroke.
    closed: &'a mut BezPath,
    /// The peniko stops a gradient paint converts into — the `Painter`'s
    /// buffer.
    stops: &'a mut Vec<peniko::ColorStop>,
}

/// Replays `scene`'s recorded commands into `ctx` under `transform`,
/// resolving the recording's `FontId`/`ImageId` names through the scene's
/// own registration table.
///
/// # Panics
///
/// Panics when the recording leaves a scope unclosed — a defect in the
/// content, not a condition to paint around. Vocabulary `vello_cpu` has no
/// equivalent for (mesh gradients, shader paints, `Extend::None`, filter
/// chains, line shadows) is refused at install and is `unreachable!` here.
fn replay_scene(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    scene: &Scene,
    bounds: Rect,
    transform: Affine,
    work: &mut FrameWork,
    scratch: &mut SceneScratch,
) {
    // Read-only: the node drains pending updates through `take_change` on
    // every render, so `view` here is the committed list — the painter
    // shares it and never writes. The slot holds `Some` whenever a `Scene`
    // command exists — the node emits only for installed, non-empty content.
    let content = scene.content.borrow();
    let list = content
        .as_ref()
        .expect("a Scene command exists only for installed content")
        .view();
    // A scene composites as an isolated layer clipped to its own box, as
    // `scene.rs` promises: a group whose blend is not `Normal` must not
    // reach dew's pixels — a `Clear`/`DestOut` group would erase the
    // background and widgets behind the scene, a `Multiply`/`Screen` would
    // blend against them — and clipping the layer to the scene's bounds is
    // what keeps even that isolated compositing inside the box. The flag is
    // the node's `blends_within` verdict for this generation, computed
    // where the mutable borrow was already held.
    if scene.isolated {
        ctx.set_transform(transform);
        ctx.set_fill_rule(peniko::Fill::NonZero);
        scratch.path.truncate(0);
        scratch.path.extend(bounds.path_elements(BEZIER_TOLERANCE));
        ctx.push_layer(Some(&scratch.path), None, None, None, None);
    }
    {
        let canvas = Rect::new(0.0, 0.0, f64::from(ctx.width()), f64::from(ctx.height()));
        let mut replay = SceneReplay {
            ctx,
            resources,
            target: &scene.target,
            canvas,
            work,
            current: transform,
            scopes: &mut scratch.scopes,
            scratch: &mut scratch.path,
            closed: &mut scratch.closed,
            stops: &mut scratch.stops,
        };
        replay.commands(list.commands());
        assert!(
            replay.scopes.is_empty(),
            "scene content left scope(s) unclosed: every Begin must have a matching End"
        );
    }
    if scene.isolated {
        ctx.pop_layer();
    }
    // The replay leaves the context's ambient state as it found it: a
    // recording that armed an even-odd rule or a paint transform must not
    // colour the next dew command's fill.
    ctx.set_fill_rule(peniko::Fill::NonZero);
    ctx.reset_paint_transform();
}

impl SceneReplay<'_> {
    /// Replays every command in `commands`, resolving nested pictures
    /// recursively.
    fn commands(&mut self, commands: &[draw::Command]) {
        for command in commands {
            self.command(command);
        }
    }

    fn command(&mut self, command: &draw::Command) {
        self.work.command_band_visits += 1;
        match command {
            draw::Command::Fill { shape, paint } => {
                self.ctx.set_transform(self.current);
                self.ctx.set_fill_rule(fill_rule(shape));
                self.paint(paint);
                self.shape_path(shape);
                self.ctx.fill_path(self.scratch);
                self.work.command_band_draws += 1;
            }
            draw::Command::Stroke {
                shape,
                stroke,
                paint,
            } => {
                self.ctx.set_transform(self.current);
                self.ctx.set_stroke(stroke.clone());
                self.paint(paint);
                self.shape_path(shape);
                self.ctx.stroke_path(self.scratch);
                self.work.command_band_draws += 1;
            }
            draw::Command::Shadow { shape, shadow } => {
                self.shadow(shape, shadow);
                self.work.command_band_draws += 1;
            }
            draw::Command::Glyphs { run, paint } => self.glyphs(run, paint),
            draw::Command::Image {
                image,
                dst,
                sampling,
            } => {
                self.image(*image, *dst, *sampling);
                self.work.command_band_draws += 1;
            }
            draw::Command::Picture { picture, transform } => {
                let outer = self.current;
                self.current = outer * *transform;
                self.commands(picture.display_list().commands());
                self.current = outer;
            }
            draw::Command::BeginClip { shape, .. } => {
                self.ctx.set_transform(self.current);
                self.ctx.set_fill_rule(fill_rule(shape));
                self.shape_path(shape);
                self.ctx.push_clip_path(self.scratch);
                self.scopes.push(SceneScope::Clip);
                self.work.command_band_draws += 1;
            }
            draw::Command::BeginTransform { transform, .. } => {
                let outer = self.current;
                self.current = outer * *transform;
                self.scopes.push(SceneScope::Transform(outer));
            }
            draw::Command::BeginGroup { group, .. } => {
                if let Some(filter) = group.filter {
                    unreachable!(
                        "a scene group naming filter {} is refused at install",
                        filter.raw()
                    );
                }
                // vello_cpu composites premultiplied sRGB — what every dew
                // shape already blends in. The recording's `BlendSpace`
                // selects between working-space (linear) and sRGB-encoded
                // compositing, and dew's rasterizer cannot composite in a
                // linear space, so a `Linear` group composites in sRGB too:
                // that is a documented divergence from the cherenkov
                // reference, where a 0.5-opacity layer over black lands at
                // ~188 and dew lands it at ~128 — dew's sRGB compositing,
                // not the contract's.
                // No clip: the layer needs no transform at push — every
                // command inside carries its own.
                self.ctx.push_layer(
                    None,
                    Some(blend_mode(group.blend)),
                    Some(group.opacity),
                    None,
                    None,
                );
                self.scopes.push(SceneScope::Layer);
                self.work.command_band_draws += 1;
            }
            draw::Command::End => {
                let scope = self
                    .scopes
                    .pop()
                    .expect("scene content closed a scope it never opened");
                match scope {
                    SceneScope::Clip => self.ctx.pop_clip_path(),
                    SceneScope::Layer => self.ctx.pop_layer(),
                    SceneScope::Transform(current) => self.current = current,
                }
            }
        }
    }

    /// Converts `shape` into `self.scratch`, reusing its storage.
    fn shape_path(&mut self, shape: &draw::ShapeData) {
        self.scratch.truncate(0);
        match shape {
            draw::ShapeData::Rect(rect) => {
                self.scratch.extend(rect.path_elements(BEZIER_TOLERANCE));
            }
            draw::ShapeData::RoundedRect(rect) => {
                self.scratch.extend(rect.path_elements(BEZIER_TOLERANCE));
            }
            draw::ShapeData::Continuous(continuous) => {
                // `ContinuousRect` has no `path_elements`; `to_path`'s
                // temporary frees after the copy while `self.scratch`'s
                // capacity stays — the buffer a replay reuses.
                self.scratch
                    .extend(continuous.to_path(BEZIER_TOLERANCE).iter());
            }
            draw::ShapeData::Circle(circle) => {
                self.scratch.extend(circle.path_elements(BEZIER_TOLERANCE));
            }
            draw::ShapeData::Ellipse(ellipse) => {
                self.scratch.extend(ellipse.path_elements(BEZIER_TOLERANCE));
            }
            draw::ShapeData::Line(line) => {
                self.scratch.extend(line.path_elements(BEZIER_TOLERANCE));
            }
            draw::ShapeData::Path { elements, .. } => {
                self.scratch.extend(elements.iter().copied());
            }
        }
    }

    /// Installs `paint` as the context's paint, folding the recording's
    /// `Transformed` chain and an image pattern's own transform into the
    /// paint transform.
    fn paint(&mut self, paint: &draw::Paint) {
        let mut accumulated = Affine::IDENTITY;
        let mut paint = paint;
        while let draw::Paint::Transformed(transformed) = paint {
            accumulated *= transformed.transform;
            paint = transformed.paint.as_ref();
        }
        match paint {
            draw::Paint::Solid(color) => {
                self.ctx.set_paint(to_peniko(*color));
            }
            draw::Paint::Linear(gradient) => {
                let paint = self.gradient_paint(gradient);
                self.ctx.set_paint(paint);
            }
            draw::Paint::Radial(gradient) => {
                let paint = self.radial_paint(gradient);
                self.ctx.set_paint(paint);
            }
            draw::Paint::Sweep(gradient) => {
                let paint = self.sweep_paint(gradient);
                self.ctx.set_paint(paint);
            }
            draw::Paint::Image(pattern) => {
                let source = self.target.image(pattern.image);
                self.ctx.set_paint(Image {
                    image: source,
                    sampler: ImageSampler {
                        x_extend: extend(pattern.extend_x),
                        y_extend: extend(pattern.extend_y),
                        quality: quality(pattern.sampling),
                        alpha: 1.0,
                    },
                });
                self.ctx
                    .set_paint_transform(accumulated * pattern.transform);
                return;
            }
            draw::Paint::Mesh(_) | draw::Paint::Shader(_) => {
                unreachable!("mesh and shader paints are refused at install")
            }
            draw::Paint::Transformed(_) => {
                unreachable!("the Transformed chain was consumed above")
            }
        }
        if accumulated == Affine::IDENTITY {
            self.ctx.reset_paint_transform();
        } else {
            self.ctx.set_paint_transform(accumulated);
        }
    }

    /// Draws `dst`'s rectangle filled by the registered `image`, sampled
    /// `sampling` — what the recording's own lowering does: an image pattern
    /// whose transform maps image pixels onto `dst`.
    fn image(&mut self, image: draw::ImageId, dst: Rect, sampling: draw::Sampling) {
        let source = self.target.image(image);
        let ImageSource::Pixmap(pixmap) = &source else {
            unreachable!("dew's scene target registers pixmap sources only")
        };
        let transform = Affine::translate((dst.x0, dst.y0))
            * Affine::scale_non_uniform(
                dst.width() / f64::from(pixmap.width()),
                dst.height() / f64::from(pixmap.height()),
            );
        self.ctx.set_transform(self.current);
        self.ctx.set_fill_rule(peniko::Fill::NonZero);
        self.ctx.set_paint(Image {
            image: source,
            sampler: ImageSampler {
                x_extend: peniko::Extend::Pad,
                y_extend: peniko::Extend::Pad,
                quality: quality(sampling),
                alpha: 1.0,
            },
        });
        self.ctx.set_paint_transform(transform);
        self.scratch.truncate(0);
        self.scratch.extend(dst.path_elements(BEZIER_TOLERANCE));
        self.ctx.fill_path(self.scratch);
        self.ctx.reset_paint_transform();
    }

    /// Draws `run`'s glyphs under the ambient transform, splitting the run
    /// where a per-glyph transform changes: `vello_cpu`'s builder takes one
    /// `glyph_transform` for the whole run, so glyphs sharing a transform
    /// draw together in order. Each glyph is culled per band like dew's own
    /// glyph runs: glifo draws `current * pen_position * glyph_transform *
    /// FLIP_Y * outline`, so the cull box is the font's `head` bounds —
    /// already mirrored into screen space when the font registered — scaled
    /// to the run size, padded by half the stroke width times the miter
    /// limit when the run is stroked, and transformed under the glyph's own
    /// transform so a rotated or skewed glyph (upright CJK) is not culled
    /// from bands its real outline covers. `head` covers only the outline
    /// strikes of the default instance: a run of a font carrying bitmap or
    /// COLR glyphs, or non-default normalized coords, draws every glyph —
    /// culling is an optimisation, drawing is always correct.
    fn glyphs(&mut self, run: &draw::GlyphRun, paint: &draw::Paint) {
        self.ctx.set_transform(self.current);
        self.paint(paint);
        let font = self.target.font(run.font);
        let size = f64::from(run.size);
        let stroke_pad = match &run.style {
            draw::GlyphStyle::Fill => 0.0,
            draw::GlyphStyle::Stroke(stroke) => stroke.width * 0.5 * stroke.miter_limit.max(1.0),
        };
        let local = Rect::new(
            font.em_bbox.x0.mul_add(size, -stroke_pad),
            font.em_bbox.y0.mul_add(size, -stroke_pad),
            font.em_bbox.x1.mul_add(size, stroke_pad),
            font.em_bbox.y1.mul_add(size, stroke_pad),
        );
        let cullable = font.outline_only && run.coords.iter().all(|&coord| coord == 0);
        let mut start = 0;
        while start < run.glyphs.len() {
            let glyph_transform = run.glyphs[start].transform;
            let mut end = start + 1;
            while end < run.glyphs.len() && run.glyphs[end].transform == glyph_transform {
                end += 1;
            }
            let segment = &run.glyphs[start..end];
            let visible = |glyph: &draw::Glyph| {
                !cullable
                    || (self.current
                        * Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                        * glyph.transform.unwrap_or(Affine::IDENTITY))
                    .transform_rect_bbox(local)
                    // Hinting snaps the glyph's y translate to a whole
                    // device pixel — a half-pixel shift — so the cull box
                    // pads by a full one.
                    .inflate(1.0, 1.0)
                    .intersect(self.canvas)
                    .area()
                        > 0.0
            };
            if !segment.iter().any(&visible) {
                start = end;
                continue;
            }
            match &run.style {
                draw::GlyphStyle::Fill => self.ctx.set_fill_rule(peniko::Fill::NonZero),
                draw::GlyphStyle::Stroke(stroke) => self.ctx.set_stroke(stroke.clone()),
            }
            let builder = self
                .ctx
                .glyph_run(self.resources, &font.font)
                .font_size(run.size)
                .normalized_coords(&run.coords)
                .hint(true);
            let builder = match glyph_transform {
                Some(transform) => builder.glyph_transform(transform),
                None => builder,
            };
            let glyphs =
                segment
                    .iter()
                    .filter(|glyph| visible(glyph))
                    .map(|glyph| vello_cpu::Glyph {
                        id: glyph.id,
                        x: glyph.x,
                        y: glyph.y,
                    });
            match &run.style {
                draw::GlyphStyle::Fill => builder.fill_glyphs(glyphs),
                draw::GlyphStyle::Stroke(_) => builder.stroke_glyphs(glyphs),
            }
            self.work.command_band_draws += 1;
            start = end;
        }
    }

    /// Casts `shadow` from `shape`: a drop-shadow-only filter layer over the
    /// shape's silhouette, which is the recording's shadow semantics — the
    /// shape itself is a separate fill.
    ///
    /// `vello_cpu` snapshots the context transform when a filter layer is
    /// pushed and applies it to the filter's offset and sigma again, so the
    /// layer goes up under identity and the offset takes the ambient linear
    /// part exactly once. Inside the layer the silhouette rasterizes in the
    /// shadow colour; spread is exact two ways, matching cherenkov-cpu's
    /// lowering. Under an axis-aligned transform `Rect`, `RoundedRect` and
    /// `Circle` take the closed form in device space — the transformed rect
    /// with `spread * smax` on each half-extent and `radius * smax` on the
    /// corners, spread zero included — so a non-uniform axis-aligned scale
    /// still yields circular corners, a stadium. Every other case draws the
    /// fill plus a round-joined, round-capped stroke `2 * |spread|` wide
    /// *in content space*: a linear map carries the Minkowski sum,
    /// `A(S ⊕ rD) = AS ⊕ A(rD)`, so the stroke under `current` is exactly
    /// `spread_taps`' device-space ellipse — exact for paths whose every
    /// edge bounds coverage; an edge that bounds none (a path of only
    /// `M L`) is still dilated where cherenkov draws nothing, and the fix
    /// is the same path-boolean union, tracked separately. A negative
    /// spread removes that stroke through `DestOut`: exact for
    /// non-overlapping subpaths; on a self-overlapping non-zero path the
    /// stroke also runs along the interior edges, where cherenkov's
    /// coverage erosion does not. `vello_common` declares
    /// `FilterPrimitive::Morphology`, but `PreparedFilter::new` hits
    /// `unimplemented!` on it (and on any multi-primitive graph), and its
    /// single radius is SVG's square kernel, not the disc `spread_taps`
    /// uses — so no vello primitive can erode the union's coverage. The
    /// exact fix is a path-boolean union of the silhouette's outline
    /// before the stroke pass, tracked separately.
    ///
    /// Sigma diverges on the silhouette branch only: cherenkov blurs each
    /// transform axis at `sigma * |axis|`; dew feeds `sigma * smax`,
    /// differing under non-uniform scale or skew. The closed form's
    /// isotropic `sigma * smax` is what cherenkov uses there too.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "shadow geometry is far below f32 precision limits"
    )]
    fn shadow(&mut self, shape: &draw::ShapeData, shadow: &draw::Shadow) {
        let [a, b, c, d, _, _] = self.current.as_coeffs();
        let (sx, sy) = (a.hypot(b), c.hypot(d));
        let smax = sx.max(sy).max(1e-12);
        // Identity must be armed before the push: `vello_cpu` reads the
        // context transform as the layer is pushed and scales the filter's
        // offset and sigma by it — `dx`/`dy`/`std_deviation` are already in
        // device units, so a leftover transform would apply them twice.
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx
            .push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadowOnly {
                dx: c.mul_add(shadow.offset.y, a * shadow.offset.x) as f32,
                dy: d.mul_add(shadow.offset.y, b * shadow.offset.x) as f32,
                std_deviation: (shadow.sigma * smax) as f32,
                color: to_peniko(shadow.color),
                edge_mode: EdgeMode::None,
            }));
        self.ctx.set_fill_rule(fill_rule(shape));
        // The filter draws from the layer's alpha coverage; the silhouette
        // itself must fill opaque or the shadow inherits the leaks.
        self.ctx.set_paint(peniko::Color::BLACK);
        if axis_aligned(self.current)
            && let Some((rect, radii)) = rect_radii(shape)
        {
            // The closed form, in device space under identity — spread zero
            // lands here too, so it cannot diverge from a small positive
            // spread.
            self.spread_path(rect, radii, smax, shadow.spread * smax);
            if !self.scratch.is_empty() {
                self.ctx.fill_path(self.scratch);
            }
        } else {
            // Silhouette in content space: the stroke dilates the outline
            // by a disc of radius `|spread|`, which `current` maps to the
            // same device-space ellipse `spread_taps` convolves with.
            self.ctx.set_transform(self.current);
            self.shape_path(shape);
            self.ctx.fill_path(self.scratch);
            if shadow.spread != 0.0 {
                self.ctx.set_stroke(
                    Stroke::new(2.0 * shadow.spread.abs())
                        .with_join(Join::Round)
                        .with_caps(Cap::Round),
                );
                // The fill closes every subpath implicitly; the stroke
                // does not — a dilation must run the closing edge a fill
                // covers, or the shadow misses a sliver of the outline.
                closed_subpaths(self.closed, self.scratch);
                if shadow.spread < 0.0 {
                    self.ctx.push_layer(
                        None,
                        Some(peniko::BlendMode::new(
                            peniko::Mix::Normal,
                            peniko::Compose::DestOut,
                        )),
                        None,
                        None,
                        None,
                    );
                }
                self.ctx.stroke_path(self.closed);
                if shadow.spread < 0.0 {
                    self.ctx.pop_layer();
                }
            }
        }
        self.ctx.pop_layer();
    }

    /// Builds the shadow silhouette for a rect-family shape into the
    /// scratch path — the closed form, in device space: `rect` transformed,
    /// each half-extent grown by `spread` and clamped at zero, the radii
    /// scaled by `smax` and grown the same way. An empty result builds no
    /// path, so the radii's clamp limit can never go negative.
    fn spread_path(&mut self, rect: Rect, radii: RoundedRectRadii, smax: f64, spread: f64) {
        self.scratch.truncate(0);
        let rect = device_rect(self.current, rect);
        let (hx, hy) = (
            0.5f64.mul_add(rect.width(), spread).max(0.0),
            0.5f64.mul_add(rect.height(), spread).max(0.0),
        );
        if hx == 0.0 || hy == 0.0 {
            return;
        }
        let limit = hx.min(hy);
        let grow = |radius: f64| {
            if radius > 0.0 {
                radius.mul_add(smax, spread).clamp(0.0, limit)
            } else {
                0.0
            }
        };
        let radii = RoundedRectRadii::new(
            grow(radii.top_left),
            grow(radii.top_right),
            grow(radii.bottom_right),
            grow(radii.bottom_left),
        );
        let rect = Rect::from_center_size(rect.center(), (hx * 2.0, hy * 2.0));
        self.scratch
            .extend(RoundedRect::from_rect(rect, radii).path_elements(BEZIER_TOLERANCE));
    }
}

/// Whether `transform`'s 2x2 is axis-aligned — `b == c == 0`, or a 90°
/// rotation with `a == d == 0` — the condition under which cherenkov-cpu
/// keeps the closed-form shadow rather than dilating the silhouette.
fn axis_aligned(transform: Affine) -> bool {
    let [c0, c1, c2, c3, _, _] = transform.as_coeffs();
    (c1 == 0.0 && c2 == 0.0) || (c0 == 0.0 && c3 == 0.0)
}

/// `shape` as a centred rect plus per-corner radii in content space, for
/// the shapes whose shadow has a closed form — `Rect`, `RoundedRect` and
/// `Circle`, as cherenkov-cpu lowers them. The radii clamp to the rect's
/// original half-extents; everything else answers `None` and dilates
/// through the stroke path instead.
fn rect_radii(shape: &draw::ShapeData) -> Option<(Rect, RoundedRectRadii)> {
    match shape {
        draw::ShapeData::Rect(rect) => Some((*rect, RoundedRectRadii::default())),
        draw::ShapeData::RoundedRect(rounded) => {
            let rect = rounded.rect();
            let limit = rect.width().min(rect.height()) / 2.0;
            let radii = rounded.radii();
            Some((
                rect,
                RoundedRectRadii::new(
                    radii.top_left.clamp(0.0, limit),
                    radii.top_right.clamp(0.0, limit),
                    radii.bottom_right.clamp(0.0, limit),
                    radii.bottom_left.clamp(0.0, limit),
                ),
            ))
        }
        draw::ShapeData::Circle(circle) => Some((
            Rect::from_center_size(circle.center, (circle.radius * 2.0, circle.radius * 2.0)),
            RoundedRectRadii::new(circle.radius, circle.radius, circle.radius, circle.radius),
        )),
        _ => None,
    }
}

/// Copies `path` into `closed` closing every subpath. `fill` closes a
/// subpath implicitly; `stroke` does not, and a dilation stroke must run
/// the closing edge the fill covers.
fn closed_subpaths(closed: &mut BezPath, path: &BezPath) {
    closed.truncate(0);
    let mut open = false;
    for el in path.iter() {
        match el {
            PathEl::MoveTo(_) => {
                if open {
                    closed.push(PathEl::ClosePath);
                }
                open = true;
            }
            // A segment after `ClosePath` with no `MoveTo` starts an
            // implicit subpath at the previous start point — unclosed
            // unless it too gets a `ClosePath`.
            PathEl::LineTo(_) | PathEl::QuadTo(_, _) | PathEl::CurveTo(_, _, _) => {
                open = true;
            }
            PathEl::ClosePath => open = false,
        }
        closed.push(el);
    }
    if open {
        closed.push(PathEl::ClosePath);
    }
}

/// The device-space rectangle of `rect` under an axis-aligned transform.
fn device_rect(transform: Affine, rect: Rect) -> Rect {
    let p0 = transform * Point::new(rect.x0, rect.y0);
    let p1 = transform * Point::new(rect.x1, rect.y1);
    Rect::new(
        p0.x.min(p1.x),
        p0.y.min(p1.y),
        p0.x.max(p1.x),
        p0.y.max(p1.y),
    )
}

/// The fill rule a `ShapeData` carries: only a general path can ask for
/// even-odd.
const fn fill_rule(shape: &draw::ShapeData) -> peniko::Fill {
    match shape {
        draw::ShapeData::Path {
            rule: draw::FillRule::EvenOdd,
            ..
        } => peniko::Fill::EvenOdd,
        _ => peniko::Fill::NonZero,
    }
}

/// `vello_cpu` has no `Extend::None` — a gradient or pattern asking for it
/// fails loudly rather than silently padding.
fn extend(extend: draw::Extend) -> peniko::Extend {
    match extend {
        draw::Extend::Pad => peniko::Extend::Pad,
        draw::Extend::Repeat => peniko::Extend::Repeat,
        draw::Extend::Reflect => peniko::Extend::Reflect,
        draw::Extend::None => unreachable!("`Extend::None` is refused at install"),
    }
}

/// The recording's two-value interpolation space onto peniko's color-space
/// tag: the linear working space becomes linear sRGB — peniko has no
/// linear-P3 tag, and both interpolate in a linear space.
const fn interpolation(interpolation: draw::Interpolation) -> peniko::color::ColorSpaceTag {
    match interpolation {
        draw::Interpolation::Working => peniko::color::ColorSpaceTag::LinearSrgb,
        draw::Interpolation::SrgbEncoded => peniko::color::ColorSpaceTag::Srgb,
    }
}

/// The recording's sampling hint onto peniko's image quality.
const fn quality(sampling: draw::Sampling) -> peniko::ImageQuality {
    match sampling {
        draw::Sampling::Nearest => peniko::ImageQuality::Low,
        draw::Sampling::Linear => peniko::ImageQuality::Medium,
    }
}

impl SceneReplay<'_> {
    /// Loads `stops` into the reused stop buffer: offsets carry over, each
    /// `WorkingColor` converts to sRGB the same way dew's own fills are.
    fn fill_stops(&mut self, stops: &[draw::ColorStop]) {
        self.stops.clear();
        self.stops
            .extend(stops.iter().map(|stop| peniko::ColorStop {
                offset: stop.offset,
                color: peniko::color::DynamicColor::from_alpha_color(to_peniko(stop.color)),
            }));
    }

    fn gradient_paint(&mut self, gradient: &draw::LinearGradient) -> peniko::Gradient {
        self.fill_stops(&gradient.stops);
        peniko::Gradient::new_linear(gradient.start, gradient.end)
            .with_extend(extend(gradient.extend))
            .with_interpolation_cs(interpolation(gradient.interpolation))
            .with_stops(&self.stops[..])
    }

    fn radial_paint(&mut self, gradient: &draw::RadialGradient) -> peniko::Gradient {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "gradient geometry is far below f32 precision limits"
        )]
        let (start_radius, end_radius) = (gradient.start_radius as f32, gradient.end_radius as f32);
        self.fill_stops(&gradient.stops);
        peniko::Gradient::new_two_point_radial(
            gradient.start_center,
            start_radius,
            gradient.end_center,
            end_radius,
        )
        .with_extend(extend(gradient.extend))
        .with_interpolation_cs(interpolation(gradient.interpolation))
        .with_stops(&self.stops[..])
    }

    fn sweep_paint(&mut self, gradient: &draw::SweepGradient) -> peniko::Gradient {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "gradient angles are far below f32 precision limits"
        )]
        let (start_angle, end_angle) = (gradient.start_angle as f32, gradient.end_angle as f32);
        self.fill_stops(&gradient.stops);
        peniko::Gradient::new_sweep(gradient.center, start_angle, end_angle)
            .with_extend(extend(gradient.extend))
            .with_interpolation_cs(interpolation(gradient.interpolation))
            .with_stops(&self.stops[..])
    }
}

/// The recording's flat `BlendMode` onto peniko's mix/compose pair: the
/// colour-mixing modes keep `SrcOver` composition; the Porter-Duff modes
/// compose with normal mixing.
fn blend_mode(blend: draw::BlendMode) -> peniko::BlendMode {
    use peniko::{Compose, Mix};
    match blend {
        draw::BlendMode::Normal => Mix::Normal,
        draw::BlendMode::Multiply => Mix::Multiply,
        draw::BlendMode::Screen => Mix::Screen,
        draw::BlendMode::Overlay => Mix::Overlay,
        draw::BlendMode::Darken => Mix::Darken,
        draw::BlendMode::Lighten => Mix::Lighten,
        draw::BlendMode::ColorDodge => Mix::ColorDodge,
        draw::BlendMode::ColorBurn => Mix::ColorBurn,
        draw::BlendMode::HardLight => Mix::HardLight,
        draw::BlendMode::SoftLight => Mix::SoftLight,
        draw::BlendMode::Difference => Mix::Difference,
        draw::BlendMode::Exclusion => Mix::Exclusion,
        draw::BlendMode::Hue => Mix::Hue,
        draw::BlendMode::Saturation => Mix::Saturation,
        draw::BlendMode::Color => Mix::Color,
        draw::BlendMode::Luminosity => Mix::Luminosity,
        draw::BlendMode::Clear => return Compose::Clear.into(),
        draw::BlendMode::Src => return Compose::Copy.into(),
        draw::BlendMode::Dst => return Compose::Dest.into(),
        draw::BlendMode::DestOver => return Compose::DestOver.into(),
        draw::BlendMode::SrcIn => return Compose::SrcIn.into(),
        draw::BlendMode::DestIn => return Compose::DestIn.into(),
        draw::BlendMode::SrcOut => return Compose::SrcOut.into(),
        draw::BlendMode::DestOut => return Compose::DestOut.into(),
        draw::BlendMode::SrcAtop => return Compose::SrcAtop.into(),
        draw::BlendMode::DestAtop => return Compose::DestAtop.into(),
        draw::BlendMode::Xor => return Compose::Xor.into(),
        draw::BlendMode::PlusLighter => return Compose::PlusLighter.into(),
    }
    .into()
}

impl ScratchSlot {
    /// Rasterizes the recorded scene into the slot's pixmap.
    fn render(&mut self, resources: &mut Resources) {
        self.context
            .render_with(&mut self.pixmap, resources, self.rasterizer);
    }
}

/// Pushes every clip layer in force for a command, returning how many were
/// pushed, or [`None`] when the clip admits no pixel at all.
///
/// Clips are in window coordinates, so they only need the region shift, not
/// the command transform. Each region becomes one layer: their intersection is
/// the mask, and the rasterizer computes it exactly rather than approximating
/// it with a boolean path operation.
fn push_clip_layers(ctx: &mut RenderContext, clip: Option<&Clip>, shift: Affine) -> Option<usize> {
    let Some(clip) = clip else {
        return Some(0);
    };
    let bounds = clip.bounds();
    if bounds.width() <= 0.0 || bounds.height() <= 0.0 {
        return None;
    }
    ctx.set_transform(shift);
    for region in clip.regions() {
        match region {
            ClipRegion::Rect(rect) => ctx.push_clip_path(&rect.to_path(BEZIER_TOLERANCE)),
            ClipRegion::Shape { path, .. } => ctx.push_clip_path(path),
        }
    }
    Some(clip.regions().len())
}

/// Sets the context paint, converting and caching image brushes once.
pub(crate) fn target_render_profile() -> RenderProfile {
    RenderProfile {
        context: RenderSettings::default(),
        rasterizer: RasterizerSettings {
            render_mode: if cfg!(target_arch = "xtensa") {
                RenderMode::OptimizeQuality
            } else {
                RenderMode::OptimizeSpeed
            },
            ..RasterizerSettings::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compositor::BandScheduler;
    use kurbo::Rect;
    use peniko::{Color, ImageAlphaType, ImageBrush, ImageData, ImageFormat};

    /// Every command index, i.e. the un-indexed scan the band index replaces.
    fn all_candidates(list: &DisplayList) -> Vec<u32> {
        (0..u32::try_from(list.commands().len()).expect("test scenes stay small")).collect()
    }

    fn rasterize(list: &DisplayList, region: DeviceRegion) -> Pixmap {
        rasterize_with(&mut Painter::default(), list, region)
    }

    fn rasterize_with(painter: &mut Painter, list: &DisplayList, region: DeviceRegion) -> Pixmap {
        let mut work = FrameWork::ZERO;
        painter
            .rasterize_region(list, region, &all_candidates(list), &mut work)
            .clone()
    }

    fn checker_scene() -> DisplayList {
        let mut list = DisplayList::new();
        list.fill(
            &Rect::new(0.0, 0.0, 64.0, 64.0),
            Affine::IDENTITY,
            Color::from_rgb8(20, 40, 80),
        );
        list.fill(
            &Rect::new(8.5, 8.5, 31.5, 31.5),
            Affine::IDENTITY,
            Color::from_rgb8(220, 60, 40),
        );
        list.fill(
            &kurbo::Circle::new((44.0, 44.0), 14.0),
            Affine::IDENTITY,
            Color::from_rgb8(60, 200, 120),
        );
        list
    }

    fn image_scene() -> DisplayList {
        let image = ImageData {
            data: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ]
            .into(),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: 2,
            height: 2,
        };
        let mut list = DisplayList::new();
        list.fill(
            &Rect::new(0.0, 0.0, 2.0, 2.0),
            Affine::IDENTITY,
            peniko::Brush::Image(ImageBrush::new(image)),
        );
        list
    }

    #[test]
    fn full_region_renders_expected_colors() {
        let pixmap = rasterize(
            &checker_scene(),
            DeviceRegion {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        );
        let pixel = |x: usize, y: usize| {
            let i = (y * 64 + x) * 4;
            let data = pixmap.data_as_u8_slice();
            [data[i], data[i + 1], data[i + 2], data[i + 3]]
        };
        assert_eq!(pixel(2, 2), [20, 40, 80, 255]);
        assert_eq!(pixel(16, 16), [220, 60, 40, 255]);
        assert_eq!(pixel(44, 44), [60, 200, 120, 255]);
    }

    /// A retained clip must mask fills during rasterization: pixels inside
    /// the clip render, pixels outside stay untouched.
    #[test]
    fn clipped_command_renders_only_inside_the_clip() {
        let mut list = DisplayList::new();
        list.push_clip(Rect::new(0.0, 0.0, 32.0, 32.0));
        list.fill(
            &Rect::new(0.0, 0.0, 64.0, 64.0),
            Affine::IDENTITY,
            Color::from_rgb8(220, 60, 40),
        );
        list.pop_clip();
        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        );
        let pixel = |x: usize, y: usize| {
            let i = (y * 64 + x) * 4;
            let data = pixmap.data_as_u8_slice();
            [data[i], data[i + 1], data[i + 2], data[i + 3]]
        };
        assert_eq!(pixel(16, 16), [220, 60, 40, 255]);
        assert_eq!(pixel(48, 16), [0, 0, 0, 0]);
        assert_eq!(pixel(16, 48), [0, 0, 0, 0]);
    }

    /// Band-by-band rendering must be byte-identical to rendering the same
    /// area in one pass — otherwise band seams would be visible.
    #[test]
    fn banded_render_matches_single_pass() {
        let list = checker_scene();
        let full = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        );
        let scheduler = BandScheduler::new(64, 64, 16);
        let full_data = full.data_as_u8_slice();
        // One painter across all bands: same-size bands share one reset
        // scratch slot, so this also proves reuse renders like fresh state.
        let mut painter = Painter::default();
        for band in scheduler.schedule(&[Rect::new(0.0, 0.0, 64.0, 64.0)]) {
            let pixmap = rasterize_with(&mut painter, &list, band);
            let band_data = pixmap.data_as_u8_slice();
            for row in 0..band.height as usize {
                let band_row =
                    &band_data[row * band.width as usize * 4..(row + 1) * band.width as usize * 4];
                let full_start = ((band.y as usize + row) * 64 + band.x as usize) * 4;
                let full_row = &full_data[full_start..full_start + band.width as usize * 4];
                assert_eq!(band_row, full_row, "band seam mismatch at row {row}");
            }
        }
    }

    #[test]
    fn image_brush_matches_across_bands_and_is_converted_once() {
        let list = image_scene();
        let full = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
        );
        let mut painter = Painter::default();
        let top = rasterize_with(
            &mut painter,
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
            },
        );
        let bottom = rasterize_with(
            &mut painter,
            &list,
            DeviceRegion {
                x: 0,
                y: 1,
                width: 2,
                height: 1,
            },
        );

        assert_eq!(
            [top.data_as_u8_slice(), bottom.data_as_u8_slice()].concat(),
            full.data_as_u8_slice()
        );
        assert_eq!(painter.images.len(), 1);
        assert_eq!(
            full.data_as_u8_slice(),
            [
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ]
        );
    }

    use std::cell::RefCell;
    use std::rc::Rc;

    use waterui_graphics::SceneBackend;
    use waterui_graphics::draw::{self, Draw as _};

    /// A `Scene` command replaying `body` recorded at `size` against a fresh
    /// dew scene target — the direct-painter equivalent of what a
    /// `SceneView` installs. The `isolated` flag is computed the same way the
    /// node computes it: `blends_within` on the committed list.
    fn recorded_scene(size: kurbo::Size, body: impl FnOnce(&mut draw::Recorder)) -> DrawCommand {
        recorded_scene_at(SceneTarget::new(), size, body)
    }

    /// [`recorded_scene`] against a caller's `target` — for tests that
    /// register fonts or images into the recording.
    fn recorded_scene_at(
        target: Rc<SceneTarget>,
        size: kurbo::Size,
        body: impl FnOnce(&mut draw::Recorder),
    ) -> DrawCommand {
        let layout = draw::LayoutSize::new();
        layout.set(&draw::LayoutSize::change(size, None));
        let mut content = draw::Content::record(&layout, body);
        let list = content.snapshot();
        let isolated = draw::blends_within(list, 0..list.len());
        DrawCommand::Scene {
            scene: Scene::new(Rc::new(RefCell::new(Some(content))), 0, isolated, target),
            transform: Affine::IDENTITY,
            bounds: Rect::new(0.0, 0.0, size.width, size.height),
            clip: None,
        }
    }

    fn white() -> draw::WorkingColor {
        draw::WorkingColor::new([1.0, 1.0, 1.0, 1.0])
    }

    /// An opaque backdrop with a half-transparent group over it: the group
    /// is a real compositing layer, so the pixels under it are the blend of
    /// the two, not either one alone.
    fn layered_scene() -> DisplayList {
        let mut list = DisplayList::new();
        list.fill(
            &Rect::new(0.0, 0.0, 64.0, 64.0),
            Affine::IDENTITY,
            Color::from_rgb8(0, 0, 0),
        );
        list.push_placed(
            recorded_scene(kurbo::Size::new(64.0, 64.0), |recorder| {
                recorder.group(draw::Group::new().opacity(0.5), |recorder| {
                    recorder.fill(Rect::new(16.0, 16.0, 48.0, 48.0), white());
                });
            }),
            Rect::new(0.0, 0.0, 64.0, 64.0),
        );
        list
    }

    /// A scene's compositing layer is honoured, not flattened into a
    /// per-shape alpha. The assertion is dew's sRGB compositing — a 0.5
    /// white group over black lands near 128 — not the cherenkov
    /// reference's working-space contract, where it lands near 188 (see the
    /// `BeginGroup` arm of `SceneReplay` and the README's asymmetries).
    #[test]
    fn a_scene_layer_composites_its_opacity_in_dews_srgb() {
        let pixmap = rasterize(
            &layered_scene(),
            DeviceRegion {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        );
        let data = pixmap.data_as_u8_slice();
        let pixel = |x: usize, y: usize| {
            let i = (y * 64 + x) * 4;
            [data[i], data[i + 1], data[i + 2], data[i + 3]]
        };
        assert_eq!(pixel(4, 4), [0, 0, 0, 255], "outside the layer stays black");
        let inside = pixel(32, 32);
        assert_eq!(inside[3], 255);
        assert!(
            (120..=136).contains(&inside[0]),
            "half-opacity white over black composites to mid grey in sRGB, got {inside:?}"
        );
    }

    /// A blend group inside a scene composites as an isolated layer — a
    /// `DestOut` group erases the scene's own pixels only, never the dew
    /// commands the scene sits over.
    #[test]
    fn a_blend_group_isolates_from_dews_backdrop() {
        let widget = Color::from_rgb8(60, 120, 200);
        let mut list = DisplayList::new();
        list.fill(&Rect::new(0.0, 0.0, 32.0, 32.0), Affine::IDENTITY, widget);
        list.push_placed(
            recorded_scene(kurbo::Size::new(32.0, 32.0), |recorder| {
                // An opaque cover with a `DestOut` group punching a hole in
                // it: erased pixels show what the scene was placed over.
                recorder.fill(Rect::new(0.0, 0.0, 32.0, 32.0), white());
                recorder.group(
                    draw::Group::new().blend(draw::BlendMode::DestOut),
                    |recorder| {
                        recorder.fill(Rect::new(8.0, 8.0, 24.0, 24.0), white());
                    },
                );
            }),
            Rect::new(0.0, 0.0, 32.0, 32.0),
        );
        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
        );
        let data = pixmap.data_as_u8_slice();
        let pixel = |x: usize, y: usize| {
            let i = (y * 32 + x) * 4;
            [data[i], data[i + 1], data[i + 2], data[i + 3]]
        };
        assert_eq!(
            pixel(16, 16),
            [60, 120, 200, 255],
            "the `DestOut` hole shows the widget the scene sits over, not erased pixels"
        );
        assert_eq!(
            pixel(4, 4),
            [255, 255, 255, 255],
            "outside the hole the scene's cover composites normally"
        );
    }

    /// A scene rasterized band by band must match a single pass exactly —
    /// the replay's layers, shadows and pictures composite identically in
    /// every band, so no seam may appear where a band boundary crosses one.
    /// The shadow's blur box straddles the 16px band edges on purpose.
    #[test]
    fn banded_scene_render_matches_single_pass() {
        let mut list = DisplayList::new();
        list.fill(
            &Rect::new(0.0, 0.0, 64.0, 64.0),
            Affine::IDENTITY,
            Color::from_rgb8(24, 24, 32),
        );
        list.push_placed(
            recorded_scene(kurbo::Size::new(64.0, 64.0), |recorder| {
                recorder.fill(
                    Rect::new(8.0, 8.0, 56.0, 56.0),
                    draw::WorkingColor::new([0.2, 0.4, 0.8, 1.0]),
                );
                // A `DropShadowOnly` shadow whose blur and spread cross the
                // 16-px band edge at y=32.
                recorder.shadow(
                    Rect::new(16.0, 28.0, 48.0, 36.0),
                    draw::Shadow::new(4.0, white())
                        .offset(kurbo::Vec2::new(2.0, 3.0))
                        .spread(2.0),
                );
                // A shared picture placed twice, one copy straddling the
                // same band edge.
                let picture = draw::Picture::record(|picture| {
                    picture.fill(
                        kurbo::Circle::new((0.0, 0.0), 6.0),
                        draw::WorkingColor::new([0.9, 0.5, 0.1, 1.0]),
                    );
                });
                recorder.picture(&picture, Affine::translate((12.0, 32.0)));
                recorder.picture(&picture, Affine::translate((50.0, 50.0)));
                recorder.group(draw::Group::new().opacity(0.6), |recorder| {
                    recorder.fill(
                        Rect::new(40.0, 10.0, 58.0, 26.0),
                        draw::WorkingColor::new([0.8, 0.2, 0.2, 1.0]),
                    );
                });
            }),
            Rect::new(0.0, 0.0, 64.0, 64.0),
        );
        let full = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
        );
        let full_data = full.data_as_u8_slice();
        let scheduler = BandScheduler::new(64, 64, 16);
        let mut painter = Painter::default();
        for band in scheduler.schedule(&[Rect::new(0.0, 0.0, 64.0, 64.0)]) {
            let pixmap = rasterize_with(&mut painter, &list, band);
            let band_data = pixmap.data_as_u8_slice();
            for row in 0..band.height as usize {
                let band_row =
                    &band_data[row * band.width as usize * 4..(row + 1) * band.width as usize * 4];
                let full_start = ((band.y as usize + row) * 64 + band.x as usize) * 4;
                let full_row = &full_data[full_start..full_start + band.width as usize * 4];
                assert_eq!(band_row, full_row, "scene band seam mismatch at row {row}");
            }
        }
    }

    /// A scene must not leak its drawing state into the commands rasterized
    /// after it: the scene records an even-odd fill over a self-overlapping
    /// `BezPath` under a transformed paint — both rule and transform must be
    /// restored — and dew follows with a gradient-brushed fill over a path
    /// whose inner ring disagrees under the two rules: a leaked even-odd
    /// punches a hole in it, a leaked paint transform slides the gradient
    /// off its anchor.
    #[test]
    fn a_scene_does_not_leak_its_state_into_later_commands() {
        let mut winding = BezPath::new();
        winding.extend(Rect::new(0.0, 0.0, 4.0, 4.0).path_elements(BEZIER_TOLERANCE));
        winding.extend(Rect::new(1.0, 1.0, 3.0, 3.0).path_elements(BEZIER_TOLERANCE));
        let mut list = DisplayList::new();
        list.push_placed(
            recorded_scene(kurbo::Size::new(4.0, 4.0), move |recorder| {
                recorder.fill(
                    draw::EvenOdd(winding),
                    draw::Paint::Transformed(draw::TransformedPaint::new(
                        draw::WorkingColor::new([0.04, 0.04, 0.04, 1.0]),
                        Affine::translate((7.0, 3.0)),
                    )),
                );
            }),
            Rect::new(0.0, 0.0, 4.0, 4.0),
        );
        let mut overlapping = BezPath::new();
        overlapping.extend(Rect::new(0.0, 0.0, 32.0, 32.0).path_elements(BEZIER_TOLERANCE));
        overlapping.extend(Rect::new(8.0, 8.0, 24.0, 24.0).path_elements(BEZIER_TOLERANCE));
        // The dew fill's brush is a real gradient: a leaked paint transform
        // would move it, so the midpoint checks the transform restore too.
        let stops = [
            peniko::ColorStop {
                offset: 0.0,
                color: Color::from_rgb8(255, 0, 0).into(),
            },
            peniko::ColorStop {
                offset: 1.0,
                color: Color::from_rgb8(0, 0, 255).into(),
            },
        ];
        let gradient = peniko::Gradient::new_linear((0.0, 0.0), (32.0, 0.0)).with_stops(&stops[..]);
        list.fill(
            &overlapping,
            Affine::IDENTITY,
            peniko::Brush::Gradient(gradient),
        );

        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
        );
        let data = pixmap.data_as_u8_slice();
        let pixel = |x: usize, y: usize| {
            let i = (y * 32 + x) * 4;
            [data[i], data[i + 1], data[i + 2], data[i + 3]]
        };
        let center = pixel(16, 16);
        assert!(
            center[3] == 255 && center[0] > 90 && center[0] < 166 && center[2] > 90,
            "a leaked even-odd rule would punch a hole at the overlap center, \
             a leaked paint transform would slide the gradient, got {center:?}"
        );
    }

    /// A glyph's own transform composes *after* its pen position — glifo
    /// draws `run_transform * pen * glyph_transform` — so the cull box hangs
    /// under the glyph transform, not around the pen. A Roboto 'H' rotated
    /// 90° about its origin at (44, 44) spans forward: its ink lands in the
    /// band at y ≥ 48 that the same glyph unrotated — ink in `y ∈ [22, 44]`
    /// — is culled from entirely.
    #[test]
    fn a_rotated_glyph_is_not_culled_from_a_band_edge() {
        let target = SceneTarget::new();
        let font = target
            .register_font(waterui_graphics::FontSource::Bytes {
                data: crate::test_font_files::read_test_font("Roboto-Regular.ttf").into(),
                index: 0,
            })
            .expect("the bundled Roboto registers");
        let run = |transform: Option<Affine>| draw::GlyphRun {
            font: font.id(),
            size: 32.0,
            coords: Vec::new().into(),
            glyphs: vec![draw::Glyph {
                id: 40, // 'H' in Roboto — tall and narrow
                x: 44.0,
                y: 44.0,
                transform,
            }]
            .into(),
            style: draw::GlyphStyle::Fill,
        };
        let scene = |transform| {
            recorded_scene_at(
                target.clone(),
                kurbo::Size::new(64.0, 64.0),
                move |recorder| recorder.glyphs(run(transform), white()),
            )
        };
        let bottom_band = DeviceRegion {
            x: 0,
            y: 48,
            width: 64,
            height: 16,
        };
        let ink_in_band = |scene_command: &DrawCommand| {
            let mut list = DisplayList::new();
            list.push_placed(scene_command.clone(), Rect::new(0.0, 0.0, 64.0, 64.0));
            let pixmap = rasterize(&list, bottom_band);
            pixmap.data_as_u8_slice().iter().any(|byte| *byte != 0)
        };
        assert!(
            !ink_in_band(&scene(None)),
            "an upright 'H' at pen (44, 44) draws nothing below y=48 — the cull holds"
        );
        assert!(
            ink_in_band(&scene(Some(Affine::rotate(std::f64::consts::FRAC_PI_2)))),
            "the same 'H' rotated 90° spans forward into y ∈ [48, 64] and must not be culled"
        );
    }

    /// The `head` box is mirrored into screen space at registration —
    /// glifo flips outlines — so an upright glyph is not culled from the
    /// band above its baseline. A Roboto 'H' at size 30 with its pen at
    /// y = 43 has a cap height of ≈21.4 px: its top sits in y ∈ [16, 32),
    /// one band above the baseline's own. A cull box kept in y-up font
    /// space would answer for the box below the baseline instead and cull
    /// the top outright — a loud fail, so the probe is only placed where a
    /// correct box must draw: one band up, well inside the cap.
    #[test]
    fn an_upright_glyph_is_not_culled_from_the_band_above_its_baseline() {
        let target = SceneTarget::new();
        let font = target
            .register_font(waterui_graphics::FontSource::Bytes {
                data: crate::test_font_files::read_test_font("Roboto-Regular.ttf").into(),
                index: 0,
            })
            .expect("the bundled Roboto registers");
        let run = draw::GlyphRun {
            font: font.id(),
            size: 30.0,
            coords: Vec::new().into(),
            glyphs: vec![draw::Glyph {
                id: 40, // 'H' in Roboto
                x: 8.0,
                y: 43.0,
                transform: None,
            }]
            .into(),
            style: draw::GlyphStyle::Fill,
        };
        let scene = recorded_scene_at(target, kurbo::Size::new(64.0, 64.0), move |recorder| {
            recorder.glyphs(run, white());
        });
        let top_band = DeviceRegion {
            x: 0,
            y: 16,
            width: 64,
            height: 16,
        };
        let mut list = DisplayList::new();
        list.push_placed(scene, Rect::new(0.0, 0.0, 64.0, 64.0));
        let pixmap = rasterize(&list, top_band);
        assert!(
            pixmap.data_as_u8_slice().iter().any(|byte| *byte != 0),
            "the 'H' cap-top crosses into y ∈ [16, 32); a y-up cull box \
             sits below the baseline and culls the cap"
        );
    }

    /// A negative spread shrinks the silhouette: at exactly minus the
    /// half-extent it is empty and casts nothing — never a phantom shadow
    /// and never a panic — and a smaller negative spread still lands its
    /// offset blur.
    #[test]
    fn a_negative_spread_shrinks_to_nothing_without_a_phantom_shadow() {
        let shadow_scene = |spread: f64| {
            recorded_scene(kurbo::Size::new(48.0, 48.0), move |recorder| {
                recorder.shadow(
                    Rect::new(8.0, 8.0, 24.0, 24.0),
                    draw::Shadow::new(1.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                        .offset(kurbo::Vec2::new(4.0, 0.0))
                        .spread(spread),
                );
            })
        };
        let region = DeviceRegion {
            x: 0,
            y: 0,
            width: 48,
            height: 48,
        };
        let pixel = |pixmap: &Pixmap, x: usize, y: usize| {
            let i = (y * 48 + x) * 4;
            pixmap.data_as_u8_slice()[i]
        };
        // -8 collapses the 16×16 shape to an empty rect: the band shows the
        // backdrop and nothing else.
        let mut backdrop_only = DisplayList::new();
        backdrop_only.fill(
            &Rect::new(0.0, 0.0, 48.0, 48.0),
            Affine::IDENTITY,
            Color::from_rgb8(40, 80, 160),
        );
        backdrop_only.push_placed(shadow_scene(-8.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let pixmap = rasterize(&backdrop_only, region);
        assert!(
            pixmap
                .data_as_u8_slice()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px == &[40, 80, 160, 255]),
            "a collapsed silhouette casts no shadow"
        );
        // -4 leaves an 8×8 silhouette offset to (16,12)-(24,20): its black
        // shadow darkens a pixel inside it.
        let mut shrunken = DisplayList::new();
        shrunken.fill(
            &Rect::new(0.0, 0.0, 48.0, 48.0),
            Affine::IDENTITY,
            Color::from_rgb8(40, 80, 160),
        );
        shrunken.push_placed(shadow_scene(-4.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let pixmap = rasterize(&shrunken, region);
        assert!(
            pixel(&pixmap, 20, 16) != 40,
            "the shrunken silhouette still casts its offset shadow"
        );
    }

    /// Positive spread on a general path is exact too — the fill plus a
    /// `2s`-wide round outline inside the filter layer — so a shadow spreads
    /// past the shape's own edge. The probe sits 4 px below the triangle's
    /// left vertex: inside a +6 silhouette's reach and outside a sigma-0
    /// unspread silhouette entirely.
    #[test]
    fn a_spread_shadow_on_a_path_covers_past_the_shape_edge() {
        let triangle = |spread: f64| {
            recorded_scene(kurbo::Size::new(48.0, 48.0), move |recorder| {
                let mut path = BezPath::new();
                path.move_to((8.0, 8.0));
                path.line_to((40.0, 24.0));
                path.line_to((8.0, 40.0));
                path.close_path();
                recorder.shadow(
                    path,
                    draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                        .spread(spread),
                );
            })
        };
        let region = DeviceRegion {
            x: 0,
            y: 0,
            width: 48,
            height: 48,
        };
        let mut unspread = DisplayList::new();
        unspread.push_placed(triangle(0.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let unspread = rasterize(&unspread, region);
        let mut spread = DisplayList::new();
        spread.push_placed(triangle(6.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let spread = rasterize(&spread, region);
        let below = |pixmap: &Pixmap| {
            let i = (44 * 48 + 8) * 4;
            pixmap.data_as_u8_slice()[i + 3]
        };
        assert!(
            below(&spread) > below(&unspread),
            "a +6 spread must throw shadow below the path's own edge"
        );
    }

    /// Negative spread on a non-rect shape subtracts a `2s`-wide round
    /// outline through `DestOut` inside the filter layer. An ellipse with
    /// `-4` shows the shrunk silhouette's offset shadow, and `0` spread
    /// against `-4` proves the rim outside the shrunk edge carries nothing
    /// — no phantom halo from the stroke the subtraction must leave behind.
    #[test]
    fn a_negative_spread_on_an_ellipse_shrinks_without_a_phantom() {
        let scene = |spread: f64| {
            recorded_scene(kurbo::Size::new(48.0, 48.0), move |recorder| {
                recorder.shadow(
                    kurbo::Ellipse::new((24.0, 24.0), (16.0, 10.0), 0.0),
                    draw::Shadow::new(1.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                        .offset(kurbo::Vec2::new(4.0, 0.0))
                        .spread(spread),
                );
            })
        };
        let region = DeviceRegion {
            x: 0,
            y: 0,
            width: 48,
            height: 48,
        };
        // The backdrop's red is 40 and the shadow is black: a shadowed
        // pixel darkens toward 0, so the probe is the red channel.
        let red = |pixmap: &Pixmap, x: usize, y: usize| pixmap.data_as_u8_slice()[(y * 48 + x) * 4];
        let mut backdrop = DisplayList::new();
        backdrop.fill(
            &Rect::new(0.0, 0.0, 48.0, 48.0),
            Affine::IDENTITY,
            Color::from_rgb8(40, 80, 160),
        );
        let mut unspread = backdrop.clone();
        unspread.push_placed(scene(0.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let unspread = rasterize(&unspread, region);
        let mut shrunk = backdrop;
        shrunk.push_placed(scene(-4.0), Rect::new(0.0, 0.0, 48.0, 48.0));
        let shrunk = rasterize(&shrunk, region);

        // Offset by (4, 0), the unspread silhouette covers x ∈ [12, 44];
        // spread -4 shrinks the silhouette to x ∈ [16, 40]. Inside it, the
        // shadow still lands; on the rim it receded from, a phantom halo —
        // the un-subtracted outline — would sit.
        assert!(
            red(&shrunk, 30, 24) < 30,
            "the shrunk silhouette still casts its offset shadow"
        );
        assert!(
            red(&unspread, 14, 24) < red(&shrunk, 14, 24),
            "the rim the spread receded from is dark under spread 0 and \
             backdrop under spread -4 — the shrink applied"
        );
        assert!(
            red(&shrunk, 14, 24) >= 36,
            "no phantom halo: outside the shrunk silhouette nothing drew"
        );
    }

    /// `vello_cpu` snapshots the context transform when a filter layer is
    /// pushed, and the shadow's offset and sigma are already converted to
    /// device units — the layer must go up under identity or the leftover
    /// transform scales them a second time. A fill inside `scale(2)` is
    /// what leaves that transform armed; the shadow's offset must move
    /// the silhouette by one application, not two.
    #[test]
    fn a_shadows_device_offset_is_not_scaled_twice() {
        let scene = recorded_scene(kurbo::Size::new(160.0, 160.0), |recorder| {
            recorder.transform(Affine::scale(2.0), |recorder| {
                recorder.fill(Rect::new(4.0, 4.0, 12.0, 12.0), white());
                recorder.shadow(
                    Rect::new(40.0, 40.0, 60.0, 60.0),
                    draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                        .offset(kurbo::Vec2::new(4.0, 0.0)),
                );
            });
        });
        // The rect lands at (80, 80)-(120, 120); the (4, 0) offset under
        // scale 2 is (8, 0) — the silhouette covers x ∈ [88, 128]. A second
        // application of the scale would push it to x ∈ [96, 136].
        let mut list = DisplayList::new();
        list.push_placed(scene, Rect::new(0.0, 0.0, 160.0, 160.0));
        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 160,
                height: 160,
            },
        );
        let alpha = |x: usize, y: usize| pixmap.data_as_u8_slice()[(y * 160 + x) * 4 + 3];
        assert!(
            alpha(92, 100) > 200,
            "the offset shadow reaches x=92: the scale applied once"
        );
        assert!(
            alpha(134, 100) < 50,
            "nothing past x=128: a second application would draw there"
        );
    }

    /// The same push-time transform snapshot under a 90-degree rotation:
    /// the offset vector would turn twice, moving the shadow sideways
    /// instead of forward.
    #[test]
    fn a_shadows_device_offset_is_not_rotated_twice() {
        let scene = recorded_scene(kurbo::Size::new(160.0, 160.0), |recorder| {
            recorder.transform(
                Affine::translate((80.0, 20.0)) * Affine::rotate(std::f64::consts::FRAC_PI_2),
                |recorder| {
                    recorder.fill(Rect::new(0.0, 0.0, 4.0, 4.0), white());
                    recorder.shadow(
                        Rect::new(40.0, 40.0, 60.0, 60.0),
                        draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                            .offset(kurbo::Vec2::new(4.0, 0.0)),
                    );
                },
            );
        });
        // The rect lands at x ∈ [20, 40], y ∈ [60, 80]; the once-rotated
        // offset (4, 0) → (0, 4) shifts it down to y ∈ [64, 84]. Applied
        // twice the offset becomes (-4, 0): x ∈ [16, 36], y ∈ [60, 80].
        let mut list = DisplayList::new();
        list.push_placed(scene, Rect::new(0.0, 0.0, 160.0, 160.0));
        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 160,
                height: 160,
            },
        );
        let alpha = |x: usize, y: usize| pixmap.data_as_u8_slice()[(y * 160 + x) * 4 + 3];
        assert!(
            alpha(38, 82) > 200,
            "the offset shadow covers (38, 82): the rotation applied once"
        );
        assert!(
            alpha(18, 70) < 50,
            "nothing at (18, 70): a second rotation would put it there"
        );
    }

    /// The closed-form silhouette is chosen before spread is tested, so a
    /// rect-family shadow under a non-uniform axis-aligned scale keeps
    /// cherenkov-cpu's circular corners — `radius * smax` — at any spread
    /// including zero, and nothing jumps across `spread == 0`.
    #[test]
    fn an_axis_aligned_shadow_keeps_circular_corners_at_spread_zero() {
        let scene = |spread: f64| {
            recorded_scene(kurbo::Size::new(200.0, 40.0), move |recorder| {
                recorder.transform(Affine::scale_non_uniform(2.0, 1.0), |recorder| {
                    recorder.shadow(
                        RoundedRect::from_rect(
                            Rect::new(10.0, 10.0, 30.0, 20.0),
                            RoundedRectRadii::new(4.0, 4.0, 4.0, 4.0),
                        ),
                        draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0]))
                            .spread(spread),
                    );
                });
            })
        };
        let region = DeviceRegion {
            x: 0,
            y: 0,
            width: 200,
            height: 40,
        };
        let alpha_at =
            |pixmap: &Pixmap, x: usize, y: usize| pixmap.data_as_u8_slice()[(y * 200 + x) * 4 + 3];
        // The device box is (20, 10)-(60, 20); `radius * smax` = 8 clamps
        // to the half-extent limit of 5, so the left end is the circle
        // centred (25, 15) radius 5 — a stadium cap. Pixel (25, 19) sits
        // under both candidates: the (8, 4) ellipse corner an unspread
        // transformed path would draw covers about two thirds of it where
        // the circle covers it fully.
        let zero = rasterize(
            &{
                let mut list = DisplayList::new();
                list.push_placed(scene(0.0), Rect::new(0.0, 0.0, 200.0, 40.0));
                list
            },
            region,
        );
        assert!(
            alpha_at(&zero, 25, 19) > 230,
            "the cap is a circle of radius `r * smax`, not a scaled ellipse"
        );
        // An almost-zero spread takes the same branch and lands the same
        // coverage — no discontinuity at spread 0: the two pixmaps differ
        // by at most a few units per channel.
        let tiny = rasterize(
            &{
                let mut list = DisplayList::new();
                list.push_placed(scene(0.001), Rect::new(0.0, 0.0, 200.0, 40.0));
                list
            },
            region,
        );
        assert!(
            zero.data_as_u8_slice()
                .iter()
                .zip(tiny.data_as_u8_slice())
                .all(|(a, b)| a.abs_diff(*b) <= 8),
            "spread 0.001 takes the same closed form — no jump at spread 0"
        );
    }

    /// A fill closes its subpaths implicitly but a stroke does not run the
    /// closing edge — the dilation stroke must, or a spread shadow misses
    /// the sliver the closing edge covers. That holds for a subpath left
    /// open and for the implicit subpath a segment after `ClosePath` with
    /// no `MoveTo` begins at the previous start point.
    #[test]
    fn a_spread_shadow_strokes_the_closing_edge_of_an_open_subpath() {
        let scene = recorded_scene(kurbo::Size::new(48.0, 48.0), |recorder| {
            let mut path = BezPath::new();
            path.move_to((8.0, 8.0));
            path.line_to((40.0, 8.0));
            path.line_to((40.0, 40.0));
            // Deliberately open: the fill's implicit closing edge runs
            // (40, 40) → (8, 8).
            recorder.shadow(
                path,
                draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(3.0),
            );
            let mut implicit = BezPath::new();
            implicit.move_to((4.0, 40.0));
            implicit.line_to((8.0, 40.0));
            implicit.close_path();
            // Segments after `ClosePath` with no `MoveTo` open an implicit
            // subpath at (4, 40): its closing edge runs (44, 47) → (4, 40).
            implicit.line_to((44.0, 36.0));
            implicit.line_to((44.0, 47.0));
            recorder.shadow(
                implicit,
                draw::Shadow::new(0.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(3.0),
            );
        });
        // (24.5, 27.5) is outside the triangle, 2.1 px from the closing
        // diagonal — inside its 3-px dilation band — and over 15 px from
        // either real edge. (24.5, 45.5) is outside the implicit
        // subpath's fill, 1.9 px below its closing edge — inside the
        // 3-px dilation band — and over 5 px from every real edge.
        let mut list = DisplayList::new();
        list.push_placed(scene, Rect::new(0.0, 0.0, 48.0, 48.0));
        let pixmap = rasterize(
            &list,
            DeviceRegion {
                x: 0,
                y: 0,
                width: 48,
                height: 48,
            },
        );
        let alpha_at = |x: usize, y: usize| pixmap.data_as_u8_slice()[(y * 48 + x) * 4 + 3];
        assert!(
            alpha_at(24, 27) > 0,
            "the closing edge dilates like every other edge of the outline"
        );
        assert!(
            alpha_at(24, 45) > 0,
            "the implicit subpath's closing edge dilates too"
        );
    }
}
