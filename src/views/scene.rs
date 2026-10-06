//! Self-drawn scene content: `Canvas` drawings and SVG documents.
//!
//! A `SceneView` hands the backend a [`SceneContent`] that records its
//! drawing into a cherenkov [`Recorder`]. Dew installs
//! `SceneViewMergeToParent` (see [`crate::dispatch::DewRenderer::render_tree`]),
//! so the content arrives here rather than falling back to a GPU surface dew
//! has no way to create.
//!
//! Dew's realization is the recording layer alone: `build_scene` records a
//! `cherenkov_record` [`Content`] against dew's own [`SceneBackend`], and the
//! painter replays the recorded commands band by band into `vello_cpu` — no
//! offscreen surface, no readback, no bitmap. The recording is re-run only
//! when the content invalidates itself, when the box it was built for
//! resizes, or when it asked for another frame; bound signals and animated
//! operands inside it repaint through [`Content::take_change`] and
//! [`Content::sample`] without re-running `build_scene`.
//!
//! Scene content draws inside the box it was given: the command is emitted
//! under a clip of exactly that box, which is also what the `GpuSurface`
//! realization of the same content does by rendering into a texture of that
//! size.

use core::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use accesskit::{Node as AccessibilityNode, NodeId, Role};
use kurbo::{Affine, Rect};
use skrifa::{Tag, raw::TableProvider as _};
use waterui_backend_core::frame_signals::FrameSignals;
use waterui_core::layout::{ProposalSize, Size, StretchAxis, ViewDimensions};
use waterui_graphics::draw::{
    self, Animating, Content, ContentChange, ContentSpare, FontId, ImageId, ImageLimits,
    LayoutSize, LiveOwner, SampleFlag,
};
use waterui_graphics::{
    FontSource, Handle, HeldResources, ImageColorSpace, ImageData, RecordingResources,
    ResourceError, ResourceHandle, Rgba8, Rgba16F, SceneBackend, SceneContent, SceneResources,
    SceneView, resolve_scene_proposal, scene_stretch_axis,
};

use crate::dispatch::{DewNode, DewRenderer, RenderContext};
use crate::display_list::{DrawCommand, Scene};
use crate::text::DewState;

/// The largest image the target registers. No board configuration supplies
/// a heap budget today, so the cap comes from dew's contract — a heap of a
/// few hundred KiB shared by the bands and the retained tree: 65 536 texels
/// is 256 KiB of RGBA8, enough for a 256-px atlas or a partial-screen photo
/// without starving the rest, and a larger upload could not fit the budget
/// no matter what it drew.
const IMAGE_LIMITS: ImageLimits = ImageLimits {
    max_dimension: 2048,
    max_texels: 1 << 16,
};

/// A registered font: the `peniko` font data glyph runs draw with and the
/// font's `head`-table bounds as an em-square box — the painter's per-band
/// glyph cull box scaled by run size.
#[derive(Clone, Debug)]
pub struct SceneFont {
    /// The font data a `vello_cpu` glyph run takes.
    pub font: peniko::FontData,
    /// `head`'s `(x_min, -y_max, x_max, -y_min)` divided by `units_per_em`
    /// — mirrored into screen space, where glifo draws the outlines — the
    /// box every glyph's outline stays inside, in ems around its origin.
    pub em_bbox: Rect,
    /// Whether `em_bbox` bounds every glyph the font can draw: `head`
    /// covers the outline strikes only, so a font carrying bitmap
    /// (`CBDT`/`EBDT`/`sbix`) or `COLR` glyphs marks `false` and its runs
    /// are never culled on the box.
    pub outline_only: bool,
}

/// Registered scene resources by raw id: the fonts glyph runs draw with and
/// the `vello_cpu` image sources image commands and image paints resolve.
#[derive(Debug, Default)]
struct SceneStores {
    fonts: HashMap<u64, SceneFont>,
    images: HashMap<u64, vello_cpu::ImageSource>,
}

#[derive(Debug)]
struct SceneTargetInner {
    stores: RefCell<SceneStores>,
    /// Monotonic raw-id source; every registration owns its id.
    next: Cell<u64>,
}

/// Dew's scene target: the [`SceneBackend`] [`SceneResources`] registers
/// scene fonts and images against, and the table the painter resolves the
/// recording's ids through.
///
/// A registration lives exactly as long as its handle: `Table` keeps only a
/// weak entry, so when the last `Registered` or recording holding it drops,
/// the dew-side handle's `Drop` removes the table entry — the registration
/// cannot outlive what names it.
#[derive(Debug)]
pub struct SceneTarget {
    inner: Rc<SceneTargetInner>,
}

/// `peniko::Blob` wants `Arc<dyn AsRef<[u8]>>`, which an unsized `Arc<[u8]>`
/// cannot cast into — this sized shell shares the same texels instead of
/// copying them into a `Vec`.
struct ImageBytes(Arc<[u8]>);

impl AsRef<[u8]> for ImageBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A font registration: drops the table entry when the last holder lets go.
#[derive(Debug)]
struct DewFont {
    id: u64,
    inner: Weak<SceneTargetInner>,
}

impl Drop for DewFont {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.stores.borrow_mut().fonts.remove(&self.id);
        }
    }
}

impl ResourceHandle for DewFont {
    type Id = FontId;

    fn id(&self) -> FontId {
        FontId::new(self.id)
    }
}

/// An image registration: drops the table entry when the last holder lets go.
#[derive(Debug)]
struct DewImage {
    id: u64,
    inner: Weak<SceneTargetInner>,
}

impl Drop for DewImage {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.stores.borrow_mut().images.remove(&self.id);
        }
    }
}

impl ResourceHandle for DewImage {
    type Id = ImageId;

    fn id(&self) -> ImageId {
        ImageId::new(self.id)
    }
}

impl SceneTarget {
    /// A target with an empty registration table.
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(Self {
            inner: Rc::new(SceneTargetInner {
                stores: RefCell::new(SceneStores::default()),
                next: Cell::new(0),
            }),
        })
    }

    fn next_id(&self) -> u64 {
        self.inner.next.replace(self.inner.next.get() + 1)
    }

    /// The font registered under `id`, for a glyph run the painter replays.
    ///
    /// A `ResourceError` never reaches here: a `Scene` command only names a
    /// registered id from a recording that holds the registration alive.
    ///
    /// # Panics
    ///
    /// Panics when `id` names nothing — a malformed recording, since a live
    /// registration always keeps its entry.
    #[must_use]
    pub(crate) fn font(&self, id: FontId) -> SceneFont {
        self.inner.stores.borrow().fonts[&id.raw()].clone()
    }

    /// The image source registered under `id`.
    ///
    /// # Panics
    ///
    /// Panics when `id` names nothing — see [`SceneTarget::font`].
    #[must_use]
    pub fn image(&self, id: ImageId) -> vello_cpu::ImageSource {
        self.inner.stores.borrow().images[&id.raw()].clone()
    }
}

impl SceneBackend for SceneTarget {
    fn register_font(&self, source: FontSource) -> Result<Handle<FontId>, ResourceError> {
        match source {
            FontSource::System { family } => Err(ResourceError::Font(format!(
                "system font '{family}' is not registered: dew has no platform font stack"
            ))),
            FontSource::Bytes { data, index } => {
                let font = skrifa::FontRef::from_index(&data, index).map_err(|error| {
                    ResourceError::Font(format!("unparsable font data: {error}"))
                })?;
                // The cull box comes from `head`: the union of every glyph's
                // outline in the font's default instance — no glyph can paint
                // outside it, so a miss is a real miss. `head` is y-up font
                // space; glifo flips outlines into screen space
                // (`glyph_transform * FLIP_Y * outline`), so the stored box
                // is mirrored across the baseline.
                let head = font.head().map_err(|error| {
                    ResourceError::Font(format!("font has no head table: {error}"))
                })?;
                let upem = f64::from(head.units_per_em());
                assert!(
                    upem > 0.0,
                    "a font with `units_per_em` of 0 cannot form an em box"
                );
                let em_bbox = Rect::new(
                    f64::from(head.x_min()) / upem,
                    -f64::from(head.y_max()) / upem,
                    f64::from(head.x_max()) / upem,
                    -f64::from(head.y_min()) / upem,
                );
                // `head` bounds the outline strikes only — bitmap
                // (`CBDT`/`EBDT`/`sbix`) and `COLR` glyphs can paint
                // outside it, so a font carrying any of them never
                // culls on the box. Presence is the test, not parse
                // success: an unreadable table still means glyphs can
                // paint outside the box.
                let outline_only = font.table_data(Tag::new(b"CBDT")).is_none()
                    && font.table_data(Tag::new(b"EBDT")).is_none()
                    && font.table_data(Tag::new(b"sbix")).is_none()
                    && font.table_data(Tag::new(b"COLR")).is_none();
                let font = SceneFont {
                    font: peniko::FontData::new(peniko::Blob::new(Arc::new(data)), index),
                    em_bbox,
                    outline_only,
                };
                let id = self.next_id();
                self.inner.stores.borrow_mut().fonts.insert(id, font);
                Ok(Handle::new(DewFont {
                    id,
                    inner: Rc::downgrade(&self.inner),
                }))
            }
        }
    }

    fn register_rgba8(&self, data: ImageData<Rgba8>) -> Result<Handle<ImageId>, ResourceError> {
        if data.color_space != ImageColorSpace::Srgb {
            return Err(ResourceError::Image(format!(
                "{:?} image data is not registered: dew's bands are sRGB8",
                data.color_space
            )));
        }
        let image = peniko::ImageData {
            // The registration already shares its texels by `Arc`; the
            // sized shell is what the unsized `Arc<[u8]>` needs to cast
            // into `Arc<dyn AsRef<[u8]>>` — vello's premultiplied
            // conversion below is the only copy dew pays for an image.
            data: peniko::Blob::new(Arc::new(ImageBytes(data.data().clone()))),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: if data.premultiplied {
                peniko::ImageAlphaType::AlphaPremultiplied
            } else {
                peniko::ImageAlphaType::Alpha
            },
            width: data.width(),
            height: data.height(),
        };
        let source = vello_cpu::ImageSource::from_peniko_image_data(&image);
        let id = self.next_id();
        self.inner.stores.borrow_mut().images.insert(id, source);
        Ok(Handle::new(DewImage {
            id,
            inner: Rc::downgrade(&self.inner),
        }))
    }

    fn register_rgba16f(
        &self,
        _data: ImageData<Rgba16F>,
    ) -> Result<Handle<ImageId>, ResourceError> {
        Err(ResourceError::Unsupported(
            "HDR image data on dew's sRGB8 bands",
        ))
    }

    fn image_limits(&self) -> ImageLimits {
        IMAGE_LIMITS
    }
}

/// The shared recording pipeline every scene on a renderer registers
/// against: dew's [`SceneTarget`] and the [`SceneResources`] table over it.
///
/// One per renderer rather than one per scene so identical resources
/// deduplicate across scenes — two canvases drawing the same font hold one
/// registration. Created lazily on the first `SceneView`'s first frame.
pub struct ScenePipeline {
    /// The resource target; `Scene` commands carry it so the painter
    /// resolves the recording's ids.
    pub target: Rc<SceneTarget>,
    resources: SceneResources,
}

impl ScenePipeline {
    /// Builds the resource table over a fresh dew target.
    #[must_use]
    pub fn new() -> Self {
        let target = SceneTarget::new();
        Self {
            resources: SceneResources::new(target.clone()),
            target,
        }
    }

    /// The registration borrow a `build_scene` call draws against.
    fn recording(&self) -> RecordingResources<'_> {
        self.resources.recording()
    }
}

/// The `LiveOwner` an installed scene's recording notifies: a bound signal
/// queueing a change wakes the frame pump for a scene the current frame
/// drew, and applies it at once — no refresh — for a hidden scene, per the
/// `LiveOwner` contract. An animated operand behaves differently: its
/// change queues in the recording's animation slot, not `pending`, so a
/// hidden scene does not sample it — `take_change` finds nothing until the
/// scene is shown and its animation runs.
#[derive(Debug)]
struct SceneOwner {
    signals: FrameSignals,
    /// The installed content — the node's one `Rc` for its whole life — for
    /// a hidden scene's apply-at-once. The slot is `None` while a re-record
    /// or an empty box has no content installed; a `changed` arriving then
    /// has nothing to drain.
    content: RefCell<Weak<RefCell<Option<Content>>>>,
    /// The renderer's frame counter.
    frame: Rc<Cell<u64>>,
    /// The frame this scene last emitted a command in; a change for a scene
    /// the current frame does not draw applies at once instead of waking a
    /// refresh — the frame stamp is what keeps that true when the node is
    /// retained but not rendered (a tab page that is not selected).
    emitted: Cell<u64>,
    /// The node's change marker, shared so the hidden path's apply bumps it.
    generation: Rc<Cell<u64>>,
    /// The node's isolation flag, shared so the hidden apply recomputes it:
    /// an update can write a `BeginGroup`'s blend operand too.
    isolated: Rc<Cell<bool>>,
    /// The name panics about unrecordable updates cite: the accessibility
    /// label the node last published, refreshed only on frames where
    /// accessibility is enabled — `None` when it never ran — and read only
    /// on the hidden-apply panic path.
    name: Rc<RefCell<Option<String>>>,
    /// The scene's node id — the fallback name when no label was published.
    accessibility_id: NodeId,
}

impl LiveOwner for SceneOwner {
    fn changed(&self) {
        if self.emitted.get() == self.frame.get() {
            self.signals.request_refresh();
        } else if let Some(content) = self.content.borrow().upgrade() {
            // Hidden: drain the change into the recording now — `take_change`
            // applies it in place and bumps the shared generation so the
            // next frame that draws the scene diffs correctly — and no
            // refresh is scheduled for content nothing is showing.
            let (finding, blends) = {
                let mut slot = content.borrow_mut();
                let Some(content) = slot.as_mut() else {
                    return;
                };
                match content.take_change() {
                    Some(ContentChange::Update(updates)) => {
                        // The same refusal `drain_live` runs: an update can
                        // write whatever operand it likes — a mesh paint or
                        // an `Extend::None` arriving while the scene is
                        // hidden is refused here, never allowed through to
                        // a band's `unreachable!`.
                        self.generation.set(self.generation.get() + 1);
                        let list = content.snapshot();
                        (
                            updates
                                .iter()
                                .find_map(|update| invalid_update(update, list)),
                            Some(draw::blends_within(list, 0..list.len())),
                        )
                    }
                    Some(ContentChange::Replace(_)) => {
                        unreachable!("the install consumed the first Replace")
                    }
                    None => (None, None),
                }
            };
            if let Some(blends) = blends {
                self.isolated.set(blends);
            }
            if let Some(finding) = finding {
                // `name` is the node's last published label; the recording
                // borrow is already released.
                let name = self
                    .name
                    .borrow()
                    .clone()
                    .unwrap_or_else(|| format!("scene node {:?}", self.accessibility_id));
                panic!("{name} recorded {finding}");
            }
        }
    }
}

/// An installed recording and everything it currently answers with.
struct Installed {
    /// The live recording: owns the bound-signal subscriptions, and is the
    /// single owner of the display list — the one `Rc` the node keeps for
    /// its whole life, shared with `DrawCommand::Scene`, which replays
    /// `Content::view` read-only. A re-record takes the old `Content` out
    /// under `borrow_mut` and installs the new one into the same slot — the
    /// `Option` means no placeholder recording is ever built. `None` is a
    /// scene whose box is empty or whose re-record is still running.
    ///
    /// Note the spare does not recycle the display list itself:
    /// `Content::retire` drops the `Picture` and keeps only `live`, so a
    /// re-record allocates a fresh list. Reclaiming the buffers needs a
    /// cherenkov-record API that hands a uniquely-held `Picture`'s storage
    /// into the `ContentSpare` — it does not exist yet.
    content: Rc<RefCell<Option<Content>>>,
    /// The registrations the recording names, kept alive while it is
    /// installed — dropping them early would free ids the list still
    /// references; its value is its `Drop`, which `record` holds until
    /// the replacement is installed.
    held: HeldResources,
    /// The box the content was recorded at.
    size: Cell<kurbo::Size>,
}

/// The retained node behind a `SceneView`.
struct SceneNode {
    content: Box<dyn SceneContent>,
    /// The box the recording was made at — one per node, wired into every
    /// `record_into` call so `layout_size`-bound operands track it.
    layout_size: LayoutSize,
    installed: Option<Installed>,
    /// Recording storage handed back by the last `retire`.
    spare: ContentSpare,
    /// Set by the content's invalidator, and by content that asked for
    /// another frame; cleared by the re-record it triggers.
    invalidated: Rc<Cell<bool>>,
    /// The owner the recording notifies.
    owner: Rc<SceneOwner>,
    /// The sampling flag `attach` hands to the recording's `LiveState`:
    /// it sets it when an animated operand queues. The node never reads it
    /// — `needs_sample`/`sample` carry the same signal — but `attach`
    /// requires a live instance, so the node keeps it.
    flag: SampleFlag,
    /// The intrinsic size `patch` last validated — the node's only
    /// measurement input besides the proposal.
    measured_intrinsic: Cell<Option<Size>>,
    /// Bumped on every `Replace`/`Update` the installed content takes — the
    /// change marker emitted `Scene` commands compare by. Shared with the
    /// owner so a hidden-scene apply bumps it too.
    generation: Rc<Cell<u64>>,
    /// Whether the installed generation's list isolates — recomputed by the
    /// node wherever it already holds the mutable borrow, and by the
    /// owner's hidden apply, which can write a `BeginGroup`'s blend too.
    isolated: Rc<Cell<bool>>,
    signals: FrameSignals,
    accessibility_id: NodeId,
}

impl SceneNode {
    /// The name panics about unrecordable content cite: the accessibility
    /// label when the content has one, else the scene's node id. This is
    /// user code — callers evaluate it only on the panic path, never while
    /// the recording is borrowed.
    fn label(&self) -> String {
        self.content
            .accessibility_label()
            .unwrap_or_else(|| format!("scene node {:?}", self.accessibility_id))
    }

    /// Samples running animations into pending operand updates, drains the
    /// pending updates into the recording's own list — in place, so the
    /// `Scene` commands sharing it repaint without a rebuild — and asks for
    /// the next frame while anything still animates. A bound signal or a
    /// sampled animation repaints the scene's box without re-running
    /// `build_scene`.
    ///
    /// # Panics
    ///
    /// Panics naming the scene when an update writes an operand dew cannot
    /// draw — a live paint taking a mesh value, say — here where the change
    /// lands rather than in whichever band first replays it.
    fn drain_live(&self) {
        let Some(installed) = self.installed.as_ref() else {
            return;
        };
        let (finding, blends) = {
            let mut slot = installed.content.borrow_mut();
            let Some(content) = slot.as_mut() else {
                return;
            };
            // The board's frame clock, not wall time, dates the sample so
            // animation cadence is the pump's cadence.
            let animating = if content.needs_sample() {
                content.sample(self.signals.frame_clock())
            } else {
                Animating::IDLE
            };
            if animating.is_animating() {
                self.signals.request_refresh();
            }
            match content.take_change() {
                Some(ContentChange::Update(updates)) => {
                    // `take_change` applied the operands to the list — only
                    // the generation bumps, and only the written operands
                    // re-validate, each against its post-apply command.
                    self.generation.set(self.generation.get() + 1);
                    let list = content.snapshot();
                    (
                        updates
                            .iter()
                            .find_map(|update| invalid_update(update, list)),
                        Some(draw::blends_within(list, 0..list.len())),
                    )
                }
                Some(ContentChange::Replace(_)) => {
                    unreachable!("the install consumed the first Replace")
                }
                None => (None, None),
            }
        };
        if let Some(finding) = finding {
            // `label` is content code — it runs only on the panic path and
            // never while the recording is borrowed.
            panic!("{} recorded {finding}", self.label());
        }
        if let Some(blends) = blends {
            self.isolated.set(blends);
        }
    }

    /// Re-records the content at `width` × `height` and installs the result
    /// live into the node's one `Rc`. The spare carries the previous
    /// recording's live state into the new one; the display-list buffers
    /// themselves are not recycled — `Content::retire` drops the `Picture`,
    /// and reclaiming its storage needs a cherenkov-record API that does
    /// not exist yet.
    ///
    /// # Panics
    ///
    /// Panics on a recording dew cannot draw — an unsupported paint, filter
    /// or spread — naming the scene, here at install where the defect
    /// originates rather than in whichever band first replays it.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "logical-pixel geometry is far below f32 precision limits"
    )]
    fn record(&mut self, renderer: &mut DewRenderer, width: f64, height: f64) {
        // The old registrations must outlive the new recording's assembly —
        // the `SceneResources` table the new `record_into` registers against
        // may still resolve through them — so hold them past the install.
        let old = self.installed.take();
        let old_held = old.as_ref().map(|old| old.held.clone());
        let old_slot = old.map(|old| old.content);
        if let Some(content_rc) = &old_slot {
            // Retire before `layout_size.set`: the resize must not mark
            // operands on a dead recording and wake a spurious frame. The
            // slot is `Option` — the old content leaves under the one short
            // borrow and no placeholder replaces it; a hidden `changed`
            // arriving mid-record finds `None` and does nothing, which is
            // correct: this record supersedes the change.
            if let Some(retiring) = content_rc.borrow_mut().take() {
                self.spare.merge(retiring.retire());
            }
        }
        self.layout_size
            .set(&LayoutSize::change(kurbo::Size::new(width, height), None));
        let mut wants_another_frame = false;
        let mut resources = renderer.scene_pipeline().recording();
        let mut content = Content::record_into(
            core::mem::take(&mut self.spare),
            &self.layout_size,
            |recorder| {
                wants_another_frame =
                    self.content
                        .build_scene(recorder, &mut resources, width as f32, height as f32);
            },
        );
        let held = resources.finish();
        let owner: Rc<dyn LiveOwner> = self.owner.clone();
        content.attach(Rc::downgrade(&owner), &self.flag);
        let Some(ContentChange::Replace(_)) = content.take_change() else {
            panic!("a newly installed content's first change is always its whole picture")
        };
        validate(content.view().commands(), || self.label());
        let list = content.snapshot();
        self.isolated.set(draw::blends_within(list, 0..list.len()));
        let content_rc = if let Some(content_rc) = old_slot {
            *content_rc.borrow_mut() = Some(content);
            content_rc
        } else {
            let content_rc = Rc::new(RefCell::new(Some(content)));
            *self.owner.content.borrow_mut() = Rc::downgrade(&content_rc);
            content_rc
        };
        self.generation.set(self.generation.get() + 1);
        if wants_another_frame {
            // Animated content asked for another frame: the refresh request
            // schedules it, and the invalidation is what makes that frame
            // re-record rather than re-emit this one.
            self.invalidated.set(true);
            self.signals.request_refresh();
        }
        self.installed = Some(Installed {
            content: content_rc,
            held,
            size: Cell::new(kurbo::Size::new(width, height)),
        });
        drop(old_held);
    }
}

impl DewNode for SceneNode {
    fn measure(&self, _state: &RefCell<DewState>, proposal: ProposalSize) -> ViewDimensions {
        // Content that is naturally a size (an SVG's viewBox, a formula's
        // typeset box) answers with it on whichever axis the container left
        // open, and keeps its aspect ratio when only one axis was named.
        // Content that has no size of its own fills whatever it is proposed,
        // like a colour or a shape, and is sized by `.frame()` or its container.
        let proposal = resolve_scene_proposal(self.content.intrinsic_size(), proposal);
        ViewDimensions::new(Size::new(
            proposal
                .width
                .filter(|width| width.is_finite())
                .unwrap_or(0.0)
                .max(0.0),
            proposal
                .height
                .filter(|height| height.is_finite())
                .unwrap_or(0.0)
                .max(0.0),
        ))
    }

    fn render(&mut self, renderer: &mut DewRenderer, ctx: RenderContext) {
        let width = ctx.bounds.width().max(0.0);
        let height = ctx.bounds.height().max(0.0);
        if width <= 0.0 || height <= 0.0 {
            // An empty box draws nothing: it must not consume an
            // invalidation (the flag stays for the first real box to
            // record against), and the installed slot goes `None` — no
            // placeholder recording is built — so nothing is emitted under
            // a zero-area clip while the slot stays alive.
            if let Some(installed) = self.installed.as_ref()
                && installed.size.get() != kurbo::Size::ZERO
            {
                // Not already empty: retire the installed recording once —
                // a scene kept at zero size does not allocate per frame.
                if let Some(retiring) = installed.content.borrow_mut().take() {
                    self.spare.merge(retiring.retire());
                }
                self.generation.set(self.generation.get() + 1);
                installed.size.set(kurbo::Size::ZERO);
                self.isolated.set(false);
            }
            return;
        }
        self.drain_live();
        let stale = self
            .installed
            .as_ref()
            .is_none_or(|installed| installed.size.get() != kurbo::Size::new(width, height));
        if self.invalidated.replace(false) || stale {
            self.record(renderer, width, height);
        }
        if let Some(installed) = self.installed.as_ref()
            && !installed
                .content
                .borrow()
                .as_ref()
                .is_none_or(Content::is_empty)
        {
            let transform = ctx.transform * Affine::translate((ctx.bounds.x0, ctx.bounds.y0));
            let bounds = Rect::new(0.0, 0.0, width, height);
            let window_bounds = transform.transform_rect_bbox(bounds);
            let scene = Scene::new(
                Rc::clone(&installed.content),
                self.generation.get(),
                self.isolated.get(),
                Rc::clone(&renderer.scene_pipeline().target),
            );
            let list = renderer.list_mut();
            list.push_clip(window_bounds);
            list.push_placed(
                DrawCommand::Scene {
                    scene,
                    transform,
                    bounds,
                    clip: None,
                },
                window_bounds,
            );
            list.pop_clip();
            // A `LiveOwner` change arriving while this command is what the
            // frame is drawing requests the next frame; the stamp is what
            // tells that change from one for a scene this frame skipped.
            self.owner.emitted.set(self.owner.frame.get());
        }
        if renderer.accessibility_enabled() {
            // What the drawing says about itself — a formula's MathML, say.
            // The recorded list is opaque to accessibility, so this node is
            // the only place its content can be announced, and the content is
            // the only thing that knows what it drew. Read every frame, so
            // content that follows a signal republishes what it currently
            // draws.
            let label = self.content.accessibility_label();
            self.owner.name.borrow_mut().clone_from(&label);
            let value = self.content.accessibility_value();
            renderer.register_built_accessibility_node(
                self.accessibility_id,
                ctx.window_bounds(),
                move || {
                    let mut node = AccessibilityNode::new(Role::Image);
                    if let Some(label) = label {
                        node.set_label(label);
                    }
                    if let Some(value) = value {
                        node.set_value(value);
                    }
                    (node, None)
                },
            );
        }
    }

    fn stretch_axis(&self) -> StretchAxis {
        scene_stretch_axis(self.content.intrinsic_size())
    }

    fn patch(&mut self, _renderer: &mut DewRenderer) -> bool {
        // Content may resize itself — a swapped SVG document, a data-driven
        // canvas — through the same invalidator it uses for repaints, or by
        // simply answering differently here, so the size input is re-read
        // rather than trusted to be constant. A repaint that changed nothing
        // about the intrinsic size invalidates no measurement.
        let intrinsic = self.content.intrinsic_size();
        self.measured_intrinsic.replace(intrinsic) != intrinsic
    }
}

impl Drop for SceneNode {
    fn drop(&mut self) {
        // The content's invalidator must outlive no node: an unmounted
        // scene waking a refresh would schedule frames for a view that no
        // longer draws.
        self.content.set_invalidator(None);
    }
}

/// Whether `command` is one dew cannot draw — the finding the install-time
/// scan and the per-update scan both report.
fn command_finding(command: &draw::Command) -> Option<String> {
    match command {
        draw::Command::Fill { paint, .. }
        | draw::Command::Stroke { paint, .. }
        | draw::Command::Glyphs { paint, .. } => paint_finding(paint),
        draw::Command::Shadow { shape, .. } => {
            matches!(shape, draw::ShapeData::Line(_)).then(|| {
                "a shadow on a Line, which dew cannot draw — a zero-spread one \
                 draws nothing and cherenkov-cpu refuses them too"
                    .to_string()
            })
        }
        draw::Command::BeginGroup { group, .. } => group.filter.map(|filter| {
            format!(
                "a group naming filter {}, but dew's scene target registers no \
                 filter chains: `SceneBackend` exposes none to name",
                filter.raw()
            )
        }),
        draw::Command::Picture { picture, .. } => picture
            .display_list()
            .commands()
            .iter()
            .find_map(command_finding),
        draw::Command::Image { .. }
        | draw::Command::BeginClip { .. }
        | draw::Command::BeginTransform { .. }
        | draw::Command::End => None,
    }
}

/// Whether one post-apply command — the one `update` wrote an operand of —
/// is undrawable, per [`command_finding`].
fn invalid_update(update: &draw::SlotUpdate, list: &draw::DisplayList) -> Option<String> {
    list.commands()
        .get(update.command as usize)
        .and_then(command_finding)
}

/// Whether `paint` is one dew cannot draw, recursing through a
/// `Transformed` chain to the paint it wraps.
fn paint_finding(paint: &draw::Paint) -> Option<String> {
    match paint {
        draw::Paint::Mesh(_) => Some("a mesh gradient paint".to_string()),
        draw::Paint::Shader(_) => Some("a shader paint".to_string()),
        draw::Paint::Linear(gradient) if gradient.extend == draw::Extend::None => {
            Some("a gradient with `Extend::None`".to_string())
        }
        draw::Paint::Radial(gradient) if gradient.extend == draw::Extend::None => {
            Some("a gradient with `Extend::None`".to_string())
        }
        draw::Paint::Sweep(gradient) if gradient.extend == draw::Extend::None => {
            Some("a gradient with `Extend::None`".to_string())
        }
        draw::Paint::Image(pattern)
            if pattern.extend_x == draw::Extend::None || pattern.extend_y == draw::Extend::None =>
        {
            Some("an image pattern with `Extend::None`".to_string())
        }
        draw::Paint::Transformed(transformed) => paint_finding(&transformed.paint),
        draw::Paint::Solid(_)
        | draw::Paint::Linear(_)
        | draw::Paint::Radial(_)
        | draw::Paint::Sweep(_)
        | draw::Paint::Image(_) => None,
    }
}

/// Validates that every recorded command is one dew can draw — unsupported
/// paints, filter chains and line shadows fail here, at install, rather
/// than in whichever band first replays the scene.
///
/// # Panics
///
/// Panics naming the scene on the first unsupported command. `name` runs
/// only then — it may be user code, and it must never run while the
/// recording is borrowed.
fn validate(commands: &[draw::Command], name: impl Fn() -> String) {
    if let Some(finding) = commands.iter().find_map(command_finding) {
        panic!("{} recorded {finding}", name());
    }
}

/// Builds the retained node for a scene view, wiring the content's
/// invalidation to dew's frame pump.
pub fn build(renderer: &mut DewRenderer, scene: SceneView) -> Box<dyn DewNode> {
    let mut content = scene.into_content();
    let invalidated = Rc::new(Cell::new(false));
    let signals = renderer.signals();
    content.set_invalidator(Some(Rc::new({
        let invalidated = Rc::clone(&invalidated);
        let signals = signals.clone();
        move || {
            invalidated.set(true);
            signals.request_refresh();
        }
    })));
    let generation = Rc::new(Cell::new(0));
    let isolated = Rc::new(Cell::new(false));
    let accessibility_id = renderer.allocate_accessibility_id();
    let owner = Rc::new(SceneOwner {
        signals: signals.clone(),
        content: RefCell::new(Weak::new()),
        frame: renderer.frame_number(),
        emitted: Cell::new(0),
        generation: Rc::clone(&generation),
        isolated: Rc::clone(&isolated),
        name: Rc::new(RefCell::new(None)),
        accessibility_id,
    });
    Box::new(SceneNode {
        measured_intrinsic: Cell::new(content.intrinsic_size()),
        content,
        layout_size: LayoutSize::new(),
        installed: None,
        spare: ContentSpare::default(),
        invalidated,
        owner,
        flag: SampleFlag::new(),
        generation,
        isolated,
        signals,
        accessibility_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use nami::binding;
    use waterui_graphics::draw::Draw as _;

    /// Scene content that counts how often it records, for tests that must
    /// see each `build_scene` run.
    struct Tracker {
        builds: Rc<Cell<usize>>,
    }

    impl SceneContent for Tracker {
        fn build_scene(
            &mut self,
            recorder: &mut draw::Recorder,
            _resources: &mut RecordingResources<'_>,
            width: f32,
            height: f32,
        ) -> bool {
            self.builds.set(self.builds.get() + 1);
            recorder.fill(
                Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
                draw::WorkingColor::new([0.2, 0.4, 0.8, 1.0]),
            );
            false
        }

        fn rebuild_for_engine(&mut self) {}
    }

    /// A `SceneNode` built exactly as `build` builds it, but kept concrete so
    /// a test can read the invalidation flag.
    fn tracker_node(renderer: &mut DewRenderer, builds: &Rc<Cell<usize>>) -> SceneNode {
        let invalidated = Rc::new(Cell::new(false));
        let signals = renderer.signals();
        let generation = Rc::new(Cell::new(0));
        let isolated = Rc::new(Cell::new(false));
        let accessibility_id = renderer.allocate_accessibility_id();
        let owner = Rc::new(SceneOwner {
            signals: signals.clone(),
            content: RefCell::new(Weak::new()),
            frame: renderer.frame_number(),
            emitted: Cell::new(0),
            generation: Rc::clone(&generation),
            isolated: Rc::clone(&isolated),
            name: Rc::new(RefCell::new(None)),
            accessibility_id,
        });
        SceneNode {
            measured_intrinsic: Cell::new(None),
            content: Box::new(Tracker {
                builds: Rc::clone(builds),
            }),
            layout_size: LayoutSize::new(),
            installed: None,
            spare: ContentSpare::default(),
            invalidated,
            owner,
            flag: SampleFlag::new(),
            generation,
            isolated,
            signals,
            accessibility_id,
        }
    }

    fn renderer() -> DewRenderer {
        DewRenderer::new(FrameSignals::new(Instant::now()), crate::test_fonts())
    }

    /// A zero-area box: the scene draws nothing and, per the render path's
    /// early return, does not consume its invalidation.
    #[test]
    fn an_empty_box_keeps_its_invalidation() {
        let mut renderer = renderer();
        let builds = Rc::new(Cell::new(0));
        let mut node = tracker_node(&mut renderer, &builds);
        node.render(&mut renderer, RenderContext::root(8.0, 8.0));
        assert_eq!(builds.get(), 1, "the first box records once");

        node.invalidated.set(true);
        node.render(
            &mut renderer,
            RenderContext {
                transform: Affine::IDENTITY,
                bounds: Rect::new(0.0, 0.0, 0.0, 8.0),
                proposal: ProposalSize::new(Some(0.0), Some(8.0)),
            },
        );
        assert!(
            node.invalidated.get(),
            "an empty box must not consume an invalidation"
        );
        assert_eq!(builds.get(), 1, "an empty box records nothing");

        node.render(&mut renderer, RenderContext::root(8.0, 8.0));
        assert_eq!(
            builds.get(),
            2,
            "the kept invalidation records on the next real box"
        );
    }

    /// A change on a scene the current frame never drew applies into the
    /// recording at once — `take_change` plus the shared generation — and
    /// wakes nothing. The same change while the scene is showing asks for a
    /// refresh instead.
    #[test]
    fn a_hidden_scene_applies_updates_without_waking_a_frame() {
        let signals = FrameSignals::new(Instant::now());
        let frame = Rc::new(Cell::new(1_u64));
        let generation = Rc::new(Cell::new(0_u64));
        let mut renderer = renderer();
        let owner = Rc::new(SceneOwner {
            signals: signals.clone(),
            content: RefCell::new(Weak::new()),
            frame: Rc::clone(&frame),
            emitted: Cell::new(0), // last emitted in frame 0 — hidden in frame 1
            generation: Rc::clone(&generation),
            isolated: Rc::new(Cell::new(false)),
            name: Rc::new(RefCell::new(None)),
            accessibility_id: renderer.allocate_accessibility_id(),
        });
        let fill = binding(draw::WorkingColor::new([1.0, 0.0, 0.0, 1.0]));
        let layout = LayoutSize::new();
        layout.set(&LayoutSize::change(kurbo::Size::new(8.0, 8.0), None));
        let content = Rc::new(RefCell::new(Some(Content::record(&layout, {
            let fill = fill.clone();
            move |recorder| recorder.fill(Rect::new(0.0, 0.0, 8.0, 8.0), fill)
        }))));
        let flag = SampleFlag::new();
        let owner_typed: Rc<dyn LiveOwner> = owner.clone();
        content
            .borrow_mut()
            .as_mut()
            .unwrap()
            .attach(Rc::downgrade(&owner_typed), &flag);
        *owner.content.borrow_mut() = Rc::downgrade(&content);
        // Consume the install's `Replace`; only live updates remain after it.
        assert!(matches!(
            content.borrow_mut().as_mut().unwrap().take_change(),
            Some(ContentChange::Replace(_))
        ));

        fill.set(draw::WorkingColor::new([0.0, 1.0, 0.0, 1.0]));
        assert_eq!(
            generation.get(),
            1,
            "the hidden apply bumps the shared generation"
        );
        assert!(
            !signals.take_patch_request() && !signals.take_redraw_request(),
            "a hidden scene never wakes a refresh"
        );
        let green = content
            .borrow()
            .as_ref()
            .unwrap()
            .view()
            .commands()
            .first()
            .and_then(|command| match command {
                draw::Command::Fill {
                    paint: draw::Paint::Solid(color),
                    ..
                } => Some(color.components[1]),
                _ => None,
            });
        assert!(
            green.is_some_and(|green| green > 0.5),
            "the bound operand reached the list in place"
        );

        // The same change while the scene is showing wakes the pump.
        owner.emitted.set(frame.get());
        fill.set(draw::WorkingColor::new([0.0, 0.0, 1.0, 1.0]));
        assert!(
            signals.take_patch_request(),
            "a change on a visible scene requests a refresh"
        );
    }

    /// `validate` refuses a `Line` shadow — cherenkov-cpu rejects them, and
    /// dew has nothing to draw — naming the scene in the panic.
    #[test]
    #[should_panic(expected = "a shadow on a Line")]
    fn a_line_shadow_is_refused_at_install() {
        let layout = LayoutSize::new();
        let content = Content::record(&layout, |recorder| {
            recorder.shadow(
                kurbo::Line::new((0.0, 0.0), (8.0, 8.0)),
                draw::Shadow::new(1.0, draw::WorkingColor::new([0.0, 0.0, 0.0, 1.0])),
            );
        });
        validate(content.view().commands(), || "the test scene".to_string());
    }

    /// The hidden apply runs `invalid_update` too: a live paint taking an
    /// `Extend::None` gradient while the scene is hidden panics on the
    /// owner's path, naming the scene, rather than reaching the painter's
    /// `unreachable!`.
    #[test]
    #[should_panic(expected = "the hidden test scene recorded a gradient with `Extend::None`")]
    fn a_hidden_scenes_update_is_validated() {
        let signals = FrameSignals::new(Instant::now());
        let frame = Rc::new(Cell::new(1_u64));
        let mut renderer = renderer();
        let owner = Rc::new(SceneOwner {
            signals,
            content: RefCell::new(Weak::new()),
            frame: Rc::clone(&frame),
            emitted: Cell::new(0), // hidden — the change applies at once
            generation: Rc::new(Cell::new(0)),
            isolated: Rc::new(Cell::new(false)),
            name: Rc::new(RefCell::new(Some("the hidden test scene".to_string()))),
            accessibility_id: renderer.allocate_accessibility_id(),
        });
        let paint = binding(draw::Paint::Solid(draw::WorkingColor::new([
            1.0, 0.0, 0.0, 1.0,
        ])));
        let layout = LayoutSize::new();
        layout.set(&LayoutSize::change(kurbo::Size::new(8.0, 8.0), None));
        let content = Rc::new(RefCell::new(Some(Content::record(&layout, {
            let paint = paint.clone();
            move |recorder| recorder.fill(Rect::new(0.0, 0.0, 8.0, 8.0), paint)
        }))));
        let flag = SampleFlag::new();
        let owner_typed: Rc<dyn LiveOwner> = owner.clone();
        content
            .borrow_mut()
            .as_mut()
            .unwrap()
            .attach(Rc::downgrade(&owner_typed), &flag);
        *owner.content.borrow_mut() = Rc::downgrade(&content);
        assert!(matches!(
            content.borrow_mut().as_mut().unwrap().take_change(),
            Some(ContentChange::Replace(_))
        ));
        // `set` notifies the owner synchronously — the hidden apply lands
        // the unrecordable paint and the validation panics inside the call.
        paint.set(draw::Paint::Linear(
            draw::LinearGradient::new((0.0, 0.0), (8.0, 0.0))
                .stop(0.0, draw::WorkingColor::new([1.0, 0.0, 0.0, 1.0]))
                .stop(1.0, draw::WorkingColor::new([0.0, 0.0, 1.0, 1.0]))
                .extend(draw::Extend::None),
        ));
    }
}
