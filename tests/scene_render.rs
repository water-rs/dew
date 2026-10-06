//! Scene content — `Canvas` drawings and SVG documents — rendered end to end
//! on dew.
//!
//! These are the proof that scene content is engine-portable: the same
//! recordings that hydrolysis resolves into its GPU surfaces reach dew's
//! painter as `cherenkov_record` command lists, replayed band by band into
//! `vello_cpu`, with no component code aware of either engine.
//!
//! Two properties matter beyond "it draws". A scene is opaque to the display
//! list, so it must (1) stay one command whose bounds are exactly the box the
//! view was given, and (2) be rebuilt only when its own content invalidates —
//! a canvas that redrew itself every frame would dirty its whole rect every
//! frame, which is precisely what dew's banded, dirty-region engine exists to
//! avoid. A third property lives at the pixel level: a bound signal must
//! repaint the scene through the recording's own update path — never through
//! `build_scene`.
//!
//! Run with `--no-capture` to export the review PNGs under
//! `/tmp/waterui_dew_scene2d/`.

use core::cell::Cell;
use std::rc::Rc;

use accesskit::Role;
use kurbo::{Affine, Rect, Shape as _};
use nami::{Binding, Signal, binding};
use waterui::prelude::*;
use waterui_canvas::Canvas;
use waterui_core::AnyView;
use waterui_core::layout::{Point, Rect as LayoutRect, Size};
use waterui_dew::{ClipRegion, DewRuntime, DisplayList, DrawCommand, HostBoard, render_view_png};
use waterui_graphics::color::Srgb;
use waterui_graphics::draw::{Color as DrawColor, Draw, Recorder};
use waterui_graphics::{
    RecordingResources, SceneContent, SceneInvalidator, SceneView, invalidate_on_change,
};
use waterui_layout::scroll::ScrollView;
use waterui_math::ast::MathStyle;
use waterui_math::view::Math;
use waterui_math::{latex, mathml, speech};
use waterui_svg::Svg;

mod support;

/// A small document exercising fills, strokes and a group opacity — the three
/// things an SVG asks of a scene, and the third of which needs a real
/// compositing layer rather than a per-shape alpha.
const INLINE_SVG: &str = include_str!("assets/scene.svg");

fn render_scene<V: View>(build: impl Fn() -> V + 'static, width: u32, height: u32) -> DisplayList {
    let mut renderer = support::test_renderer();
    renderer.render_tree(
        AnyView::new(build()),
        &support::test_environment(),
        f64::from(width),
        f64::from(height),
    )
}

/// The single scene command a root-level scene view emits, after the
/// background fill.
fn only_scene(list: &DisplayList) -> (&DrawCommand, Rect) {
    assert_eq!(
        list.commands().len(),
        2,
        "a root scene view emits one background fill and one scene command"
    );
    let placed = &list.commands()[1];
    let command = placed.command();
    assert!(
        matches!(command, DrawCommand::Scene { .. }),
        "the content command must be a scene, got {command:?}"
    );
    (command, placed.bounds())
}

fn export(name: &str, png: &[u8]) {
    std::fs::write(support::export_path("scene2d", name), png).expect("write the review PNG");
}

/// A canvas group drawn at 0.5 opacity composites through the recording's
/// layer group — not per-shape alpha — and its strokes anti-alias their edges
/// into the band.
///
/// The group's recorded `blend_space` is `Linear`, and cherenkov honours it:
/// on hydrolysis white at 0.5 over black lands near 188. dew composites in the
/// encoding `vello_cpu` works in — sRGB — and lands near 128. This test pins
/// dew's sRGB compositing, the documented asymmetry, not the framework's
/// linear-light contract.
#[test]
fn a_group_at_half_opacity_composites_in_dews_srgb_and_antialiases() {
    let mut runtime = DewRuntime::new(
        HostBoard::new(120, 120),
        support::test_environment(),
        16,
        || {
            AnyView::new(Canvas::new(|ctx| {
                // A black canvas underneath makes the composited result exact:
                // white at 0.5 over black is 127 or 128, independent of the
                // theme background behind the scene.
                ctx.set_fill_style(Srgb::new_u8(0, 0, 0));
                ctx.fill_rect(box_of(ctx.width, ctx.height));

                ctx.push_alpha_rect(0.5, box_of(ctx.width, ctx.height));
                ctx.set_fill_style(Srgb::new_u8(255, 255, 255));
                ctx.fill_rect(LayoutRect::new(
                    Point::new(20.0, 20.0),
                    Size::new(40.0, 40.0),
                ));
                ctx.fill_circle(Point::new(90.0, 60.0), 20.0);
                ctx.pop_layer();
            }))
        },
    );
    runtime.pump().expect("the first frame renders");
    let display = runtime.board().framebuffer();

    // White at 0.5 over black composites to ~128 in dew's sRGB space (in a
    // linear space it would be ~188 — the documented asymmetry); allow a
    // unit of rasterization tolerance.
    let inside = display.pixel(30, 30);
    assert!(
        (120..=135).contains(&inside[0])
            && inside[0] == inside[1]
            && inside[1] == inside[2]
            && inside[3] == 255,
        "a 0.5 group over black must composite mid-grey, got {inside:?}"
    );
    let center = display.pixel(90, 60);
    assert_eq!(
        center, inside,
        "the circle's interior carries the same group-alpha grey"
    );
    // The circle's edge anti-aliases: tracing a column through its top edge
    // must find a coverage-mixed pixel strictly between black and the grey.
    let edge: Vec<u8> = (35..50).map(|y| display.pixel(90, y)[0]).collect();
    assert!(
        edge.iter().any(|p| *p > 10 && *p < inside[0] - 4),
        "the circle's edge must show a partially covered pixel, got {edge:?}"
    );
    // Outside the circle the canvas stays black.
    assert_eq!(display.pixel(60, 100), [0, 0, 0, 255]);
}

/// A canvas fill bound to a signal repaints through the recording's live
/// operand update — no `build_scene` pass, only the scene's own box dirty.
///
/// The canvas sits in a padded frame over a sibling fill, so the dirty rect
/// can only be the scene's box: a full repaint or a rebuild would dirty the
/// whole surface and would re-run the closure.
#[test]
fn a_bound_canvas_fill_repaints_without_rebuilding() {
    let color = binding(Color::srgb(255, 0, 0));
    let builds = Rc::new(Cell::new(0usize));
    let mut runtime = DewRuntime::new(HostBoard::new(80, 60), support::test_environment(), 16, {
        let color = color.clone();
        let builds = Rc::clone(&builds);
        move || {
            AnyView::new(
                Color::srgb(0, 32, 0).overlay(
                    Canvas::new({
                        let color = color.clone();
                        let builds = Rc::clone(&builds);
                        move |ctx| {
                            builds.set(builds.get() + 1);
                            ctx.set_fill_style(waterui::color::signal_color(color.clone()));
                            ctx.fill_rect(box_of(ctx.width, ctx.height));
                        }
                    })
                    .padding_with(10.0f32),
                ),
            )
        }
    });
    runtime.pump().expect("the first frame renders");
    assert_eq!(builds.get(), 1, "the scene builds its drawing once");
    assert_eq!(
        runtime.board().framebuffer().pixel(40, 30),
        [255, 0, 0, 255],
        "the bound color paints on the first frame"
    );

    color.set(Color::srgb(0, 0, 255));
    let frame = runtime
        .pump()
        .expect("setting a bound color must schedule a repaint");
    assert_eq!(
        builds.get(),
        1,
        "a live operand update must not re-run the content's build_scene"
    );
    assert_eq!(
        frame.dirty,
        vec![Rect::new(10.0, 10.0, 70.0, 50.0)],
        "the repaint dirties exactly the scene's padded box, not the surface"
    );
    assert_eq!(
        runtime.board().framebuffer().pixel(40, 30),
        [0, 0, 255, 255],
        "the repaint shows the signal's new value"
    );
    let margin = runtime.board().framebuffer().pixel(4, 4);
    assert!(
        margin[1] > margin[0] + 10 && margin[1] > margin[2] + 10 && margin[3] == 255,
        "the padding margin keeps the sibling's green, got {margin:?}"
    );
}

/// A box covering the canvas' own coordinate space.
const fn box_of(width: f32, height: f32) -> LayoutRect {
    LayoutRect::new(Point::new(0.0, 0.0), Size::new(width, height))
}

fn swatch() -> Canvas {
    Canvas::new(move |ctx| {
        ctx.set_fill_style(Srgb::new_u8(32, 44, 68));
        let box_of_canvas = box_of(ctx.width, ctx.height);
        ctx.fill_rect(box_of_canvas);
    })
}

/// A canvas drawing is one scene command, placed at the view's box.
#[test]
fn a_canvas_is_one_scene_command_bounded_by_its_view() {
    let list = render_scene(swatch, 200, 120);
    let (command, bounds) = only_scene(&list);
    let DrawCommand::Scene {
        transform,
        bounds: local,
        clip,
        ..
    } = command
    else {
        unreachable!("only_scene asserted the variant")
    };
    assert_eq!(*transform, Affine::IDENTITY);
    assert_eq!(*local, Rect::new(0.0, 0.0, 200.0, 120.0));
    assert_eq!(bounds, Rect::new(0.0, 0.0, 200.0, 120.0));
    // The scene is clipped to the box it was built for, exactly as the
    // `GpuSurface` realization clips it by rendering into a texture of that
    // size — which is also what makes the command's dirty bounds exact.
    let clip = clip.as_ref().expect("a scene is clipped to its own box");
    let [ClipRegion::Rect(rect)] = clip.regions() else {
        panic!("a scene's clip is one rectangle")
    };
    assert_eq!(*rect, Rect::new(0.0, 0.0, 200.0, 120.0));
}

/// A scene view publishes an accessibility node, so scene content is not a
/// hole in the tree.
#[test]
fn a_scene_publishes_an_accessibility_node() {
    let mut runtime = DewRuntime::new(
        HostBoard::new(160, 160),
        support::test_environment(),
        16,
        || AnyView::new(swatch()),
    );
    runtime.pump().expect("the first frame renders");
    let update = runtime
        .board()
        .accessibility_tree()
        .expect("dew publishes an accessibility tree");
    assert!(
        update
            .nodes
            .iter()
            .any(|(_, node)| node.role() == Role::Image),
        "a scene view is published as an image node"
    );
}

/// Scene content that knows what it drew names its own node.
///
/// A formula reaches the panel as anonymous filled paths, so this node is the
/// only place its content can be announced at all. Dew published it unnamed:
/// the tree said "image" and nothing else. The content's own
/// `SceneContent::accessibility_label` — the formula spoken as a sentence — is
/// what names it, the same answer hydrolysis offers for the same drawing, so a
/// formula is readable on both self-drawn backends rather than on whichever one
/// the test happened to run.
#[test]
fn scene_content_names_its_own_accessibility_node() {
    const FORMULA: &str = r"\frac{a}{b}";

    let mut runtime = DewRuntime::new(
        HostBoard::new(160, 160),
        support::test_environment(),
        16,
        || AnyView::new(Math::new(FORMULA)),
    );
    runtime.pump().expect("the first frame renders");

    // The expectation comes from the same public converter and speech engine the
    // view publishes through, so it tracks them instead of rotting into a stale
    // literal.
    let markup = mathml::to_mathml(
        &latex::parse(FORMULA).expect("the fixture formula parses"),
        MathStyle::Text,
    );
    let expected = speech::speak(&markup).expect("the fixture formula speaks");

    let update = runtime
        .board()
        .accessibility_tree()
        .expect("dew publishes an accessibility tree");
    let labels: Vec<Option<&str>> = update
        .nodes
        .iter()
        .filter(|(_, node)| node.role() == Role::Image)
        .map(|(_, node)| node.label())
        .collect();
    assert!(
        labels.contains(&Some(expected.as_str())),
        "the formula's image node must carry its speech, got {labels:?}"
    );
}

/// Scene content that draws through the contract renders on dew's CPU
/// rasterizer: filled and stroked shapes plus a gradient-brushed path.
#[test]
fn export_canvas_for_visual_review() {
    let png = render_view_png(
        || {
            Canvas::new(|ctx| {
                ctx.set_fill_style(Srgb::new_u8(16, 22, 38));
                ctx.fill_rect(box_of(ctx.width, ctx.height));

                ctx.set_fill_style(Srgb::new_u8(224, 85, 60));
                ctx.fill_circle(Point::new(70.0, 70.0), 44.0);

                ctx.set_stroke_style(Srgb::new_u8(78, 201, 160));
                ctx.set_line_width(6.0);
                ctx.stroke_rect(LayoutRect::new(
                    Point::new(140.0, 26.0),
                    Size::new(160.0, 88.0),
                ));

                let mut gradient = ctx.create_linear_gradient(20.0, 150.0, 300.0, 150.0);
                gradient.add_color_stop(0.0, Srgb::new_u8(255, 214, 102));
                gradient.add_color_stop(1.0, Srgb::new_u8(120, 88, 232));
                ctx.set_fill_style(gradient);
                ctx.fill_rect(LayoutRect::new(
                    Point::new(20.0, 140.0),
                    Size::new(280.0, 40.0),
                ));
            })
        },
        support::test_environment(),
        320,
        200,
    );
    export("canvas", &png);
}

/// An inline SVG document — fills, a stroked path and a group opacity —
/// renders through the same command.
#[test]
fn an_svg_document_renders_as_a_scene() {
    let list = render_scene(|| Svg::new(INLINE_SVG), 160, 160);
    let (_, bounds) = only_scene(&list);
    assert_eq!(bounds, Rect::new(0.0, 0.0, 160.0, 160.0));

    let png = render_view_png(
        || Svg::new(INLINE_SVG),
        support::test_environment(),
        160,
        160,
    );
    export("svg", &png);
}

/// Scene content that counts how often it is asked to draw.
///
/// It tracks its own input exactly as `Canvas` does: the signal it reads is
/// watched, and the watcher calls the backend's invalidator. That is the whole
/// contract by which content-driven redraws reach dew's dirty regions.
struct CountingContent {
    builds: Rc<Cell<usize>>,
    fill: Binding<u8>,
    animated: bool,
    guard: Option<<Binding<u8> as Signal>::Guard>,
}

impl SceneContent for CountingContent {
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        _resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        self.builds.set(self.builds.get() + 1);
        // Animating content varies its fill with the build count, so every
        // re-record produces a genuinely different picture — every re-record
        // also bumps the `Scene` generation, so even identical pixels would
        // repaint; varying the fill is what keeps the test honest about
        // content rather than bookkeeping.
        let level = if self.animated {
            f32::from(u8::try_from(self.builds.get() % 4).expect("a mod-4 index fits")) / 3.0
        } else {
            f32::from(self.fill.snapshot()) / 255.0
        };
        recorder.fill(
            Rect::new(0.0, 0.0, f64::from(width), f64::from(height)).to_path(0.1),
            DrawColor::<waterui_graphics::draw::Srgb>::new([level, level, level, 1.0]),
        );
        self.animated
    }

    fn rebuild_for_engine(&mut self) {}

    fn set_invalidator(&mut self, invalidator: Option<SceneInvalidator>) {
        self.guard = invalidator.map(|invalidator| invalidate_on_change(&invalidator, &self.fill));
    }
}

fn counting_scene(builds: &Rc<Cell<usize>>, fill: &Binding<u8>, animated: bool) -> SceneView {
    SceneView::new(CountingContent {
        builds: Rc::clone(builds),
        fill: fill.clone(),
        animated,
        guard: None,
    })
}

/// A static scene draws once and never again: neither an idle frame nor a
/// neighbouring widget's change rebuilds or repaints it.
#[test]
fn a_static_scene_is_drawn_once_and_dirties_nothing() {
    let builds = Rc::new(Cell::new(0));
    let fill = binding(80u8);
    let label = binding(Str::from("one"));
    let mut runtime = DewRuntime::new(HostBoard::new(200, 200), support::test_environment(), 16, {
        let builds = Rc::clone(&builds);
        let label = label.clone();
        move || {
            AnyView::new(
                vstack((text(label.clone()), counting_scene(&builds, &fill, false))).spacing(0.0),
            )
        }
    });

    runtime.pump().expect("the first frame renders");
    assert_eq!(builds.get(), 1, "the scene draws itself once");
    assert!(
        runtime.pump().is_none(),
        "a static scene must not keep the frame pump awake"
    );

    label.set(Str::from("two"));
    let frame = runtime
        .pump()
        .expect("a neighbouring text change refreshes the tree");
    assert_eq!(
        builds.get(),
        1,
        "an unrelated change must not redraw the scene"
    );
    let scene_rect = Rect::new(0.0, 20.0, 200.0, 200.0);
    assert!(
        frame
            .dirty
            .iter()
            .all(|rect| rect.intersect(scene_rect).area() <= 0.0),
        "an unrelated change must not dirty the scene's pixels, got {:?}",
        frame.dirty
    );
}

/// A signal the content reads invalidates exactly that content: the scene
/// redraws, and the dirty region is its own rect and nothing else.
#[test]
fn a_content_signal_dirties_exactly_the_scene() {
    let builds = Rc::new(Cell::new(0));
    let fill = binding(80u8);
    let mut runtime = DewRuntime::new(HostBoard::new(200, 120), support::test_environment(), 16, {
        let builds = Rc::clone(&builds);
        let fill = fill.clone();
        move || AnyView::new(counting_scene(&builds, &fill, false))
    });
    runtime.pump().expect("the first frame renders");
    assert_eq!(builds.get(), 1);

    fill.set(200);
    let frame = runtime
        .pump()
        .expect("a content signal must schedule a frame");
    assert_eq!(builds.get(), 2, "the scene rebuilds its drawing");
    assert_eq!(
        frame.dirty,
        vec![Rect::new(0.0, 0.0, 200.0, 120.0)],
        "only the scene's own box repaints"
    );
}

/// Content that asks for another frame keeps getting one, and keeps repainting
/// only its own region.
#[test]
fn animated_content_keeps_asking_for_frames() {
    let builds = Rc::new(Cell::new(0));
    let fill = binding(80u8);
    let mut runtime = DewRuntime::new(HostBoard::new(100, 100), support::test_environment(), 16, {
        let builds = Rc::clone(&builds);
        move || AnyView::new(counting_scene(&builds, &fill, true))
    });
    runtime.pump().expect("the first frame renders");
    for _ in 0..3 {
        let frame = runtime
            .pump()
            .expect("animated content keeps the pump running");
        assert_eq!(
            frame.dirty,
            vec![Rect::new(0.0, 0.0, 100.0, 100.0)],
            "an animating scene repaints its own region only"
        );
    }
    assert_eq!(builds.get(), 4, "every frame redraws the animated content");
}

/// Scene content that *is* 100 x 200 logical points — an SVG's `viewBox`, an
/// image's pixel size, a formula's typeset box.
struct NaturallySizedContent;

impl SceneContent for NaturallySizedContent {
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        _resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        recorder.fill(
            Rect::new(0.0, 0.0, f64::from(width), f64::from(height)).to_path(0.1),
            DrawColor::<waterui_graphics::draw::Srgb>::new([0.0, 0.4, 1.0, 1.0]),
        );
        false
    }

    fn rebuild_for_engine(&mut self) {}

    fn intrinsic_size(&self) -> Option<Size> {
        Some(Size::new(100.0, 200.0))
    }
}

/// The one scene command in a list that also carries a scroll view's chrome.
fn find_scene(list: &DisplayList) -> (&DrawCommand, Rect) {
    let placed = list
        .commands()
        .iter()
        .find(|placed| matches!(placed.command(), DrawCommand::Scene { .. }))
        .expect("the list must carry exactly one scene command");
    (placed.command(), placed.bounds())
}

/// Dew resolves an unconstrained scroll axis from the content's natural size,
/// exactly as hydrolysis does — the two self-drawn backends must not disagree
/// about how big a drawing is (water-rs/waterui#253).
#[test]
fn an_unconstrained_scroll_axis_resolves_to_the_natural_size() {
    let list = render_scene(
        || ScrollView::vertical(SceneView::new(NaturallySizedContent)),
        100,
        120,
    );
    let (command, _) = find_scene(&list);
    let DrawCommand::Scene { bounds: local, .. } = command else {
        unreachable!("find_scene asserted the variant")
    };
    // The viewport names the width (100, the natural width), and leaves the
    // scroll axis open; the drawing is laid out at the 200 points it is, rather
    // than collapsing to zero and being clamped up to the 120-point viewport.
    assert_eq!(*local, Rect::new(0.0, 0.0, 100.0, 200.0));
}

/// Content with no natural size still fills the viewport on the scroll axis,
/// which is what a background or a shader wants.
#[test]
fn a_sizeless_scene_still_fills_an_unconstrained_scroll_axis() {
    let list = render_scene(|| ScrollView::vertical(swatch()), 100, 120);
    let (command, _) = find_scene(&list);
    let DrawCommand::Scene { bounds: local, .. } = command else {
        unreachable!("find_scene asserted the variant")
    };
    assert_eq!(*local, Rect::new(0.0, 0.0, 100.0, 120.0));
}
