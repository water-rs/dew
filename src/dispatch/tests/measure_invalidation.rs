//! Regression coverage for retained measurement invalidation.
//!
//! The retained tree keeps a [`ProposalSize`]-keyed measurement cache on every
//! node across frames, cleared only when `patch` reports that a sizing input
//! actually moved. These tests pin the two halves of that contract: inputs
//! that can change a measured answer — text, fonts, layout parameters, child
//! membership, the offer itself — re-measure what they must and nothing more,
//! and inputs that only repaint — a value, a colour, a scroll offset — leave
//! every cached answer standing.

use core::cell::{Cell, RefCell};
use std::rc::Rc;

use nami::collection::SignalCollection;
use nami::{SignalExt, binding};
use waterui_backend_core::frame_signals::FrameSignals;
use waterui_backend_core::time::Instant;
use waterui_controls::slider::slider;
use waterui_controls::toggle::Toggle;
use waterui_core::dynamic::Dynamic;
use waterui_core::id::SelfId;
use waterui_core::layout::{ProposalSize, Size};
use waterui_core::plugin::Plugin;
use waterui_core::{AnyView, Environment};
use waterui_graphics::{Scene2D, SceneContent, SceneInvalidator, SceneView};
use waterui_layout::frame::Frame;
use waterui_layout::spacer;
use waterui_layout::stack::{VStack, hstack, vstack};
use waterui_text::font::{FontWeight, ResolvedFont};
use waterui_text::styled::StyledStr;
use waterui_text::{Text, text};

use crate::dispatch::{DewRenderer, build_node};

fn test_renderer() -> DewRenderer {
    DewRenderer::new(FrameSignals::new(Instant::now()), crate::test_fonts())
}

fn init_executor() {
    let _ = executor_core::try_init_global_executor(native_executor::NativeExecutor::new());
    waterui_testing::install_test_executor();
}

/// The intrinsic size the retained root currently measures under an
/// unspecified offer.
fn root_size(renderer: &DewRenderer) -> Size {
    renderer
        .root
        .as_ref()
        .expect("a rendered tree has a retained root")
        .measure(renderer.state_cell(), ProposalSize::UNSPECIFIED)
        .size
}

fn measured(renderer: &DewRenderer) -> (u64, u64) {
    let work = renderer.state_cell().borrow().work;
    (work.measures_computed, work.measures_reused)
}

/// A refresh that moved nothing re-measures nothing: every probe a container
/// makes is answered by the retained caches, which is what makes a retained
/// tree cheaper than a rebuild.
#[test]
fn unchanged_tree_reuses_every_measurement() {
    init_executor();
    let env = Environment::new();
    let mut renderer = test_renderer();
    let first = renderer.render_tree(
        AnyView::new(vstack((
            text("alpha"),
            hstack((text("left"), text("right"))),
        ))),
        &env,
        200.0,
        60.0,
    );
    assert!(first.work().measures_computed > 0);

    let second = renderer.refresh_tree(200.0, 60.0);
    assert_eq!(
        second.work().measures_computed,
        0,
        "a frame that changed no sizing input must measure nothing"
    );
    assert!(
        second.work().measures_reused > 0,
        "the retained caches must answer the frame's probes"
    );
}

/// Changing a text's content signal re-measures the chain that sized for it —
/// the text and the ancestors that measured it — while an unchanged sibling
/// keeps serving its cache. This is the stale-cache case the per-frame clear
/// used to hide: the old code re-measured everything, so nothing could go
/// stale.
#[test]
fn changed_text_remeasures_its_chain_and_reuses_siblings() {
    init_executor();
    let count = binding(0_i32);
    let env = Environment::new();
    let mut renderer = test_renderer();
    let first = renderer.render_tree(
        AnyView::new(vstack((
            Text::computed(count.map(|value: i32| StyledStr::plain(value.to_string()))),
            text("x"),
        ))),
        &env,
        200.0,
        60.0,
    );
    let baseline = first.work().measures_computed;
    let before = root_size(&renderer);

    count.set(100_000);
    let frame = renderer.refresh_tree(200.0, 60.0);
    let work = frame.work();
    assert!(
        work.measures_computed > 0,
        "a text change must invalidate the measurement that sized for it"
    );
    assert!(
        work.measures_computed < baseline,
        "a one-leaf change must not re-measure the whole tree"
    );
    assert!(
        work.measures_reused > 0,
        "the unchanged sibling must keep answering from its cache"
    );
    assert!(
        work.text_layouts_shaped > 0,
        "the changed text must be re-shaped"
    );
    let after = root_size(&renderer);
    assert!(
        after.width > before.width,
        "\"100000\" is wider than \"0\", and the measure must see it"
    );

    // And the next frame is clean again — the invalidation was consumed.
    let clean = renderer.refresh_tree(200.0, 60.0);
    assert_eq!(clean.work().measures_computed, 0);
}

/// A reactive layout parameter — here the stack's spacing — invalidates the
/// container that reads it through the layout's own `watch_invalidation`,
/// and asks for the frame the change otherwise would never produce.
#[test]
fn reactive_spacing_invalidates_its_container() {
    init_executor();
    let spacing = binding(0.0_f32);
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((text("a"), text("b"))).spacing(spacing.clone())),
        &env,
        200.0,
        80.0,
    );
    let before = root_size(&renderer);

    let _ = renderer.signals().take_patch_request();
    spacing.set(20.0);
    assert!(
        renderer.signals().take_patch_request(),
        "a layout parameter change must request a frame"
    );
    renderer.refresh_tree(200.0, 80.0);
    let after = root_size(&renderer);
    assert!(
        measured(&renderer).0 > 0,
        "a moved spacing must invalidate the container that reads it"
    );
    assert!(
        after.height > before.height,
        "twenty points of spacing must grow the stack's measured height"
    );
}

/// A `.frame` bound is a layout parameter too: the modifier's layout object
/// watches its own signals, and a moved bound re-measures the frame's box.
#[test]
fn reactive_frame_width_remeasures() {
    init_executor();
    let width = binding(40.0_f32);
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(Frame::new(text("framed")).width(width.clone())),
        &env,
        200.0,
        60.0,
    );

    width.set(120.0);
    let frame = renderer.refresh_tree(200.0, 60.0);
    assert!(
        frame.work().measures_computed > 0,
        "a moved frame bound must invalidate the node that sized for it"
    );
    let measured_width = root_size(&renderer).width;
    assert!(
        (f64::from(measured_width) - 120.0).abs() < 0.5,
        "the frame must now measure its new bound, got {measured_width}"
    );
}

/// `Dynamic` is the structural seam: swapping its child rebuilds the subtree
/// and invalidates every ancestor that measured it, while siblings keep their
/// caches.
#[test]
fn replaced_child_invalidates_its_ancestors() {
    init_executor();
    let (handler, dynamic) = Dynamic::new();
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((dynamic, text("x")))),
        &env,
        200.0,
        80.0,
    );
    let before = root_size(&renderer);

    handler.set(text("a much wider replacement line"));
    let frame = renderer.refresh_tree(200.0, 80.0);
    let work = frame.work();
    assert!(
        work.measures_computed > 0,
        "a replaced child must invalidate the ancestors that measured it"
    );
    assert!(
        work.measures_reused > 0,
        "the unchanged sibling must keep answering from its cache"
    );
    let after = root_size(&renderer);
    assert!(
        after.width > before.width,
        "the new child is wider, and the measure must see it"
    );
}

/// A lazy container's membership is a sizing input: inserting a row
/// invalidates the ancestors that sized for the old set, while the items
/// whose identity survived keep their retained subtrees — caches and all.
#[test]
fn lazy_membership_rebuilds_only_moved_items() {
    init_executor();
    let items = binding(vec![SelfId::new(1_i32), SelfId::new(2)]);
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(VStack::for_each(
            SignalCollection::new(items.clone()),
            |item| text(format!("row {}", *item)),
        )),
        &env,
        200.0,
        80.0,
    );

    // Reorder the survivors and insert one new id.
    let _ = renderer.signals().take_patch_request();
    items.set(vec![SelfId::new(3), SelfId::new(1), SelfId::new(2)]);
    assert!(
        renderer.signals().take_patch_request(),
        "a membership change must request a frame"
    );
    let frame = renderer.refresh_tree(200.0, 80.0);
    let work = frame.work();
    assert!(
        work.measures_computed > 0,
        "a membership change must invalidate the container that sized the old set"
    );
    assert!(
        work.measures_reused > 0,
        "items whose identity survived must keep answering from their caches"
    );
}

/// The cache is keyed by the raw offer, so a resize that revisits a proposal
/// the node has already answered reuses it — a drag back to a seen size
/// re-measures nothing.
#[test]
fn resizing_back_to_a_seen_offer_reuses_its_answers() {
    init_executor();
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((
            text("a line of text long enough to wrap at sixty points"),
            spacer(),
        ))),
        &env,
        200.0,
        60.0,
    );

    let narrow = renderer.refresh_tree(60.0, 120.0);
    assert!(
        narrow.work().measures_computed > 0,
        "proposals never offered before must compute fresh answers"
    );
    let back = renderer.refresh_tree(200.0, 60.0);
    assert_eq!(
        back.work().measures_computed,
        0,
        "returning to a seen offer must reuse its cached answers"
    );
}

/// Equal bounds under different offers are different questions: a text wraps
/// to a bounded width and runs free without one, so the cache keeps the two
/// answers keyed by the proposal that produced them.
#[test]
fn equal_bounds_keep_distinct_offers_in_the_cache() {
    init_executor();
    let mut env = Environment::new();
    crate::theme::install_default_fonts(&mut env);
    let mut renderer = test_renderer();
    let node = build_node(
        &mut renderer,
        AnyView::new(text("an offer width probe")),
        &env,
        0,
    );
    let bounded = node
        .measure(renderer.state_cell(), ProposalSize::new(Some(40.0), None))
        .size;
    let open = node
        .measure(renderer.state_cell(), ProposalSize::UNSPECIFIED)
        .size;
    assert!(
        open.width > bounded.width,
        "the two offers must answer differently for this test to mean anything"
    );

    let (computed_before, reused_before) = measured(&renderer);
    let rebounded = node
        .measure(renderer.state_cell(), ProposalSize::new(Some(40.0), None))
        .size;
    let reopened = node
        .measure(renderer.state_cell(), ProposalSize::UNSPECIFIED)
        .size;
    let (computed_after, reused_after) = measured(&renderer);
    assert_eq!(rebounded, bounded);
    assert_eq!(reopened, open);
    assert_eq!(
        computed_after - computed_before,
        0,
        "re-probing seen offers must not recompute"
    );
    assert_eq!(reused_after - reused_before, 2);
}

/// Probes arrive in whatever order the layout negotiation picks; the keyed
/// cache answers each the same regardless of the order it was first asked in.
#[test]
fn probes_in_any_order_reuse_their_keyed_entries() {
    init_executor();
    let mut env = Environment::new();
    crate::theme::install_default_fonts(&mut env);
    let mut renderer = test_renderer();
    let node = build_node(&mut renderer, AnyView::new(text("probe order")), &env, 0);
    let proposals = [
        ProposalSize::UNSPECIFIED,
        ProposalSize::new(Some(80.0), Some(20.0)),
        ProposalSize::new(Some(40.0), Some(20.0)),
        ProposalSize::new(Some(0.0), None),
        ProposalSize::new(Some(-0.0), None),
    ];
    for proposal in proposals {
        node.measure(renderer.state_cell(), proposal);
    }
    for proposal in proposals.iter().rev() {
        node.measure(renderer.state_cell(), *proposal);
    }
    let (computed, reused) = measured(&renderer);
    assert_eq!(computed, 5);
    assert_eq!(reused, 5);
}

/// A theme font is a sizing input every text reads: moving the body slot
/// re-measures the subscribers while nodes that read no font keep their
/// caches.
#[test]
fn body_font_change_remeasures_the_subscribers() {
    init_executor();
    let body = binding(ResolvedFont::new(16.0, FontWeight::Normal));
    let mut env = Environment::new();
    waterui::theme::Theme::new()
        .fonts(waterui::theme::FontSettings::new().body(body.clone()))
        .install(&mut env);

    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((text("body text"), spacer(), spacer()))),
        &env,
        200.0,
        80.0,
    );
    let before = root_size(&renderer);

    body.set(ResolvedFont::new(32.0, FontWeight::Normal));
    let frame = renderer.refresh_tree(200.0, 80.0);
    let work = frame.work();
    assert!(
        work.measures_computed > 0,
        "a font change must re-measure the texts that read it"
    );
    assert!(
        work.measures_reused > 0,
        "nodes that read no font must keep their caches"
    );
    let after = root_size(&renderer);
    assert!(
        after.height > before.height,
        "a body font twice the size must measure taller"
    );
}

/// A control's bound value is paint: the fill and the thumb move, the
/// measured box does not. Per-frame value churn — a vending simulation
/// ticking — must leave every measurement standing.
#[test]
fn paint_only_value_change_keeps_measurements() {
    init_executor();
    let value = binding(0.5_f64);
    let on = binding(true);
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((
            slider("Volume", &value),
            Toggle::new("Ready", &on),
        ))),
        &env,
        220.0,
        80.0,
    );

    value.set(0.9);
    on.set(false);
    let frame = renderer.refresh_tree(220.0, 80.0);
    assert_eq!(
        frame.work().measures_computed,
        0,
        "paint-only state changes must not invalidate any measurement"
    );
    assert!(
        frame.work().measures_reused > 0,
        "the caches answer the frame's probes as usual"
    );
}

/// Scene content can resize itself — a swapped document, a data-driven
/// canvas — and its intrinsic size is a measurement input the node re-reads
/// rather than trusts to be constant.
#[test]
fn scene_content_resize_invalidates_its_measure() {
    struct TestContent {
        intrinsic: Rc<Cell<Option<Size>>>,
        invalidator: Rc<RefCell<Option<SceneInvalidator>>>,
    }

    impl SceneContent for TestContent {
        fn build_scene(&mut self, _scene: &mut dyn Scene2D, _width: f32, _height: f32) -> bool {
            false
        }

        fn set_invalidator(&mut self, invalidator: Option<SceneInvalidator>) {
            *self.invalidator.borrow_mut() = invalidator;
        }

        fn intrinsic_size(&self) -> Option<Size> {
            self.intrinsic.get()
        }
    }

    init_executor();
    let intrinsic = Rc::new(Cell::new(Some(Size::new(40.0, 40.0))));
    let invalidator = Rc::new(RefCell::new(None));
    let env = Environment::new();
    let mut renderer = test_renderer();
    renderer.render_tree(
        AnyView::new(vstack((
            SceneView::new(TestContent {
                intrinsic: Rc::clone(&intrinsic),
                invalidator: Rc::clone(&invalidator),
            }),
            text("static sibling"),
        ))),
        &env,
        200.0,
        120.0,
    );
    let before = root_size(&renderer);

    let _ = renderer.signals().take_patch_request();
    intrinsic.set(Some(Size::new(80.0, 60.0)));
    (invalidator
        .borrow()
        .as_ref()
        .expect("the retained scene installs its invalidator at build"))();
    assert!(
        renderer.signals().take_patch_request(),
        "the content invalidator must request a frame"
    );
    let frame = renderer.refresh_tree(200.0, 120.0);
    assert!(
        frame.work().measures_computed > 0,
        "an intrinsic-size change must invalidate the scene's measurement"
    );
    let after = root_size(&renderer);
    assert!(
        after.height > before.height,
        "the resized scene must measure at its new intrinsic size"
    );
}
