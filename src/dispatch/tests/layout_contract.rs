//! The proposal-aware placement contract, proven on the retained tree.
//!
//! `Layout::place` answers both a frame and the proposal each child was
//! selected under, and equal bounds can carry different proposals. These
//! tests put a recording layout behind real dew nodes and check that the
//! offer the layout sees at render time is the one the parent was placed
//! under — not one reconstructed from the bounds it happens to share with
//! another offer.

use core::cell::{Cell, RefCell};
use std::rc::Rc;

use kurbo::Affine;
use nami::Computed;
use waterui_backend_core::frame_signals::FrameSignals;
use waterui_backend_core::time::Instant;
use waterui_core::accessibility::AccessibilityLabel;
use waterui_core::layout::{
    Layout, LayoutPriority, Point, ProposalSize, Rect, Size, SubView, SubviewPlacement,
};
use waterui_core::{AnyView, Environment, IgnorableMetadata, Metadata, Retain, Str, View};
use waterui_graphics::color::Color;
use waterui_layout::container::FixedContainer;
use waterui_layout::stack::Axis;
use waterui_layout::{Spacer, scroll_horizontal, spacer, spacer_min};

use crate::dispatch::{DewNode, DewRenderer, RenderContext, build_node};

/// What [`ProbeLayout::place`] observed on one call: the proposal it was
/// handed and the frames it produced.
#[derive(Debug)]
struct Observation {
    proposal: ProposalSize,
    frames: Vec<Rect>,
}

type Trace = Rc<RefCell<Vec<Observation>>>;

/// A two-child layout that splits its bounds according to the proposal's
/// main-axis offer: equal widths when a width is offered, and one quarter
/// versus three quarters when the axis was left unspecified.
#[derive(Debug)]
struct ProbeLayout {
    trace: Trace,
}

impl Layout for ProbeLayout {
    fn size_that_fits(&self, _proposal: ProposalSize, _children: &[&dyn SubView]) -> Size {
        Size::new(160.0, 20.0)
    }

    fn place(
        &self,
        bounds: Rect,
        proposal: ProposalSize,
        children: &[&dyn SubView],
    ) -> Vec<SubviewPlacement> {
        assert_eq!(children.len(), 2);
        let extents = if proposal.width.is_none() {
            [40.0, 120.0]
        } else {
            [80.0, 80.0]
        };
        let mut cursor = 0.0;
        let placements: Vec<_> = extents
            .into_iter()
            .map(|extent| {
                let origin = Point::new(bounds.x() + cursor, bounds.y());
                cursor += extent;
                let size = Size::new(extent, 20.0);
                SubviewPlacement::new(
                    Rect::new(origin, size),
                    ProposalSize::new(Some(size.width), Some(size.height)),
                )
            })
            .collect();
        self.trace.borrow_mut().push(Observation {
            proposal,
            frames: placements.iter().map(|placement| placement.frame).collect(),
        });
        placements
    }
}

/// A view whose retained node tree is `ProbeLayout` over two colour leaves.
///
/// `builds` counts body evaluations: the retained tree is built once and
/// relaid out through `patch`/`render`, so a second placement must never
/// come from a second body.
struct ProbeContent {
    trace: Trace,
    builds: Rc<Cell<usize>>,
}

impl View for ProbeContent {
    fn body(self, _env: &Environment) -> impl View {
        self.builds.set(self.builds.get() + 1);
        FixedContainer::new(
            ProbeLayout { trace: self.trace },
            (Color::srgb_hex("#2563EB"), Color::srgb_hex("#DC2626")),
        )
    }
}

/// A retained node under test together with the trace its layout writes.
struct Fixture {
    node: Box<dyn DewNode>,
    trace: Trace,
    builds: Rc<Cell<usize>>,
}

impl Fixture {
    fn new<V: View>(
        wrap: impl FnOnce(ProbeContent) -> V,
        renderer: &mut DewRenderer,
        env: &Environment,
    ) -> Self {
        let trace = Rc::new(RefCell::new(Vec::new()));
        let builds = Rc::new(Cell::new(0));
        let content = ProbeContent {
            trace: Rc::clone(&trace),
            builds: Rc::clone(&builds),
        };
        Self {
            node: build_node(renderer, AnyView::new(wrap(content)), env, 0),
            trace,
            builds,
        }
    }

    /// One relayout of the retained tree under `proposal`, through the same
    /// patch-then-render order `DewRenderer::refresh_tree` uses.
    fn layout(&mut self, renderer: &mut DewRenderer, proposal: ProposalSize, size: Size) {
        self.node.patch(renderer);
        self.node.render(
            renderer,
            RenderContext {
                transform: Affine::IDENTITY,
                bounds: kurbo::Rect::new(0.0, 0.0, f64::from(size.width), f64::from(size.height)),
                proposal,
            },
        );
    }

    /// The last placement the retained subtree produced: the proposal
    /// `place` ran under, the frames it answered, and one body evaluation
    /// for the whole tree's life.
    fn assert_placement(&self, proposal: ProposalSize, extents: [f32; 2]) {
        let trace = self.trace.borrow();
        let observation = trace.last().expect("the retained subtree must be placed");
        assert_eq!(observation.proposal, proposal);
        assert_eq!(
            observation
                .frames
                .iter()
                .map(|frame| *frame.size())
                .collect::<Vec<_>>(),
            extents.map(|extent| Size::new(extent, 20.0)).to_vec(),
        );
        assert_eq!(self.builds.get(), 1);
    }
}

fn test_renderer() -> DewRenderer {
    DewRenderer::new(FrameSignals::new(Instant::now()), crate::test_fonts())
}

/// Equal 160×20 bounds under `None` and `Some(160)` main-axis offers place
/// the children 40/120 and 80/80 — and revisiting an offer after other
/// probes and relayouts reproduces its own answer, so the proposal the
/// layout sees is the selected one, not an accident of bounds or order.
#[test]
fn equal_bounds_keep_the_selected_proposal_after_other_probes() {
    let env = Environment::new();
    let mut renderer = test_renderer();
    let mut fixture = Fixture::new(|content| content, &mut renderer, &env);
    for main in [None, Some(160.0), None] {
        for probe in [Some(0.0), Some(80.0), Some(f32::INFINITY)] {
            let dimensions = fixture
                .node
                .measure(renderer.state_cell(), ProposalSize::new(probe, Some(20.0)));
            assert_eq!(dimensions.size, Size::new(160.0, 20.0));
        }
        fixture.trace.borrow_mut().clear();
        let proposal = ProposalSize::new(main, Some(20.0));
        fixture.layout(&mut renderer, proposal, Size::new(160.0, 20.0));
        fixture.assert_placement(
            proposal,
            if main.is_none() {
                [40.0, 120.0]
            } else {
                [80.0, 80.0]
            },
        );
    }
}

/// A horizontal scroll measures and renders its content under a cleared
/// main-axis offer: the content's intrinsic width is its own, however wide
/// the viewport happens to be.
#[test]
fn scroll_preserves_its_unconstrained_content_axis() {
    let env = Environment::new();
    let mut renderer = test_renderer();
    let mut fixture = Fixture::new(scroll_horizontal, &mut renderer, &env);
    fixture.layout(
        &mut renderer,
        ProposalSize::new(Some(160.0), Some(40.0)),
        Size::new(160.0, 40.0),
    );
    fixture.assert_placement(ProposalSize::new(None, Some(40.0)), [40.0, 120.0]);
}

/// `Retain`, `LayoutPriority` and accessibility naming are all transparent
/// for layout: stacked on one subtree, the layout inside still sees the
/// selected proposal.
#[test]
fn transparent_metadata_preserves_the_selected_proposal() {
    let env = Environment::new();
    let mut renderer = test_renderer();
    let mut fixture = Fixture::new(
        |content| {
            Metadata::new(
                IgnorableMetadata::new(
                    Metadata::new(content, Retain::new(())),
                    AccessibilityLabel::new(Computed::constant(Str::from_static("probe"))),
                ),
                LayoutPriority::new(3),
            )
        },
        &mut renderer,
        &env,
    );
    let proposal = ProposalSize::new(None, Some(20.0));
    fixture.layout(&mut renderer, proposal, Size::new(160.0, 20.0));
    fixture.assert_placement(proposal, [40.0, 120.0]);
}

/// The priorities a container reads off its children come from the nodes:
/// `i32::MIN` for a bare or transparently wrapped spacer, and the explicit
/// value when `.layout_priority` overrides it.
#[test]
fn spacer_default_priority_survives_wrappers_and_explicit_overrides() {
    let env = Environment::new();
    let mut renderer = test_renderer();
    for explicit in [None, Some(0), Some(7)] {
        let priority = Rc::new(Cell::new(None));
        let wrapped = Metadata::new(spacer(), Retain::new(()));
        let content = match explicit {
            Some(value) => AnyView::new(Metadata::new(wrapped, LayoutPriority::new(value))),
            None => AnyView::new(wrapped),
        };
        let mut node = build_node(
            &mut renderer,
            AnyView::new(FixedContainer::new(
                PriorityProbe(Rc::clone(&priority)),
                vec![content],
            )),
            &env,
            0,
        );
        node.render(
            &mut renderer,
            RenderContext {
                transform: Affine::IDENTITY,
                bounds: kurbo::Rect::new(0.0, 0.0, 160.0, 20.0),
                proposal: ProposalSize::new(Some(160.0), Some(20.0)),
            },
        );
        assert_eq!(
            priority.get(),
            Some(explicit.unwrap_or(Spacer::DEFAULT_LAYOUT_PRIORITY)),
        );
    }
}

/// A spacer measures its minimum length on the enclosing stack's main axis
/// and zero on the cross axis — the floor the stack keeps under compression
/// and expands from when it distributes surplus — and claims nothing outside
/// a stack.
#[test]
fn spacer_reports_its_minimum_length_and_default_priority() {
    let mut renderer = test_renderer();
    let measure = |renderer: &mut DewRenderer, view: AnyView, env: &Environment| {
        let node = build_node(renderer, view, env, 0);
        assert_eq!(node.priority(), Spacer::DEFAULT_LAYOUT_PRIORITY);
        node.measure(renderer.state_cell(), ProposalSize::UNSPECIFIED)
            .size
    };
    let mut column = Environment::new();
    column.insert(Axis::Vertical);
    let mut row = Environment::new();
    row.insert(Axis::Horizontal);
    let outside = Environment::new();
    assert_eq!(
        measure(&mut renderer, AnyView::new(spacer_min(12.0)), &column),
        Size::new(0.0, 12.0),
    );
    assert_eq!(
        measure(&mut renderer, AnyView::new(spacer_min(12.0)), &row),
        Size::new(12.0, 0.0),
    );
    assert_eq!(
        measure(&mut renderer, AnyView::new(spacer_min(12.0)), &outside),
        Size::new(0.0, 0.0),
    );
    assert_eq!(
        measure(&mut renderer, AnyView::new(spacer()), &column),
        Size::new(0.0, 0.0),
    );
}

/// Records the priority one child reports to its container.
#[derive(Debug)]
struct PriorityProbe(Rc<Cell<Option<i32>>>);

impl Layout for PriorityProbe {
    fn size_that_fits(&self, proposal: ProposalSize, children: &[&dyn SubView]) -> Size {
        children[0].measure(proposal).size
    }

    fn place(
        &self,
        bounds: Rect,
        proposal: ProposalSize,
        children: &[&dyn SubView],
    ) -> Vec<SubviewPlacement> {
        self.0.set(Some(children[0].priority()));
        vec![SubviewPlacement::new(
            Rect::new(
                Point::new(bounds.x(), bounds.y()),
                children[0].measure(proposal).size,
            ),
            proposal,
        )]
    }
}
