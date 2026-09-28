//! Persistent [`ScrollView`] node.

use core::cell::{Cell, RefCell};

use nami::watcher::BoxWatcherGuard;
use nami::{Binding, Signal};
use waterui_core::Environment;
use waterui_core::layout::{
    Point, ProposalSize, Rect as LayoutRect, Size, StretchAxis, SubviewPlacement, ViewDimensions,
};
use waterui_layout::scroll::{Axis, ScrollController, ScrollView, ScrollViewParts};

use crate::dispatch::{DewNode, DewRenderer, RenderContext, build_node};
use crate::text::DewState;
use crate::views::to_f32;

struct ScrollNode {
    axis: Axis,
    child: Box<dyn DewNode>,
    controller: Option<ScrollController<Point>>,
    report_offset: Option<Binding<Point>>,
    applied_scroll_generation: Cell<i32>,
    offset: Cell<Point>,
    _controller_guard: Option<BoxWatcherGuard>,
}

pub fn build(
    renderer: &mut DewRenderer,
    scroll: ScrollView,
    env: &Environment,
    depth: usize,
) -> Box<dyn DewNode> {
    let ScrollViewParts {
        axis,
        content,
        controller,
        offset: report_offset,
        ..
    } = scroll.into_inner();
    let controller_guard = controller.as_ref().map(|controller| {
        let signals = renderer.signals();
        controller.generation().watch(move |_| {
            signals.request_refresh();
        })
    });
    Box::new(ScrollNode {
        axis,
        child: build_node(renderer, content, env, depth),
        controller,
        report_offset,
        applied_scroll_generation: Cell::new(0),
        offset: Cell::new(Point::zero()),
        _controller_guard: controller_guard,
    })
}

impl DewNode for ScrollNode {
    fn measure(&self, state: &RefCell<DewState>, proposal: ProposalSize) -> ViewDimensions {
        let intrinsic = self
            .child
            .measure(state, content_proposal(self.axis, proposal))
            .size;
        ViewDimensions::new(Size::new(
            proposal.width.unwrap_or(intrinsic.width),
            proposal.height.unwrap_or(intrinsic.height),
        ))
    }

    fn render(&mut self, renderer: &mut DewRenderer, ctx: RenderContext) {
        let viewport = ctx.bounds;
        // The offer `measure` made to the content, verbatim: the scroll axis
        // stays unspecified so the content keeps its intrinsic extent, and
        // the placement below carries it rather than an offer reconstructed
        // from the viewport the content happens to overfill.
        let offer = content_proposal(self.axis, ctx.proposal);
        let intrinsic = self.child.measure(renderer.state_cell(), offer).size;
        let (content_width, content_height) = content_size(self.axis, viewport, intrinsic);
        if let Some(controller) = &self.controller {
            let generation = controller.generation().snapshot();
            if generation != self.applied_scroll_generation.get() {
                let target = controller.target().snapshot();
                let max_x = (content_width - to_f32(viewport.width())).max(0.0);
                let max_y = (content_height - to_f32(viewport.height())).max(0.0);
                let offset = match self.axis {
                    Axis::Horizontal => Point::new(target.x.clamp(0.0, max_x), 0.0),
                    Axis::Vertical => Point::new(0.0, target.y.clamp(0.0, max_y)),
                    Axis::All => Point::new(target.x.clamp(0.0, max_x), target.y.clamp(0.0, max_y)),
                    _ => panic!("dew does not support scroll axis {:?}", self.axis),
                };
                self.set_offset(offset);
                self.applied_scroll_generation.set(generation);
            }
        }
        let clip = ctx.transform.transform_rect_bbox(viewport);
        renderer.list_mut().push_clip(clip);
        let offset = self.offset.get();
        self.child.render(
            renderer,
            ctx.child(SubviewPlacement::new(
                LayoutRect::new(
                    Point::new(-offset.x, -offset.y),
                    Size::new(content_width, content_height),
                ),
                offer,
            )),
        );
        renderer.list_mut().pop_clip();
    }

    fn stretch_axis(&self) -> StretchAxis {
        StretchAxis::Both
    }

    fn patch(&mut self, renderer: &mut DewRenderer) -> bool {
        self.child.patch(renderer)
    }
}

impl ScrollNode {
    /// Moves the content offset, reporting the new position into the
    /// `report_offset` binding when one is connected. The binding is dew's
    /// answer back to the app — written, never read — so only an actual
    /// change produces a write; programmatic scroll goes through the
    /// controller.
    fn set_offset(&self, offset: Point) {
        if self.offset.get() == offset {
            return;
        }
        self.offset.set(offset);
        if let Some(report) = &self.report_offset {
            report.set(offset);
        }
    }
}

fn content_proposal(axis: Axis, proposal: ProposalSize) -> ProposalSize {
    match axis {
        Axis::Horizontal => ProposalSize::new(None, proposal.height),
        Axis::Vertical => ProposalSize::new(proposal.width, None),
        Axis::All => ProposalSize::UNSPECIFIED,
        _ => panic!("dew does not support scroll axis {axis:?}"),
    }
}

fn content_size(axis: Axis, viewport: kurbo::Rect, intrinsic: Size) -> (f32, f32) {
    let viewport_width = to_f32(viewport.width());
    let viewport_height = to_f32(viewport.height());
    match axis {
        Axis::Horizontal => (intrinsic.width.max(viewport_width), viewport_height),
        Axis::Vertical => (viewport_width, intrinsic.height.max(viewport_height)),
        Axis::All => (
            intrinsic.width.max(viewport_width),
            intrinsic.height.max(viewport_height),
        ),
        _ => panic!("dew does not support scroll axis {axis:?}"),
    }
}

#[cfg(test)]
mod tests {
    use kurbo::Affine;
    use nami::binding;
    use waterui_backend_core::frame_signals::FrameSignals;
    use waterui_backend_core::time::Instant;
    use waterui_core::layout::{
        Layout, Point, ProposalSize, Rect, Size, SubView, SubviewPlacement,
    };
    use waterui_core::{AnyView, Environment};
    use waterui_graphics::color::Color;
    use waterui_layout::container::FixedContainer;

    use super::*;

    /// A layout whose intrinsic answer is the size it was built with; every
    /// child it holds fills the bounds it was placed under.
    #[derive(Debug)]
    struct ProbeLayout {
        size: Size,
    }

    impl Layout for ProbeLayout {
        fn size_that_fits(&self, _proposal: ProposalSize, _children: &[&dyn SubView]) -> Size {
            self.size
        }

        fn place(
            &self,
            bounds: Rect,
            proposal: ProposalSize,
            children: &[&dyn SubView],
        ) -> Vec<SubviewPlacement> {
            children
                .iter()
                .map(|_| SubviewPlacement::new(bounds, proposal))
                .collect()
        }
    }

    /// Both halves of the `report_offset` contract: a `scroll_to` request on
    /// the controller moves the content — clamped to the overflow — and the
    /// offset that results is written into the binding on every change.
    #[test]
    fn controller_scroll_clamps_and_reports_each_offset() {
        let env = Environment::new();
        let mut renderer = DewRenderer::new(FrameSignals::new(Instant::now()), crate::test_fonts());

        let controller = ScrollController::<Point>::new(Point::zero());
        let report = binding(Point::zero());
        let view = ScrollView::vertical(FixedContainer::new(
            ProbeLayout {
                size: Size::new(160.0, 400.0),
            },
            (Color::srgb_hex("#2563EB"),),
        ))
        .scroll_controller(&controller)
        .report_offset(&report);
        let mut node = build_node(&mut renderer, AnyView::new(view), &env, 0);

        let frame = |node: &mut Box<dyn DewNode>, renderer: &mut DewRenderer| {
            node.render(
                renderer,
                RenderContext {
                    transform: Affine::IDENTITY,
                    bounds: kurbo::Rect::new(0.0, 0.0, 160.0, 100.0),
                    proposal: ProposalSize::new(Some(160.0), Some(100.0)),
                },
            );
        };

        frame(&mut node, &mut renderer);
        assert_eq!(
            report.snapshot(),
            Point::zero(),
            "unscrolled: no offset to report"
        );

        controller.scroll_to(Point::new(0.0, 150.0));
        frame(&mut node, &mut renderer);
        assert_eq!(report.snapshot(), Point::new(0.0, 150.0));

        // 400pt of content under a 100pt viewport overflows by 300: the
        // request clamps, and the binding reports the clamped position.
        controller.scroll_to(Point::new(0.0, 9999.0));
        frame(&mut node, &mut renderer);
        assert_eq!(report.snapshot(), Point::new(0.0, 300.0));

        controller.scroll_to(Point::new(0.0, 50.0));
        frame(&mut node, &mut renderer);
        assert_eq!(report.snapshot(), Point::new(0.0, 50.0));
    }
}
