//! The one `WorkingColor` → `peniko::Color` conversion every paint on dew
//! goes through.

use waterui_graphics::color::{WorkingColor, working};

/// `color` as the sRGB `peniko::Color` the vello pipeline paints: the RGB
/// channels through the working-space → sRGB conversion, the alpha channel
/// carried raw — the sRGB transfer does not touch alpha.
pub fn to_peniko(color: WorkingColor) -> peniko::Color {
    let srgb = working::to_srgb(color);
    peniko::Color::new([srgb.red, srgb.green, srgb.blue, color.components[3]])
}
