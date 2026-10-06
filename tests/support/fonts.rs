//! The repository's test faces, read from the directory `test-fonts/install.py`
//! fills.
//!
//! Shared by the crate's unit tests (through `#[path]`) and the integration
//! tests' `support` module, so every caller fails the same way on a checkout
//! that has not fetched them.

use std::path::PathBuf;

/// The command that fetches the test fonts, run from the repository root.
const INSTALL_COMMAND: &str = "uv run test-fonts/install.py";

/// Reads the test face `file_name` (for example `Roboto-Regular.ttf`).
///
/// # Panics
///
/// Panics, naming the installer command, when the face cannot be read — the
/// fonts are fetched rather than committed, so a missing face means the
/// checkout has not run the installer, never that the test should be skipped.
#[allow(
    dead_code,
    reason = "compiled into the lib's unit tests (via `#[path]`) and every integration test binary, not all of which read a face"
)]
pub fn read_test_font(file_name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-fonts")
        .join(file_name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "test font `{}` could not be read ({error}); fetch the pinned test fonts with `{INSTALL_COMMAND}` from the repository root",
            path.display()
        )
    })
}
