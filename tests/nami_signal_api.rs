//! The nami pin regression: dew's manifest declares `nami = "=0.11.1"` and
//! `[patch.crates-io]` redirects it to the rev carrying nami#26, where
//! `Signal::get` is `Signal::snapshot` and `Binding::get` is gone. A graph
//! that resolves any other nami — a pre-rename release or a second copy —
//! fails to compile this file rather than silently dropping the pin.

use nami::{Signal, SignalExt as _, binding};

#[test]
fn bindings_read_through_snapshot() {
    let value = binding(41i32);
    assert_eq!(value.snapshot(), 41);
    value.set(42);
    assert_eq!(value.snapshot(), 42);
}

#[test]
fn computed_reads_through_snapshot() {
    let value = binding(2i32);
    let doubled = value.map(|v: i32| v * 2);
    assert_eq!(doubled.snapshot(), 4);
}
