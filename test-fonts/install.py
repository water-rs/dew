#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fetch dew's test fonts from a pinned, hash-verified upstream release.

The shaping tests and the vending-machine simulation read these faces from
this directory at test time; a machine that has not run this script fails
those tests with an error naming it. Run once per checkout, from anywhere:

    uv run test-fonts/install.py

The release archive is pinned by URL and verified by SHA-256, and every font
extracted from it is verified against its expected SHA-256 — the font bytes
are part of the test contract, so a drifted download is a hard failure, never
a silent replacement. The fonts are binary assets and stay out of the
repository; `test-fonts/*.ttf` is gitignored.

Re-running is idempotent: a font already present byte-identically is left in
place; anything else is replaced.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# The Roboto release the `water` CLI's font registry ships to applications —
# the same archive WaterUI's Hydrolysis test fonts are taken from.
ROBOTO_URL = "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-android.zip"
ROBOTO_SHA256 = "c825453253f590cfe62557733e7173f9a421fff103b00f57d33c4ad28ae53baf"

# Expected SHA-256 of every file this script writes.
EXPECTED = {
    "Roboto-Regular.ttf": "797e35f7f5d6020a5c6ea13b42ecd668bcfb3bbc4baa0e74773527e5b6cb3174",
    "Roboto-Bold.ttf": "36f3709dea3e3ce3c6aedc058079e55980825f898f1e901d091c73c40de8bab1",
}

FETCH_ATTEMPTS = 4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, expected_sha256: str) -> bytes:
    """Download `url` and verify its SHA-256.

    Transient network errors are retried with backoff, as runners see them;
    the last failure, and any hash mismatch, exits non-zero.
    """
    print(f"fetch {url}")
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            break
        except OSError as error:
            if attempt == FETCH_ATTEMPTS:
                raise SystemExit(
                    f"fetch {url} failed after {FETCH_ATTEMPTS} attempts: {error}"
                )
            delay = 2**attempt
            print(f"  attempt {attempt} failed ({error}); retrying in {delay}s")
            time.sleep(delay)
    digest = sha256(data)
    if digest != expected_sha256:
        raise SystemExit(
            f"sha256 mismatch for {url}:\n  got      {digest}\n  expected {expected_sha256}"
        )
    return data


def verify(name: str, data: bytes) -> None:
    digest = sha256(data)
    if digest != EXPECTED[name]:
        raise SystemExit(
            f"{name}: extracted sha256 {digest} != expected {EXPECTED[name]}\n"
            "the font bytes are part of the test contract — investigate the "
            "source instead of accepting the output"
        )


def write(name: str, data: bytes) -> None:
    dest = OUT_DIR / name
    # Write beside the destination and rename over it, so an interrupted run
    # never leaves a truncated font for a test to read.
    partial = dest.with_name(f"{name}.partial")
    partial.write_bytes(data)
    os.replace(partial, dest)
    print(f"  wrote {name} ({len(data)} bytes, {sha256(data)[:12]}…)")


def installed(name: str) -> bool:
    dest = OUT_DIR / name
    return dest.is_file() and sha256(dest.read_bytes()) == EXPECTED[name]


def main() -> None:
    missing = [name for name in sorted(EXPECTED) if not installed(name)]
    for name in sorted(EXPECTED):
        if name not in missing:
            print(f"  {name}: already installed")
    if missing:
        archive = fetch(ROBOTO_URL, ROBOTO_SHA256)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            fonts = {name: zf.read(name) for name in missing}
        # Verify every face before writing any, so a drifted archive leaves
        # the directory as it was.
        for name, data in fonts.items():
            verify(name, data)
        for name, data in fonts.items():
            write(name, data)
    print(f"all {len(EXPECTED)} test fonts verified in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
