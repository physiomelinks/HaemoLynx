"""Opt-in browser display for HTML diagnostics written by tests.

Several tests write rotatable plotly diagnostics that are genuinely useful to
look at while debugging an assignment or boundary-labelling failure. Opening
them automatically, though, means a plain ``pytest`` run hijacks the screen with
a browser tab per test. Display is therefore opt-in, mirroring how the plotting
tests only display when the matplotlib backend can (``plot.backend_can_display``).

Set the environment variable to look at them::

    HAEMOLYNX_OPEN_TEST_HTML=1 pytest tests/test_synthetic_vessel_assignment_pipeline.py

The HTML is written either way, so the files are there to open by hand.
"""
from __future__ import annotations

import os
import webbrowser
from pathlib import Path

OPEN_TEST_HTML_ENV_VAR = "HAEMOLYNX_OPEN_TEST_HTML"

_TRUTHY = {"1", "true", "yes", "on"}


def browser_display_requested() -> bool:
    """True when the environment opts in to opening diagnostics in a browser."""
    return os.environ.get(OPEN_TEST_HTML_ENV_VAR, "").strip().lower() in _TRUTHY


def open_diagnostic_html(html_path: Path | str) -> bool:
    """Open *html_path* in a browser tab, but only when explicitly requested.

    Returns True when a tab was opened. Never raises: a failure to open a
    diagnostic must not fail the test that produced it.
    """
    if not browser_display_requested():
        return False
    try:
        return bool(webbrowser.open_new_tab(Path(html_path).resolve().as_uri()))
    except Exception:
        # Headless hosts, no browser installed, sandboxed CI.
        return False
