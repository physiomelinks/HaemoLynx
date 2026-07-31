"""A plain pytest run must never open a browser tab."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from browser_diagnostics import (
    OPEN_TEST_HTML_ENV_VAR,
    browser_display_requested,
    open_diagnostic_html,
)

TESTS_DIR = Path(__file__).resolve().parent
HELPER_MODULE = TESTS_DIR / "browser_diagnostics.py"


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "diagnostic.html"
    path.write_text("<html></html>", encoding="utf-8")
    return path


@pytest.fixture
def recorded_tabs(monkeypatch) -> list[str]:
    """Capture browser opens instead of performing them."""
    opened: list[str] = []

    def _record(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("browser_diagnostics.webbrowser.open_new_tab", _record)
    return opened


def test_no_tab_is_opened_by_default(monkeypatch, html_file, recorded_tabs):
    monkeypatch.delenv(OPEN_TEST_HTML_ENV_VAR, raising=False)
    assert open_diagnostic_html(html_file) is False
    assert recorded_tabs == []


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_falsey_settings_do_not_open_a_tab(monkeypatch, html_file, recorded_tabs, value):
    monkeypatch.setenv(OPEN_TEST_HTML_ENV_VAR, value)
    assert open_diagnostic_html(html_file) is False
    assert recorded_tabs == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opting_in_opens_the_file(monkeypatch, html_file, recorded_tabs, value):
    monkeypatch.setenv(OPEN_TEST_HTML_ENV_VAR, value)
    assert open_diagnostic_html(html_file) is True
    assert recorded_tabs == [html_file.resolve().as_uri()]
    assert browser_display_requested() is True


def test_a_browser_failure_never_fails_the_calling_test(monkeypatch, html_file):
    monkeypatch.setenv(OPEN_TEST_HTML_ENV_VAR, "1")

    def _explode(url: str) -> bool:
        raise OSError("no browser on this host")

    monkeypatch.setattr("browser_diagnostics.webbrowser.open_new_tab", _explode)
    assert open_diagnostic_html(html_file) is False


def test_no_test_module_opens_a_browser_directly():
    """The guard: every auto-open must route through the opt-in helper.

    Three test modules used to call ``webbrowser`` unconditionally, so a plain
    run hijacked the screen with a tab per test.
    """
    direct_call = re.compile(r"webbrowser\s*\.\s*open")
    offenders = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path in (HELPER_MODULE, Path(__file__).resolve()):
            continue
        if direct_call.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert offenders == [], (
        "These test modules open a browser directly; call "
        f"browser_diagnostics.open_diagnostic_html instead: {offenders}"
    )
