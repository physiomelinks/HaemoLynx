"""The library reports through `logging`; only the command line prints.

A `print` inside a library is output the caller cannot turn off, redirect, or
raise the threshold on, and it is invisible to anything that captures logs. The
tests here pin both halves of the rule: no library module prints, and the
records it emits carry the level that matches what they say.
"""
from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import networkx as nx
import pytest

from haemolynx.haemodynamics.poiseuille import PoiseuilleModel
from haemolynx.parsers import Schema, Setting, configure_console_logging, print_settings
from haemolynx.parsers.cli import CONSOLE_LOG_FORMAT

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "haemolynx"

#: The command line is allowed to print: everything these two modules write is
#: the answer to a flag the user typed (`--list-settings`, `--list-presets`,
#: the pre-run checklist), so it belongs on stdout whether or not the caller
#: has configured logging.
MODULES_THAT_MAY_PRINT = {
    Path("parsers/cli.py"),
    Path("parsers/checks.py"),
}


def _print_calls(source: str) -> list[int]:
    """Line numbers of every `print(...)` call in *source*."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def _library_modules() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_no_library_module_prints() -> None:
    offenders = {}
    for module in _library_modules():
        relative = module.relative_to(PACKAGE_ROOT)
        if relative in MODULES_THAT_MAY_PRINT:
            continue
        lines = _print_calls(module.read_text(encoding="utf-8"))
        if lines:
            offenders[str(relative)] = lines
    assert offenders == {}, (
        f"print() in library code: {offenders}. Use the module's "
        "logging.getLogger(__name__) instead, so a caller can control it."
    )


def test_the_command_line_still_prints() -> None:
    """The exemption is real, not a leftover: those modules do print."""
    for relative in MODULES_THAT_MAY_PRINT:
        source = (PACKAGE_ROOT / relative).read_text(encoding="utf-8")
        assert _print_calls(source), f"{relative} no longer prints; drop the exemption."


def test_library_modules_use_a_module_level_logger() -> None:
    """Loggers are named for their module, so a caller can filter by subpackage."""
    for module in _library_modules():
        source = module.read_text(encoding="utf-8")
        if "logger = logging.getLogger" not in source:
            continue
        assert "logging.getLogger(__name__)" in source, (
            f"{module.relative_to(PACKAGE_ROOT)} names its logger explicitly; use "
            "__name__ so the name follows the module."
        )


def test_no_library_module_configures_logging() -> None:
    """Only an application chooses where log records go."""
    offenders = [
        str(module.relative_to(PACKAGE_ROOT))
        for module in _library_modules()
        if "logging.basicConfig" in module.read_text(encoding="utf-8")
        and module.relative_to(PACKAGE_ROOT) not in MODULES_THAT_MAY_PRINT
    ]
    assert offenders == [], (
        f"logging.basicConfig in library code: {offenders}. Configuring logging is "
        "the caller's decision; expose it through parsers.configure_console_logging."
    )


def _network_with_one_capillary() -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 100.0))
    G.add_edge(0, 1, length=100.0, branch_order="B01")
    return G


def test_resistance_assignment_reports_progress_at_info(caplog) -> None:
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    with caplog.at_level(logging.INFO, logger="haemolynx.haemodynamics.poiseuille"):
        _, results = model.set_poiseuille_resistances(
            _network_with_one_capillary(), {"B01": 5.0}
        )

    assert results["edges_set"] == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("Edges assigned resistance: 1" in message for message in messages)
    assert {record.levelno for record in caplog.records} == {logging.INFO}


def test_an_edge_left_without_a_resistance_is_a_warning(caplog) -> None:
    """A skipped edge is a recoverable problem, so it must not hide at INFO."""
    G = _network_with_one_capillary()
    G.add_edge(0, 1, length=100.0)  # no branch_order, so no diameter to use

    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    with caplog.at_level(logging.INFO, logger="haemolynx.haemodynamics.poiseuille"):
        _, results = model.set_poiseuille_resistances(G, {"B01": 5.0})

    assert len(results["missing_branch_order"]) == 1
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any("Edges missing branch_order: 1" in message for message in warnings)


def test_nothing_is_emitted_when_the_caller_has_not_configured_logging(capsys) -> None:
    """A library import must not turn a caller's stdout into a progress log."""
    logging.getLogger("haemolynx").handlers.clear()
    model = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)
    model.set_poiseuille_resistances(_network_with_one_capillary(), {"B01": 5.0})
    assert capsys.readouterr().out == ""


def test_configure_console_logging_sends_records_to_stdout() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    root.handlers.clear()
    try:
        configure_console_logging()
        handler = root.handlers[-1]
        assert handler.stream is sys.stdout
        assert handler.formatter._fmt == CONSOLE_LOG_FORMAT
        assert root.level == logging.INFO

        root.handlers.clear()
        configure_console_logging(verbose=True)
        assert root.level == logging.DEBUG
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


class _Cp1252Stream:
    """A stdout stand-in whose encoding cannot represent Greek letters."""

    encoding = "cp1252"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def test_console_logging_replaces_characters_the_console_cannot_encode() -> None:
    from haemolynx.parsers.cli import _ConsoleHandler

    stream = _Cp1252Stream()
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(
        name="haemolynx",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="resistance / (π * diameter^4) in μm",
        args=(),
        exc_info=None,
    )
    handler.emit(record)
    assert stream.written
    assert "diameter^4" in stream.written[0]
    stream.written[0].encode("cp1252")


def test_list_settings_writes_to_stdout(capsys) -> None:
    """`--list-settings` is an answer to a flag, so it prints regardless of logging."""
    schema = Schema([Setting("voxel_size_um", "float", 1.0, "Voxel edge length", "Input")])
    print_settings(schema, {"voxel_size_um": 0.5})
    out = capsys.readouterr().out
    assert "Input" in out
    assert "voxel_size_um" in out
    assert "0.5" in out
