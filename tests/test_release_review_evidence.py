"""The evidence audit: cited paths read from a row, and looked for."""
from __future__ import annotations

from release_review import evidence


def test_paths_are_read_from_backticks_only_where_they_can_be_placed():
    text = (
        "E14.5 (`outputs/p6/o2_current/solve_summary.json`), `test_flow_solvers.py:125, 207`, "
        "`pipeline/checks.py:595-616`, `{stem}_run_manifest.json`, `hang_debug.*`, `REPORT.md`, "
        "`graph.FLOW_STATE`, `outputs/release_review/2026-10-15_9b679b8/`, `tests/test_x.py:12`"
    )
    found = {(each.path, each.line, each.end) for each in evidence.citations(text)}
    assert found == {
        ("outputs/p6/o2_current/solve_summary.json", None, None),
        ("test_flow_solvers.py", 125, None),  # the first of the lines it lists
        ("pipeline/checks.py", 595, 616),
        ("outputs/release_review/2026-10-15_9b679b8/", None, None),
        ("tests/test_x.py", 12, None),
    }


def _tree(tmp_path):
    root = tmp_path / "tree"
    (root / "src" / "haemolynx" / "pipeline").mkdir(parents=True)
    (root / "src" / "haemolynx" / "parsers").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "haemolynx" / "pipeline" / "checks.py").write_text("x = 1\n" * 20)
    (root / "src" / "haemolynx" / "pipeline" / "schema.py").write_text("x = 1\n" * 10)
    (root / "src" / "haemolynx" / "parsers" / "schema.py").write_text("x = 1\n" * 500)
    (root / "tests" / "test_flow.py").write_text("def test():\n    pass\n")
    outputs = tmp_path / "main"
    (outputs / "outputs" / "kept").mkdir(parents=True)
    (outputs / "outputs" / "kept" / "REPORT.md").write_text("ok")
    return evidence.Tree(root, [outputs])


def test_the_audit_finds_missing_files_and_lines_past_the_end(tmp_path):
    tree = _tree(tmp_path)
    rows = [
        ("flow.2a", "`outputs/p6/o2_current/solve_summary.json`"),
        ("graph.6a", "`outputs/kept/REPORT.md`, `test_flow.py:2`"),
        ("robust.2a", "`pipeline/checks.py:595-616`"),
        ("stats.2a", "`schema.py:400`"),  # parsers/schema.py is long enough
        ("graph.6b", "`grid_P/marks.csv` beside the key"),  # relative to an earlier folder
        ("input.4a", "`test_gone.py`"),
    ]
    findings = evidence.audit(rows, tree, check_outputs=True)
    assert [(each.check_id, each.problem) for each in findings] == [
        ("flow.2a", "missing"),
        ("robust.2a", "past the end"),
        ("input.4a", "missing"),
    ]
    assert evidence.missing_by_check(findings) == {
        "flow.2a": ["outputs/p6/o2_current/solve_summary.json"],
        "input.4a": ["test_gone.py"],
    }


def test_a_bare_name_is_found_at_the_root_too(tmp_path):
    tree = _tree(tmp_path)
    (tree.root / "hang_debug.py").write_text("x = 1\n")
    assert evidence.audit([("code.6c", "`hang_debug.py` at the root")], tree, check_outputs=True) == []


def test_outputs_go_unchecked_where_there_are_none(tmp_path):
    """In CI the git-ignored outputs do not exist; their absence proves nothing."""
    findings = evidence.audit([("flow.2a", "`outputs/p6/x.json`")], _tree(tmp_path), check_outputs=False)
    assert findings == []
