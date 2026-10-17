"""scripts/check_stale_test_names.py: report test names a document cites that no longer exist in tests/."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_stale_test_names.py"
_spec = importlib.util.spec_from_file_location("check_stale_test_names", SCRIPT)
cs = importlib.util.module_from_spec(_spec)
sys.modules["check_stale_test_names"] = cs
_spec.loader.exec_module(cs)

CLAUDE_MD = """\
# CLAUDE.md

Some text naming `README.md` and `TODO.md`, which are out of date.

## Key references

Treat only the documents below as current. `REMINDERS.md` is stale.

- `cb_modelling_reference.md` — the main reference.
- `a_order.md`, `b_order.md` and `c_order.md` — control flow.
- `hypothesis_testing_methods.md` (gitignored) — the methods.
- `pipeline_rerun_*_notes.md` (untracked) — the re-run packages.
- `examples/preprocessing/README.md` — preprocessing.

### Obsidian vault (outside the repo)

- Start from its `index.md`.

## Agent skills

- `docs/agents/issue-tracker.md`
"""


def test_the_doc_list_is_the_markdown_files_named_in_the_key_references_bullets():
    assert cs.key_reference_patterns(CLAUDE_MD) == [
        "cb_modelling_reference.md",
        "a_order.md",
        "b_order.md",
        "c_order.md",
        "hypothesis_testing_methods.md",
        "pipeline_rerun_*_notes.md",
        "examples/preprocessing/README.md",
    ]


def test_a_missing_key_references_section_raises():
    with pytest.raises(ValueError, match="Key references"):
        cs.key_reference_patterns("# CLAUDE.md\n\n## Rules\n\n- `x.md`\n")


def test_a_pattern_expands_to_the_files_that_match_and_a_plain_name_stays_even_if_absent(tmp_path):
    (tmp_path / "pipeline_rerun_1_notes.md").write_text("x")
    (tmp_path / "pipeline_rerun_2_notes.md").write_text("x")
    got = cs.expand_patterns(["pipeline_rerun_*_notes.md", "gone.md"], tmp_path)
    assert got == ["pipeline_rerun_1_notes.md", "pipeline_rerun_2_notes.md", "gone.md"]


def test_a_pattern_that_matches_nothing_stays_in_the_list_so_it_is_reported_as_absent(tmp_path):
    assert cs.expand_patterns(["pipeline_rerun_*_notes.md"], tmp_path) == ["pipeline_rerun_*_notes.md"]


def test_a_cited_name_with_no_def_is_stale_and_carries_its_line_number_and_text():
    text = "intro\nSee `test_alive` and `test_gone` here.\n"
    hits = cs.find_stale(text, defined={"test_alive"}, files=set())
    assert [(h.lineno, h.name, h.line) for h in hits] == [(2, "test_gone", "See `test_alive` and `test_gone` here.")]


def test_a_cited_file_name_is_checked_against_the_test_files_not_the_defs():
    text = "`tests/test_present.py` and `tests/test_missing.py`\n"
    hits = cs.find_stale(text, defined=set(), files={"test_present.py"})
    assert [h.name for h in hits] == ["test_missing.py"]


def test_a_name_ending_in_an_underscore_is_a_family_and_is_alive_while_any_def_has_that_prefix():
    text = "3x `test_select_by_method_*` and 2x `test_gone_family_*`\n"
    hits = cs.find_stale(text, defined={"test_select_by_method_coordinates"}, files=set())
    assert [h.name for h in hits] == ["test_gone_family_"]


def test_a_rename_note_is_flagged_as_one_rather_than_as_a_dead_citation():
    text = "It was `test_old_name`, renamed to\n`test_new_name`.\nElsewhere `test_old_name` again.\n"
    hits = cs.find_stale(text, defined={"test_new_name"}, files=set())
    assert [(h.lineno, h.rename_note) for h in hits] == [(1, True), (3, False)]


def test_defined_names_and_files_come_from_the_tests_tree(tmp_path):
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "tests" / "test_a.py").write_text("def test_one():\n    pass\n")
    (tmp_path / "tests" / "sub" / "test_b.py").write_text("class T:\n    def test_two(self):\n        pass\n")
    defined, files = cs.tests_inventory(tmp_path)
    assert defined == {"test_one", "test_two"}
    assert files == {"test_a.py", "test_b.py"}


def _git(root, *args):
    # A pre-commit hook sets GIT_INDEX_FILE and the like; the toy repos must not see them,
    # or `git add` here would write into the repo being committed.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)


def test_the_tracked_list_ignores_git_variables_a_hook_leaves_in_the_environment(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "CLAUDE.md").write_text("## Key references\n\n- `ref.md` — a doc.\n")
    (tmp_path / "ref.md").write_text("x\n")
    (tmp_path / "other.md").write_text("bad `test_dead`\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "CLAUDE.md", "ref.md", "other.md")
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "somewhere-else-index"))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "not-a-repo"))
    assert "other.md:1:" in cs.report(tmp_path)


def test_the_report_keeps_the_key_references_apart_from_other_tracked_docs(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_alive():\n    pass\n")
    (tmp_path / "CLAUDE.md").write_text("## Key references\n\n- `ref.md` — a doc.\n")
    (tmp_path / "ref.md").write_text("ok `test_alive`\nbad `test_dead_in_ref`\n")
    (tmp_path / "other.md").write_text("bad `test_dead_in_other`\n")
    (tmp_path / "untracked.md").write_text("bad `test_dead_untracked`\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "CLAUDE.md", "ref.md", "other.md", "tests")
    out = cs.report(tmp_path)
    key_part, other_part = out.split("Other tracked documents")
    assert "ref.md:2:" in key_part and "bad `test_dead_in_ref`" in key_part
    assert "test_dead_in_other" not in key_part
    assert "other.md:1:" in other_part and "bad `test_dead_in_other`" in other_part
    assert "test_dead_untracked" not in out
    assert "test_dead_in_ref" not in other_part


def test_a_key_reference_document_that_is_not_on_disk_is_listed_as_missing_not_skipped_silently(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "CLAUDE.md").write_text("## Key references\n\n- `gitignored.md` — a doc.\n")
    _git(tmp_path, "init", "-q")
    assert "not present: gitignored.md" in cs.report(tmp_path)
