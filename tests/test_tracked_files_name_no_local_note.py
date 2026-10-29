"""No tracked file points at a local note by its file name, its item numbers or a scratch path.

A local note is kept on disk beside the code but never tracked, so a clone does not have it and
a pointer to one leads nowhere. The rules are in ``scripts/local_note_pointers.py``; this runs
them on the repo, and on toy repos that show each rule fires and reports ``path:line: text``.
The toy text is built from pieces, so this file does not match the rules itself.
"""
from pathlib import Path

from conftest import toy_git
from local_note_pointers import pointers

REPO = Path(__file__).resolve().parents[1]

#: Lines let through for now, as (path, text). This one is a value the driver writes into its
#: JSON output, so rewording it changes a written file; that is done with the other printed and
#: written text, and the entry goes then. An entry that matches no line fails the test.
NOT_YET_REWORDED = {
    ("examples/cb_h1_th_metrics.py",
     '"vessel_source": "cb_h1_batch network (' + "open" + ' item 40)",'),
}


def _toy(tmp_path, files):
    """A git repo in `tmp_path` tracking `files` ({path: text or bytes}), with nothing committed."""
    toy_git(tmp_path, "init", "-q")
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    toy_git(tmp_path, "add", "-A")
    return tmp_path


def test_a_line_citing_an_open_item_is_reported(tmp_path):
    repo = _toy(tmp_path, {"src/calc.py": "x = 1\n# Fixed under " + "Open" + " Item 12.\n"})
    assert pointers(repo) == ["src/calc.py:2: # Fixed under " + "Open" + " Item 12."]


def test_a_line_with_a_scratch_path_is_reported(tmp_path):
    repo = _toy(tmp_path, {"scripts/run.sh": "# Spec: " + "." + "scratch/feature/spec\necho\n"})
    assert pointers(repo) == ["scripts/run.sh:1: # Spec: " + "." + "scratch/feature/spec"]


def test_a_line_naming_an_untracked_md_file_is_reported(tmp_path):
    plan, guide = "plan" + ".md", "guide" + ".md"
    repo = _toy(tmp_path, {
        "README.md": "A toy.\n",
        f"docs/{guide}": "How to run it.\n",
        "src/calc.py": (f"# Usage is in README.md and docs/{guide}, or just {guide}.\n"
                        f"# The design is in {plan}.\n"),
    })
    (repo / plan).write_text("On disk, never tracked.\n")
    assert pointers(repo) == [f"src/calc.py:2: # The design is in {plan}."]


def test_what_the_rules_leave_alone(tmp_path):
    open_item = "open" + " item 3"
    repo = _toy(tmp_path, {
        # .gitignore names the local notes on purpose.
        ".gitignore": "/plan" + ".md\n",
        # Fixture data is not prose.
        "tests/data/table.csv": f"note,{open_item}\n",
        "logo.png": b"\x89PNG\0" + open_item.encode(),
        # A glob is not a file name, and tests write these names themselves.
        "scripts/hook.sh": "case $f in *.md) ;; esac\n",
        "tests/test_report.py": "NAMES = ['notes.md', 'model_results.md']\n",
        # A note that is still tracked, other than the three docs that stay.
        "docs/plan" + ".md": f"See {open_item}.\n",
    })
    assert pointers(repo) == []



def test_a_tracked_file_deleted_on_disk_is_passed_over(tmp_path):
    repo = _toy(tmp_path, {"src/gone.py": "# Fixed under " + "open" + " item 3.\n"})
    (repo / "src" / "gone.py").unlink()
    assert pointers(repo) == []

def test_the_three_kept_docs_are_checked(tmp_path):
    open_item = "open" + " item 3"
    repo = _toy(tmp_path, {name: f"See {open_item}.\n"
                           for name in ("README.md", "GLOSSARY.md", "CODING_STANDARDS.md")})
    assert pointers(repo) == [f"{name}:1: See {open_item}."
                              for name in ("CODING_STANDARDS.md", "GLOSSARY.md", "README.md")]


def _is(line, path, text):
    """Whether reported `line` is `text` in file `path`."""
    return line.startswith(path + ":") and line.endswith(": " + text)


def test_no_tracked_file_points_at_a_local_note():
    found = pointers(REPO)
    let_through = {(path, text) for path, text in NOT_YET_REWORDED
                   if any(_is(line, path, text) for line in found)}
    assert let_through == NOT_YET_REWORDED, (
        f"These no longer match a line, so drop them: {NOT_YET_REWORDED - let_through}")
    found = [line for line in found
             if not any(_is(line, path, text) for path, text in NOT_YET_REWORDED)]
    assert not found, "Tracked lines that point at a local note; say the reason in words:\n" + (
        "\n".join(found))
