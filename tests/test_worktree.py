"""scripts/_worktree.py: the throwaway worktree that coverage_by_module and mutation_check run in."""
from pathlib import Path

import _worktree
import pytest
from conftest import toy_git

from ImageLynx.specimens import BATCH_RUN_ROOT, LEGACY_WKY_A_CLASSIFIER

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    """A toy repo: one tracked README in examples/images, and gitignored data beside it."""
    repo = tmp_path / "toy"
    images = repo / "examples" / "images"
    images.mkdir(parents=True)
    (repo / ".gitignore").write_text("examples/images/*\n!examples/images/README\nexamples/outputs/\n")
    (images / "README").write_text("tracked\n")
    toy_git(repo, "init", "-q")
    toy_git(repo, "add", ".")
    toy_git(repo, "commit", "-q", "-m", "toy")
    (images / "README").write_text("this checkout's own edit, not the tracked text\n")
    (images / "sum.txt").write_text("5\n")
    (images / "sub").mkdir()
    (images / "sub" / "a.txt").write_text("a\n")
    outputs = repo / "examples" / "outputs"
    (outputs / "run1").mkdir(parents=True)
    (outputs / "run1" / "x.txt").write_text("x\n")
    (outputs / "latest").symlink_to("run1")
    return repo


def test_git_ignores_the_git_variables_a_hook_leaves_set(repo, monkeypatch):
    # A pre-commit hook runs with GIT_DIR and GIT_INDEX_FILE set; they would aim git at another repo.
    monkeypatch.setenv("GIT_DIR", str(repo.parent / "not-a-repo"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repo.parent / "somewhere-else-index"))
    assert _worktree.git(repo, "ls-files").split() == [".gitignore", "examples/images/README"]


def test_git_passes_the_variables_it_is_given(repo):
    ident = _worktree.git(repo, "var", "GIT_AUTHOR_IDENT",
                          env={"GIT_AUTHOR_NAME": "Given", "GIT_AUTHOR_EMAIL": "given@x"})
    assert ident.startswith("Given <given@x>")


def test_clean_env_drops_git_variables_and_the_prefixes_named(monkeypatch):
    monkeypatch.setenv("GIT_INDEX_FILE", "x")
    monkeypatch.setenv("COV_CORE_SOURCE", "x")
    monkeypatch.setenv("KEPT_VARIABLE", "x")
    env = _worktree.clean_env("COV_CORE_")
    assert ("GIT_INDEX_FILE" in env, "COV_CORE_SOURCE" in env, env["KEPT_VARIABLE"]) == (False, False, "x")


def test_a_tracked_file_keeps_its_tracked_version_and_its_folder_is_filled_entry_by_entry(repo):
    with _worktree.worktree(repo) as wt:
        images = wt.path / "examples" / "images"
        assert wt.linked == ["examples/images/sub", "examples/images/sum.txt", "examples/outputs"]
        assert images.is_dir() and not images.is_symlink()
        assert (images / "README").read_text() == "tracked\n"
        assert not (images / "README").is_symlink()
        assert (images / "sum.txt").resolve() == (repo / "examples" / "images" / "sum.txt").resolve()
        assert (images / "sub" / "a.txt").read_text() == "a\n"
        assert wt.rev == "HEAD"
    assert (repo / "examples" / "images" / "sum.txt").read_text() == "5\n"  # the links went, not the data


def test_data_changes_names_what_a_run_added_changed_or_removed_through_the_links(repo):
    with _worktree.worktree(repo) as wt:
        assert wt.data_changes() == []
        (wt.path / "examples" / "images" / "sum.txt").write_text("changed, longer\n")
        (wt.path / "examples" / "images" / "sub" / "a.txt").unlink()
        (wt.path / "examples" / "outputs" / "run1" / "new.txt").write_text("new\n")
        assert wt.data_changes() == ["examples/images/sub", "examples/images/sub/a.txt",
                                     "examples/images/sum.txt", "examples/outputs/run1",
                                     "examples/outputs/run1/new.txt"]


def test_a_symlinked_folder_in_the_data_is_one_entry_not_walked_into(repo):
    # outputs/latest -> run1: listed by the link itself, so run1's files are not listed twice,
    # and pointing the link elsewhere is a change.
    with _worktree.worktree(repo) as wt:
        outputs = wt.path / "examples" / "outputs"
        assert sorted(_worktree.snapshot(wt.path, wt.linked)) == [
            "examples/images/sub", "examples/images/sub/a.txt", "examples/images/sum.txt",
            "examples/outputs", "examples/outputs/latest", "examples/outputs/run1",
            "examples/outputs/run1/x.txt"]
        (outputs / "latest").unlink()
        (outputs / "latest").symlink_to("run1/x.txt")
        assert "examples/outputs/latest" in wt.data_changes()


def test_the_specimens_in_repo_data_lies_under_the_folders_linked_in():
    # DATA_FOLDERS is repo-relative so the toy repos above can use it; specimens.py owns the
    # absolute paths. This ties the two: a run folder moved out of them fails here.
    for path in (BATCH_RUN_ROOT, LEGACY_WKY_A_CLASSIFIER):
        rel = path.relative_to(REPO).as_posix()
        assert any(rel.startswith(folder + "/") for folder in _worktree.DATA_FOLDERS), rel
