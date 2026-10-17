"""scripts/coverage_by_module.py: the before/after coverage check the test cleanup relies on."""
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import TOY_ENV, realdata_test, toy_git

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("coverage_by_module", REPO / "scripts" / "coverage_by_module.py")
cbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cbm)


def _result(modules, tests=None, args=("tests/x.py",), exit_code=0, outcomes=None):
    extra = {} if outcomes is None else {"outcomes": outcomes}
    return {**extra, "commit": "0" * 40, "dirty": False, "pytest_args": list(args), "pytest_exit": exit_code,
            "modules": {rel: {"statements": 10, "covered": lines} for rel, lines in modules.items()},
            "tests": tests or {}}


def test_in_scope_keeps_the_cb_source_and_leaves_out_nerve_only_files():
    assert cbm.in_scope("src/ImageLynx/batch_outputs.py")
    assert cbm.in_scope("examples/cb_h1_batch.py")
    assert cbm.in_scope("examples/preprocessing/preprocess_cb.py")
    assert not cbm.in_scope("src/ImageLynx/haemodynamics/pericyte_mask.py")
    assert not cbm.in_scope("examples/resistance_network_pipeline.py")
    assert not cbm.in_scope("examples/OLD/cb_old.py")


def test_compare_fails_on_a_dropped_module_and_names_the_lost_lines():
    before = _result({"src/a.py": [1, 2, 3, 4, 7], "src/b.py": [1]})
    after = _result({"src/a.py": [1, 2], "src/b.py": [1, 2]})
    code, report = cbm.compare(before, after)
    assert code == 1
    assert "DROP src/a.py: 5 -> 2; lines no longer run: 3-4, 7" in report
    assert "gain src/b.py: 1 -> 2" in report


def test_compare_passes_when_no_module_drops():
    code, report = cbm.compare(_result({"src/a.py": [1]}), _result({"src/a.py": [1, 2]}))
    assert code == 0
    assert report[-1] == "no module lost covered lines"


def test_compare_refuses_runs_with_different_pytest_args():
    code, report = cbm.compare(_result({}, args=["tests/x.py"]), _result({}, args=["tests/y.py"]))
    assert code == 3
    assert "pytest args differ" in report[-1]


def test_compare_refuses_a_run_whose_tests_failed():
    code, report = cbm.compare(_result({}), _result({}, exit_code=1))
    assert code == 3
    assert "after run's pytest exited 1" in report[-1]


def test_sole_lists_only_the_lines_no_other_test_ran():
    result = _result({}, tests={"t::one": {"src/a.py": [1, 2, 3], "src/b.py": [5]},
                                "t::two": {"src/a.py": [2, 3]}})
    assert cbm.sole(result, "t::one") == {"src/a.py": [1], "src/b.py": [5]}
    assert cbm.sole(result, "t::two") == {}
    with pytest.raises(KeyError):
        cbm.sole(result, "t::missing")


def test_run_writes_per_module_and_per_test_lines_and_leaves_no_data_file(tmp_path):
    pytest.importorskip("pytest_cov")
    test_id = "tests/test_cb_settings.py::test_pipeline_and_h2_share_the_pressure_boundaries"
    out = tmp_path / "run.json"
    # --allow-dirty: the hook runs this with the commit's changes staged.
    assert cbm.main(["run", "--allow-dirty", str(out), "-q", "-p", "no:cacheprovider", test_id]) == 0
    result = json.loads(out.read_text())
    assert result["pytest_args"] == ["-q", "-p", "no:cacheprovider", test_id]
    # Every in-scope file is listed, imported or not.
    assert set(result["modules"]) == {str(p.relative_to(REPO)) for p in cbm.in_scope_files()}
    assert result["outcomes"] == {test_id: "passed"}
    assert result["modules"]["examples/carotid_image_to_model.py"]["covered"]
    assert result["tests"][test_id]["examples/carotid_image_to_model.py"]
    assert not (REPO / ".coverage").exists()


def _write_runs(tmp_path, before_modules, after_modules):
    paths = []
    for name, modules in (("before.json", before_modules), ("after.json", after_modules)):
        path = tmp_path / name
        path.write_text(json.dumps(_result(modules)))
        paths.append(str(path))
    return paths


def test_table_lists_only_the_modules_whose_covered_count_changed(tmp_path, capsys):
    before, after = _write_runs(
        tmp_path,
        {"src/a.py": [1, 2, 3], "src/b.py": [1], "src/c.py": [1, 2]},
        {"src/a.py": [1, 2], "src/b.py": [1], "src/c.py": [1, 2, 3, 4]},
    )
    assert cbm.main(["table", before, after]) == 0
    assert capsys.readouterr().out.splitlines()[3:] == [
        "| Module | Statements | Covered before | Covered after | Change |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| src/a.py | 10 | 3 | 2 | -1 |",
        "| src/c.py | 10 | 2 | 4 | +2 |",
    ]


def test_table_with_modules_lists_exactly_those_rows_even_when_unchanged(tmp_path, capsys):
    before, after = _write_runs(
        tmp_path,
        {"src/a.py": [1, 2, 3], "src/b.py": [1], "src/c.py": [1, 2]},
        {"src/a.py": [1, 2], "src/b.py": [1], "src/c.py": [1, 2, 3, 4]},
    )
    assert cbm.main(["table", before, after, "--modules", "src/b.py", "src/a.py"]) == 0
    assert capsys.readouterr().out.splitlines()[5:] == [
        "| src/b.py | 10 | 1 | 1 | 0 |",
        "| src/a.py | 10 | 3 | 2 | -1 |",
    ]


def test_table_with_an_unknown_module_fails_and_prints_no_table(tmp_path, capsys):
    before, after = _write_runs(tmp_path, {"src/a.py": [1]}, {"src/a.py": [1, 2]})
    assert cbm.main(["table", before, after, "--modules", "src/a.py", "src/typo.py"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "src/typo.py" in captured.err


def test_table_fails_rather_than_invent_zeros_for_a_module_in_only_one_run(tmp_path, capsys):
    before, after = _write_runs(tmp_path, {"src/a.py": [1]}, {"src/a.py": [1], "src/new.py": [1, 2]})
    for extra in ([], ["--modules", "src/new.py"]):
        assert cbm.main(["table", before, after, *extra]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "src/new.py" in captured.err


_BEFORE_OUTCOMES = {"t::a": "passed", "t::b": "passed", "t::gone": "passed", "t::s": "skipped"}
_AFTER_OUTCOMES = {"t::a": "passed", "t::b": "skipped", "t::new": "passed"}


def test_compare_prints_counts_and_the_test_ids_only_in_one_run():
    before = _result({"src/a.py": [1]}, outcomes=_BEFORE_OUTCOMES)
    after = _result({"src/a.py": [1]}, outcomes=_AFTER_OUTCOMES)
    code, report = cbm.compare(before, after)
    assert code == 0
    assert "before: 3 passed, 1 skipped" in report
    assert "after: 2 passed, 1 skipped" in report
    assert report[report.index("only in BEFORE (2):") + 1:][:2] == ["  t::gone", "  t::s"]
    assert report[report.index("only in AFTER (1):") + 1] == "  t::new"


def test_compare_exit_codes_are_unchanged_by_the_test_counts():
    before = _result({"src/a.py": [1, 2]}, outcomes=_BEFORE_OUTCOMES)
    after = _result({"src/a.py": [1]}, outcomes=_AFTER_OUTCOMES)
    assert cbm.compare(before, after)[0] == 1
    assert cbm.compare(before, _result({}, exit_code=1, outcomes={}))[0] == 3


def test_compare_says_so_when_a_run_recorded_no_outcomes():
    code, report = cbm.compare(_result({"src/a.py": [1]}), _result({"src/a.py": [1]}, outcomes=_AFTER_OUTCOMES))
    assert code == 0
    assert any(l.startswith("before: test counts not recorded") for l in report)
    assert not any(l.startswith("only in") for l in report)


def test_table_leads_with_the_count_lines(tmp_path, capsys):
    paths = []
    for name, outcomes in (("b.json", _BEFORE_OUTCOMES), ("a.json", _AFTER_OUTCOMES)):
        (tmp_path / name).write_text(json.dumps(_result({"src/a.py": [1]}, outcomes=outcomes)))
        paths.append(str(tmp_path / name))
    assert cbm.main(["table", *paths]) == 0
    assert capsys.readouterr().out.splitlines()[:3] == [
        "before: 3 passed, 1 skipped  ", "after: 2 passed, 1 skipped", ""]


def _write_test_file(path, names):
    path.write_text("".join(f"def {n}():\n    pass\n\n\n" for n in names))
    return path


def test_run_files_from_runs_only_the_listed_test_files(tmp_path):
    pytest.importorskip("pytest_cov")
    listed = _write_test_file(tmp_path / "test_listed.py", ["test_one", "test_two"])
    _write_test_file(tmp_path / "test_unlisted.py", ["test_three"])
    files = tmp_path / "files.txt"
    files.write_text(f"# header comment\n\n{listed}\n")
    out = tmp_path / "run.json"
    assert cbm.main(["run", "--allow-dirty", "--files-from", str(files), str(out), "-q", "-p",
                     "no:cacheprovider"]) == 0
    result = json.loads(out.read_text())
    # pytest names a file outside the repo relative to its own rootdir, so compare by file name.
    assert sorted(t.rsplit("/", 1)[-1] for t in result["outcomes"]) == [
        "test_listed.py::test_one", "test_listed.py::test_two"]
    assert result["pytest_args"] == ["-q", "-p", "no:cacheprovider", str(listed)]
    assert result["files_from"] == str(files)


@pytest.mark.parametrize("listing", ["tests/test_no_such_file.py\n", "# only a comment\n\n", None],
                         ids=["missing file", "empty list", "no list"])
def test_run_files_from_refuses_a_bad_list_and_runs_nothing(tmp_path, capsys, listing):
    files = tmp_path / "files.txt"
    if listing is not None:
        files.write_text(listing)
    out = tmp_path / "run.json"
    assert cbm.main(["run", "--allow-dirty", "--files-from", str(files), str(out)]) == 2
    assert not out.exists()
    assert str(files) in capsys.readouterr().err


# --- run --at REV: a detached worktree, the data linked in, the checkout left alone ------------

SCRIPTS = ("coverage_by_module.py", "_cbm_outcomes.py", "_worktree.py")
CALC = "def add(a, b):\n    return a + b\n"


@pytest.fixture
def toy(tmp_path):
    """A small repo with the scripts, one in-scope module, a realdata test and gitignored data."""
    pytest.importorskip("pytest_cov")
    repo = tmp_path / "toy"
    for d in ("scripts", "src/ImageLynx", "tests", "examples/images", "examples/outputs"):
        (repo / d).mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(REPO / "scripts" / name, repo / "scripts" / name)
    (repo / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n'
                                         'markers = ["realdata: reads gitignored data"]\n')
    (repo / ".gitignore").write_text("__pycache__/\nexamples/images/sum.txt\nexamples/outputs/\n")
    (repo / "src" / "ImageLynx" / "__init__.py").write_text("")
    (repo / "src" / "ImageLynx" / "calc.py").write_text(CALC)
    (repo / "examples" / "images" / "README").write_text("tracked, so the folder is in every worktree\n")
    (repo / "tests" / "test_calc.py").write_text(
        "from ImageLynx.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (repo / "tests" / "test_data.py").write_text(realdata_test("ImageLynx"))
    toy_git(repo, "init", "-q")
    toy_git(repo, "add", ".")
    toy_git(repo, "commit", "-q", "-m", "toy")
    (repo / "examples" / "images" / "sum.txt").write_text("5\n")
    (repo / "examples" / "outputs" / "old.txt").write_text("an earlier run's output\n")
    return repo


def _cbm(repo, *args):
    """Run the toy's copy of the script; its temp folders go in a fresh TMPDIR that must end empty."""
    tmp = repo.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "coverage_by_module.py"), *args],
                          cwd=repo, env={**TOY_ENV, "TMPDIR": str(tmp)}, capture_output=True, text=True,
                          timeout=120)
    assert list(tmp.iterdir()) == [], "the script left temp files behind"
    return proc


def _checkout_state(repo):
    """Every file outside .git (bytes and mtime), git's status with ignored files, the worktrees."""
    files = {p.relative_to(repo).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts}
    return (files, toy_git(repo, "status", "--porcelain", "--ignored"), toy_git(repo, "worktree", "list"),
            toy_git(repo, "rev-parse", "HEAD"))


def _commit(repo, rel, text, message):
    (repo / rel).write_text(text)
    toy_git(repo, "add", rel)
    toy_git(repo, "commit", "-q", "-m", message)


def test_run_at_measures_that_commit_with_the_data_linked_and_never_touches_the_checkout(toy):
    first = toy_git(toy, "rev-parse", "HEAD").strip()
    _commit(toy, "src/ImageLynx/calc.py", CALC + "\n\ndef sub(a, b):\n    return a - b\n", "sub")
    # A dirty checkout, which run --at neither refuses nor measures.
    (toy / "src" / "ImageLynx" / "calc.py").write_text("X = 1\n" + CALC)
    (toy / "tests" / "test_extra.py").write_text("def test_extra():\n    pass\n")
    before = _checkout_state(toy)
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--at", first, str(out), "-q", "-p", "no:cacheprovider")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _checkout_state(toy) == before
    result = json.loads(out.read_text())
    assert (result["commit"], result["at"], result["dirty"]) == (first, first, False)
    assert result["outcomes"] == {"tests/test_calc.py::test_add": "passed",
                                  "tests/test_data.py::test_add_matches_the_data": "passed"}
    assert result["modules"]["src/ImageLynx/calc.py"] == {"statements": 2, "covered": [1, 2]}
    assert (result["realdata_skipped"], result["data_changed"], result["changed_during_run"]) == ([], [], False)


def test_run_at_fails_when_a_realdata_test_skips_and_compare_refuses_the_run(toy):
    (toy / "examples" / "images" / "sum.txt").unlink()
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--at", "HEAD", str(out), "-q", "-p", "no:cacheprovider")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "realdata test(s) skipped" in proc.stderr
    assert "tests/test_data.py::test_add_matches_the_data" in proc.stderr
    result = json.loads(out.read_text())
    assert result["realdata_skipped"] == ["tests/test_data.py::test_add_matches_the_data"]
    code, lines = cbm.compare(result, result)
    assert (code, lines[-2]) == (3, "cannot compare: the before run is no baseline")
    assert lines[-1].startswith("1 realdata test(s) skipped")


def test_run_at_fails_when_a_test_imports_this_checkouts_source_not_the_worktrees(toy):
    _commit(toy, "tests/test_calc.py", textwrap.dedent(f'''\
        import sys

        sys.path.insert(0, {str(toy / "src")!r})  # the checkout's src, ahead of the worktree's
        sys.modules.pop("ImageLynx", None)
        sys.modules.pop("ImageLynx.calc", None)
        from ImageLynx.calc import add


        def test_add():
            assert add(2, 3) == 5
    '''), "a test that imports the checkout")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--at", "HEAD", str(out), "-q", "-p", "no:cacheprovider")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "imported this checkout's " in proc.stderr
    assert "src/ImageLynx/calc.py, not the worktree's" in proc.stderr


def test_run_at_names_data_a_test_wrote_through_the_links(toy):
    _commit(toy, "tests/test_calc.py", textwrap.dedent('''\
        from pathlib import Path


        def test_writes():
            Path("examples/outputs/new.txt").write_text("written by a test")
    '''), "a test that writes data")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--at", "HEAD", str(out), "-q", "-p", "no:cacheprovider")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "changed the linked data in the main checkout: " in proc.stderr
    assert "examples/outputs/new.txt" in json.loads(out.read_text())["data_changed"]
    assert (toy / "examples" / "outputs" / "new.txt").exists()  # caught, not prevented


def test_run_refuses_a_dirty_tree_unless_allowed_and_help_says_how(toy, capsys):
    (toy / "src" / "ImageLynx" / "calc.py").write_text(CALC + "\nX = 1\n")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", str(out), "-q", "-p", "no:cacheprovider", "tests/test_calc.py")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--allow-dirty" in proc.stderr
    assert not out.exists()
    proc = _cbm(toy, "run", "--allow-dirty", str(out), "-q", "-p", "no:cacheprovider",
                "tests/test_calc.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text())["dirty"] is True
    with pytest.raises(SystemExit):
        cbm.main(["run", "--help"])
    assert "--allow-dirty" in capsys.readouterr().out


def test_run_fails_when_the_tree_changes_during_the_run(toy):
    _commit(toy, "tests/test_calc.py", textwrap.dedent('''\
        from pathlib import Path


        def test_edits_a_tracked_file():
            with open(Path(__file__).resolve().parents[1] / "src" / "ImageLynx" / "calc.py", "a") as f:
                f.write("# edited mid-run\\n")
    '''), "a test that edits the tree")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", str(out), "-q", "-p", "no:cacheprovider", "tests/test_calc.py")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "the tracked files changed during the run" in proc.stderr
    result = json.loads(out.read_text())
    assert (result["changed_during_run"], result["dirty"]) == (True, True)
    code, lines = cbm.compare(result, result)
    assert lines[-2:] == ["cannot compare: the before run is no baseline",
                          "the tracked files changed during the run (edited while it ran, or by a test)"]
    assert code == 3


# --- the tree a run measured: a commit, a tree id, a dirty flag -------------------------------


def test_a_clean_run_records_the_commit_and_the_tree_of_that_commit(toy):
    (toy / "scratch.txt").write_text("untracked, so not part of the measured tree\n")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", str(out), "-q", "-p", "no:cacheprovider", "tests/test_calc.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(out.read_text())
    assert result["commit"] == toy_git(toy, "rev-parse", "HEAD").strip()
    assert result["tree"] == toy_git(toy, "rev-parse", "HEAD^{tree}").strip()
    assert result["dirty"] is False


def test_a_dirty_run_records_the_tree_it_measured_not_the_commits(toy):
    (toy / "src" / "ImageLynx" / "calc.py").write_text(CALC + "\nX = 1\n")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--allow-dirty", str(out), "-q", "-p", "no:cacheprovider", "tests/test_calc.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(out.read_text())
    head_tree = toy_git(toy, "rev-parse", "HEAD^{tree}").strip()
    assert (result["commit"], result["dirty"]) == (toy_git(toy, "rev-parse", "HEAD").strip(), True)
    assert result["tree"] != head_tree
    # Committing the same edit gives a commit whose tree is the one the run recorded.
    toy_git(toy, "commit", "-q", "-am", "the edit")
    assert result["tree"] == toy_git(toy, "rev-parse", "HEAD^{tree}").strip()


def test_run_at_records_the_tree_of_that_commit_whatever_the_checkout_holds(toy):
    first = toy_git(toy, "rev-parse", "HEAD").strip()
    _commit(toy, "src/ImageLynx/calc.py", CALC + "\nX = 1\n", "later")
    (toy / "src" / "ImageLynx" / "calc.py").write_text("dirty\n")
    out = toy.parent / "run.json"
    proc = _cbm(toy, "run", "--at", first, str(out), "-q", "-p", "no:cacheprovider", "tests/test_calc.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text())["tree"] == toy_git(toy, "rev-parse", f"{first}^{{tree}}").strip()


def _commit_with_tree(toy, text, message):
    """(commit, tree) of a new commit in `toy`."""
    _commit(toy, "src/ImageLynx/calc.py", text, message)
    return toy_git(toy, "rev-parse", "HEAD").strip(), toy_git(toy, "rev-parse", "HEAD^{tree}").strip()


def test_compare_prints_both_runs_commits_and_trees_and_warns_when_a_tree_is_not_its_commits(toy):
    commit_a, tree_a = _commit_with_tree(toy, CALC + "\nA = 1\n", "a")
    commit_b, tree_b = _commit_with_tree(toy, CALC + "\nB = 1\n", "b")
    before = {**_result({"src/a.py": [1]}), "commit": commit_a, "tree": tree_a}
    clean = {**_result({"src/a.py": [1]}), "commit": commit_b, "tree": tree_b}
    code, report = cbm.compare(before, clean, repo=toy)
    assert code == 0
    assert f"before: {commit_a[:7]}, tree {tree_a[:7]}" in report
    assert f"after:  {commit_b[:7]}, tree {tree_b[:7]}" in report
    assert not any("WARNING" in line for line in report)

    # The after run says it measured commit b but holds commit a's tree.
    dirty = {**clean, "tree": tree_a, "dirty": True}
    code, report = cbm.compare(before, dirty, repo=toy)
    assert code == 0
    warning = [line for line in report if line.startswith("WARNING")]
    assert len(warning) == 1 and "after" in warning[0] and commit_b[:7] in warning[0]


def test_compare_says_so_when_a_run_recorded_no_tree_or_its_commit_is_unknown(toy):
    commit, tree = _commit_with_tree(toy, CALC + "\nA = 1\n", "a")
    old = _result({"src/a.py": [1]})  # saved before trees were recorded
    unknown = {**_result({"src/a.py": [1]}), "tree": tree}  # commit 000...0 is not in the repo
    known = {**_result({"src/a.py": [1]}), "commit": commit, "tree": tree}
    _, report = cbm.compare(old, known, repo=toy)
    assert "before: 0000000, tree not recorded" in report
    _, report = cbm.compare(known, unknown, repo=toy)
    assert any(line.startswith("WARNING") and "0000000" in line and "not in this repository" in line
               for line in report)


def test_check_says_whether_a_runs_tree_matches_a_commit(toy):
    commit_a, tree_a = _commit_with_tree(toy, CALC + "\nA = 1\n", "a")
    commit_b, _ = _commit_with_tree(toy, CALC + "\nB = 1\n", "b")
    run = toy.parent / "run.json"
    run.write_text(json.dumps({**_result({}), "commit": commit_a, "tree": tree_a}))
    proc = _cbm(toy, "check", str(run), commit_a)
    assert (proc.returncode, proc.stdout.strip()) == (0, f"tree {tree_a[:7]} is the tree of {commit_a[:7]}")
    proc = _cbm(toy, "check", str(run), commit_b)
    assert proc.returncode == 1 and f"is not the tree of {commit_b[:7]}" in proc.stdout
    assert _cbm(toy, "check", str(run), "no-such-rev").returncode == 2
    run.write_text(json.dumps(_result({})))
    proc = _cbm(toy, "check", str(run), commit_a)
    assert proc.returncode == 3 and "no tree recorded" in proc.stderr
