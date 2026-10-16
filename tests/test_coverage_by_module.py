"""scripts/coverage_by_module.py: the before/after coverage check the test cleanup relies on."""
import importlib.util
import json
from pathlib import Path

import pytest

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
    assert cbm.main(["run", str(out), "-q", "-p", "no:cacheprovider", test_id]) == 0
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
    assert cbm.main(["run", "--files-from", str(files), str(out), "-q", "-p", "no:cacheprovider"]) == 0
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
    assert cbm.main(["run", "--files-from", str(files), str(out)]) == 2
    assert not out.exists()
    assert str(files) in capsys.readouterr().err
