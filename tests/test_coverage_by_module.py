"""scripts/coverage_by_module.py: the before/after coverage check the test cleanup relies on."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("coverage_by_module", REPO / "scripts" / "coverage_by_module.py")
cbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cbm)


def _result(modules, tests=None, args=("tests/x.py",), exit_code=0):
    return {"commit": "0" * 40, "dirty": False, "pytest_args": list(args), "pytest_exit": exit_code,
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
    assert capsys.readouterr().out.splitlines() == [
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
    assert capsys.readouterr().out.splitlines()[2:] == [
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
