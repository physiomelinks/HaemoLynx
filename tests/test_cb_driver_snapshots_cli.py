"""``examples/cb_driver_snapshots.py``: argument handling and exit codes (C5)."""
import json

import pytest

import cb_driver_snapshots as cli
from ImageLynx import driver_snapshots as ds


def _snapshot_folder(path, stdout="hello\n", exit_code=0):
    (path / "d").mkdir(parents=True)
    (path / "d" / "stdout.txt").write_text(stdout)
    (path / "manifest.json").write_text(json.dumps({
        "commit": "abc", "dirty": False, "python": "3", "packages": {},
        "drivers": [{"name": "d", "command": ["d.py"], "exit_code": exit_code}]}))
    return path


def test_compare_exits_0_on_a_match_and_1_on_a_difference(tmp_path, capsys):
    old = _snapshot_folder(tmp_path / "old")
    same = _snapshot_folder(tmp_path / "same")
    other = _snapshot_folder(tmp_path / "other", stdout="changed\n")
    assert cli.main(["compare", str(old), str(same)]) == 0
    assert "all match" in capsys.readouterr().out
    assert cli.main(["compare", str(old), str(other)]) == 1
    assert "d: DIFFERENT" in capsys.readouterr().out


def test_compare_exits_3_with_the_reason_when_a_driver_failed(tmp_path, capsys):
    old = _snapshot_folder(tmp_path / "old")
    new = _snapshot_folder(tmp_path / "new", exit_code=1)
    assert cli.main(["compare", str(old), str(new)]) == 3
    assert "driver d exited non-zero" in capsys.readouterr().err


def test_compare_exits_2_for_a_folder_that_does_not_exist(tmp_path):
    old = _snapshot_folder(tmp_path / "old")
    with pytest.raises(SystemExit) as stop:
        cli.main(["compare", str(old), str(tmp_path / "missing")])
    assert stop.value.code == 2


def test_snapshot_exits_2_for_an_existing_folder_and_for_an_unknown_driver(tmp_path):
    (tmp_path / "there").mkdir()
    with pytest.raises(SystemExit) as stop:
        cli.main(["snapshot", str(tmp_path / "there")])
    assert stop.value.code == 2
    with pytest.raises(SystemExit) as stop:
        cli.main(["snapshot", str(tmp_path / "new"), "--driver", "nope"])
    assert stop.value.code == 2


def test_snapshot_exits_3_when_a_driver_fails_and_0_otherwise(tmp_path, monkeypatch, capsys):
    outcomes = iter([0, 1])

    def _take(path, drivers, **kwargs):
        code = next(outcomes)
        return {"commit": "abcdef0123", "dirty": False,
                "drivers": [{"name": drivers[0].name, "command": [], "exit_code": code}]}

    monkeypatch.setattr(cli, "take_snapshot", _take)
    assert cli.main(["snapshot", str(tmp_path / "a"), "--driver", "cb_h2_vtk"]) == 0
    assert "running: cb_h2_vtk" in capsys.readouterr().out
    assert cli.main(["snapshot", str(tmp_path / "b"), "--driver", "cb_h1_vtk"]) == 3
    assert "cb_h1_vtk exited non-zero" in capsys.readouterr().err


def test_the_default_snapshot_runs_the_fast_drivers_and_slow_adds_two(tmp_path, monkeypatch,
                                                                      capsys):
    ran = []

    def _take(path, drivers, **kwargs):
        ran.append([d.name for d in drivers])
        return {"commit": "abcdef0123", "dirty": False, "drivers": []}

    monkeypatch.setattr(cli, "take_snapshot", _take)
    cli.main(["snapshot", str(tmp_path / "a")])
    cli.main(["snapshot", str(tmp_path / "b"), "--slow"])
    assert len(ran[1]) - len(ran[0]) == 2
    assert ran[0] == [d.name for d in ds.DRIVERS if not d.slow]
