"""Running a batch: one run, one ``.haemorun`` and one output folder per image
(`gui/batch_run.py`)."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import networkx as nx
import pytest

from haemolynx.gui import batch_run
from haemolynx.gui.batch_run import (
    CANCELLED,
    DONE,
    FAILED,
    NOT_RUN,
    SKIPPED,
    haemorun_path,
    main,
    run_batch,
)
from haemolynx.gui.run_state import RunCancelled
from haemolynx.io import BatchFolders, match_batch
from haemolynx.pipeline import default_schema

FIXTURE = Path(__file__).resolve().parent / "data" / "seven_vessel_noisy_3d.tif"


@pytest.fixture(scope="module")
def schema():
    return default_schema()


def _values(schema, tmp_path: Path) -> dict:
    values = schema.defaults()
    values.update(
        {
            "vtk_output_prefix": tmp_path / "default_out" / "run",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
        }
    )
    return values


def _batch(tmp_path: Path, *stems: str):
    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in stems:
        (folder / f"{stem}.tif").write_bytes(b"")
    return BatchFolders(folder), match_batch(BatchFolders(folder))


def _summary_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fake_run(calls, fail=(), cancel=()):
    def run_image(values, schema, *, progress=None, check_cancelled=None, haemorun=None):
        stem = Path(values["input_path"]).stem
        calls.append((stem, values))
        if stem in cancel:
            raise RunCancelled()
        if stem in fail:
            raise ValueError(f"{stem} is broken")
        haemorun.parent.mkdir(parents=True, exist_ok=True)
        haemorun.write_bytes(b"run")
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        return G

    return run_image


def test_each_image_runs_with_its_own_files_and_folder(tmp_path, schema, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    folders, matches = _batch(tmp_path, "A_seg", "B_seg")
    out = tmp_path / "out"
    summary = run_batch(_values(schema, tmp_path), schema, matches, out, folders=folders)
    assert [stem for stem, _values in calls] == ["A_seg", "B_seg"]
    assert calls[0][1]["vtk_output_prefix"] == out / "A_seg" / "run"
    assert calls[1][1]["input_path"] == folders.segmented / "B_seg.tif"
    assert [result.status for result in summary.results] == [DONE, DONE]
    assert summary.results[0].haemorun == out / "A_seg" / "A_seg.haemorun"
    rows = _summary_rows(out / "batch_summary.csv")
    assert [(row["stem"], row["status"], row["nodes"], row["vessels"]) for row in rows] == [
        ("A_seg", DONE, "2", "1"),
        ("B_seg", DONE, "2", "1"),
    ]
    settings_file = (out / "batch_settings.yaml").read_text(encoding="utf-8")
    assert f"# batch_segmented: {folders.segmented.as_posix()}" in settings_file


def test_a_failed_image_is_recorded_and_the_batch_goes_on(tmp_path, schema, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls, fail={"A_seg"}))
    _folders, matches = _batch(tmp_path, "A_seg", "B_seg")
    summary = run_batch(_values(schema, tmp_path), schema, matches, tmp_path / "out")
    assert [result.status for result in summary.results] == [FAILED, DONE]
    assert summary.results[0].error == "ValueError: A_seg is broken"
    assert "1 failed" in summary.message()


def test_a_cancel_stops_the_batch(tmp_path, schema, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls, cancel={"B_seg"}))
    _folders, matches = _batch(tmp_path, "A_seg", "B_seg", "C_seg")
    summary = run_batch(_values(schema, tmp_path), schema, matches, tmp_path / "out")
    assert [result.status for result in summary.results] == [DONE, CANCELLED, NOT_RUN]
    rows = _summary_rows(summary.csv_path)
    assert [row["status"] for row in rows] == [DONE, CANCELLED, NOT_RUN]


def test_images_already_done_are_skipped_unless_asked(tmp_path, schema, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    _folders, matches = _batch(tmp_path, "A_seg", "B_seg")
    out = tmp_path / "out"
    done = haemorun_path(out, matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"earlier")
    summary = run_batch(_values(schema, tmp_path), schema, matches, out)
    assert [result.status for result in summary.results] == [SKIPPED, DONE]
    assert [stem for stem, _values in calls] == ["B_seg"]
    run_batch(_values(schema, tmp_path), schema, matches, out, skip_done=False)
    assert [stem for stem, _values in calls] == ["B_seg", "A_seg", "B_seg"]


def test_on_item_hears_each_image_start_and_end(tmp_path, schema, monkeypatch):
    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    _folders, matches = _batch(tmp_path, "A_seg")
    heard = []
    run_batch(
        _values(schema, tmp_path),
        schema,
        matches,
        tmp_path / "out",
        on_item=lambda index, total, result: heard.append((index, total, result.status)),
    )
    assert heard == [(0, 1, "running"), (0, 1, DONE)]


def test_the_command_line_checks_every_image_before_running(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml

    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in ("A_seg", "B_seg"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)

    assert main(["--config", str(config), "--segmented", str(folder), "--dry-run"]) == 0
    assert "All 2 image(s) passed their checks." in capsys.readouterr().out
    assert calls == []

    out = tmp_path / "out"
    assert main(["--config", str(config), "--segmented", str(folder), "--output", str(out), "--only", "B_seg"]) == 0
    assert [stem for stem, _values in calls] == ["B_seg"]
    assert (out / "B_seg" / "B_seg.haemorun").is_file()


def test_the_command_line_refuses_a_batch_whose_checks_fail(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml

    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    folder = tmp_path / "seg"
    folder.mkdir()
    shutil.copy(FIXTURE, folder / "A_seg.tif")
    values = _values(schema, tmp_path)
    values.update({"inlet_node_selection_method": "node_ids", "inlet_node_ids": [1]})
    config = write_settings_yaml(tmp_path / "config.yaml", values, schema)
    assert main(["--config", str(config), "--segmented", str(folder)]) == 1
    assert "Nothing was run" in capsys.readouterr().out
    assert calls == []


@pytest.mark.slow
@pytest.mark.integration
def test_a_real_batch_writes_a_haemorun_and_outputs_per_image(tmp_path, schema):
    """Two copies of the seven-vessel fixture, run for real: each gets its own
    folder with its outputs, and a .haemorun that reads back with a checkpoint
    for each stage and the image's own input path."""
    from haemolynx.gui.run_snapshot import read_run_snapshot

    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in ("A_seg", "B_seg"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    folders = BatchFolders(folder)
    matches = match_batch(folders)
    out = tmp_path / "out"
    summary = run_batch(_values(schema, tmp_path), schema, matches, out, folders=folders)
    assert [result.status for result in summary.results] == [DONE, DONE], [
        result.error for result in summary.results
    ]
    for result in summary.results:
        stem = result.match.stem
        snapshot = read_run_snapshot(out / stem / f"{stem}.haemorun")
        assert Path(snapshot.settings["input_path"]) == folder / f"{stem}.tif"
        assert {"build_network", "assign_boundaries", "solve", "export_results"} <= set(snapshot.stages)
        assert (out / stem / f"{stem}_run_manifest.json").is_file()
        assert list((out / stem).glob("*.vt*")), "no VTK output"
        assert result.nodes and result.vessels
    assert not (tmp_path / "default_out").exists()


def test_a_summary_file_held_open_never_stops_the_batch(tmp_path, schema, monkeypatch):
    """Windows refuses to replace a file another program holds: the summary
    write retries, and a write that still fails is logged, not raised."""
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    monkeypatch.setattr(batch_run, "_REPLACE_WAIT_S", 0.0)
    real_replace = Path.replace
    refusals = {"left": 3}

    def flaky_replace(self, target):
        if refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError("held open")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    _folders, matches = _batch(tmp_path, "A_seg")
    summary = run_batch(_values(schema, tmp_path), schema, matches, tmp_path / "out")
    assert [result.status for result in summary.results] == [DONE]
    assert _summary_rows(summary.csv_path)[0]["status"] == DONE

    def refused(self, target):
        raise PermissionError("held open")

    monkeypatch.setattr(Path, "replace", refused)
    summary = run_batch(
        _values(schema, tmp_path), schema, matches, tmp_path / "out2", skip_done=False
    )
    assert [result.status for result in summary.results] == [DONE]


def _fake_run_with_statistics(lengths):
    """A run that writes the statistics CSV an image's export would."""
    from haemolynx.statistics import export_statistics_to_csv

    def run_image(values, schema, *, progress=None, check_cancelled=None, haemorun=None):
        stem = Path(values["input_path"]).stem
        haemorun.parent.mkdir(parents=True, exist_ok=True)
        haemorun.write_bytes(b"run")
        export_statistics_to_csv(
            {"Total Edge Length (microns)": lengths[stem]},
            haemorun.parent / f"{stem}_statistics.csv",
        )
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        return G

    return run_image


def test_a_batch_with_two_conditions_compares_their_statistics(tmp_path, schema, monkeypatch):
    from haemolynx.statistics.batch_comparison import BatchConditions

    lengths = {"WT_1": 100.0, "WT_2": 110.0, "KO_1": 200.0, "KO_2": 210.0, "HET_1": 150.0}
    monkeypatch.setattr(batch_run, "run_image", _fake_run_with_statistics(lengths))
    folders, matches = _batch(tmp_path, *lengths)
    out = tmp_path / "out"
    conditions = BatchConditions.from_text("wt", "ko")
    summary = run_batch(
        _values(schema, tmp_path), schema, matches, out, folders=folders, conditions=conditions
    )
    rows = {row["stem"]: row["condition"] for row in _summary_rows(summary.csv_path)}
    assert rows == {"HET_1": "", "KO_1": "ko", "KO_2": "ko", "WT_1": "wt", "WT_2": "wt"}
    assert summary.statistics.images == 5
    comparison = _summary_rows(out / "batch_condition_comparison.csv")
    assert [(row["Metric"], row["Median wt"], row["Median ko"]) for row in comparison] == [
        ("Total Edge Length", "105", "205")
    ]
    assert (out / "batch_statistics_summary.csv").is_file()
    assert "ko against wt" in summary.message()
    settings_file = (out / "batch_settings.yaml").read_text(encoding="utf-8")
    assert "# batch_condition_a: wt" in settings_file
    assert "# batch_condition_b: ko" in settings_file


def test_the_statistics_cover_images_skipped_as_done_and_leave_out_failures(tmp_path, schema, monkeypatch):
    from haemolynx.statistics import export_statistics_to_csv

    lengths = {"A_seg": 1.0, "B_seg": 2.0, "C_seg": 3.0}
    fake = _fake_run_with_statistics(lengths)

    def run_image(values, schema, **kwargs):
        if Path(values["input_path"]).stem == "C_seg":
            raise ValueError("broken")
        return fake(values, schema, **kwargs)

    monkeypatch.setattr(batch_run, "run_image", run_image)
    _folders, matches = _batch(tmp_path, *lengths)
    out = tmp_path / "out"
    done = haemorun_path(out, matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"earlier")
    export_statistics_to_csv({"Total Edge Length (microns)": 1.0}, done.parent / "A_seg_statistics.csv")
    # C_seg's folder holds statistics from some earlier run: a failed image is left out.
    export_statistics_to_csv({"Total Edge Length (microns)": 9.0}, out / "C_seg" / "C_seg_statistics.csv")
    summary = run_batch(_values(schema, tmp_path), schema, matches, out)
    assert [result.status for result in summary.results] == [SKIPPED, DONE, FAILED]
    header = next(csv.reader((out / "batch_statistics_by_image.csv").open(encoding="utf-8")))
    assert header[3:] == ["A_seg", "B_seg"]


def test_statistics_that_cannot_be_written_never_fail_the_batch(tmp_path, schema, monkeypatch):
    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(batch_run, "write_batch_statistics", broken)
    _folders, matches = _batch(tmp_path, "A_seg")
    summary = run_batch(_values(schema, tmp_path), schema, matches, tmp_path / "out")
    assert [result.status for result in summary.results] == [DONE]
    assert "Could not write the batch statistics: disk full" in summary.message()


def test_conditions_show_in_the_table_and_two_alike_fail_the_checks(tmp_path, schema):
    from dataclasses import replace

    from haemolynx.gui.batch_run import BatchSetup, check_setup, table_rows
    from haemolynx.statistics.batch_comparison import BatchConditions

    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in ("WT_1", "KO_1", "HET_1"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    folders = BatchFolders(folder)
    setup = BatchSetup(folders=folders, output_root=tmp_path / "out", matches=match_batch(folders))
    assert [row[5] for row in table_rows(setup)] == ["", "", ""]
    setup = replace(setup, conditions=BatchConditions.from_text("WT", "KO"))
    assert [(row[1], row[5]) for row in table_rows(setup)] == [
        ("HET_1.tif", "neither"), ("KO_1.tif", "KO"), ("WT_1.tif", "WT")
    ]
    assert setup.condition_counts() == "1 in WT, 1 in KO, 1 in neither"
    values = _values(schema, tmp_path)
    checked, lines = check_setup(values, schema, setup)
    assert set(checked.checked.values()) == {""}
    assert lines[0] == "Conditions: 1 in WT, 1 in KO, 1 in neither."
    alike = replace(setup, conditions=BatchConditions.from_text("WT", "wt"))
    checked, lines = check_setup(values, schema, alike)
    assert all("Both conditions" in error for error in checked.checked.values())


def test_the_command_line_compares_conditions_without_running(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml
    from haemolynx.statistics import export_statistics_to_csv

    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    folder = tmp_path / "seg"
    folder.mkdir()
    out = tmp_path / "out"
    for stem, length in (("WT_1", 1.0), ("WT_2", 2.0), ("KO_1", 5.0), ("KO_2", 6.0)):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
        export_statistics_to_csv(
            {"Total Edge Length (microns)": length}, out / stem / f"{stem}_statistics.csv"
        )
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    argv = ["--config", str(config), "--segmented", str(folder), "--output", str(out)]

    assert main(argv + ["--condition-a", "WT", "--condition-b", "KO", "--compare-only"]) == 0
    printed = capsys.readouterr().out
    assert "Conditions: 2 in WT, 2 in KO." in printed
    assert "KO against WT: 1 metrics tested" in printed
    assert calls == []
    assert (out / "batch_condition_comparison.csv").is_file()

    assert main(argv + ["--condition-a", "WT", "--condition-b", "wt", "--compare-only"]) == 1
    assert "Both conditions" in capsys.readouterr().out

    assert main(argv + ["--condition-a", "WT", "--condition-b", "KO", "--dry-run"]) == 0
    assert "[ok] KO_1 (KO)" in capsys.readouterr().out
