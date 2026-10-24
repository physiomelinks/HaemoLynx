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


def test_the_command_line_logs_each_line_with_its_time(tmp_path, schema, monkeypatch, capsys):
    """A batch runs unattended; how long each step took is read off its log."""
    import logging
    import re

    from haemolynx.pipeline.provenance import write_settings_yaml

    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    folder = tmp_path / "seg"
    folder.mkdir()
    shutil.copy(FIXTURE, folder / "A_seg.tif")
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    root = logging.getLogger()
    original_handlers, original_level = list(root.handlers), root.level
    root.handlers.clear()  # basicConfig does nothing while pytest's are there
    try:
        assert main(["--config", str(config), "--segmented", str(folder), "--dry-run"]) == 0
        capsys.readouterr()
        logging.getLogger("haemolynx.pipeline").info("Step 11/13 recover_uncovered_mask_vessels")
        line = capsys.readouterr().out
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \[INFO\] Step 11/13 recover_uncovered_mask_vessels\s*", line
    ), line


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


def test_the_command_line_pairs_conditions_by_subject(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml
    from haemolynx.statistics import export_statistics_to_csv

    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    folder = tmp_path / "seg"
    folder.mkdir()
    out = tmp_path / "out"
    for stem, length in (
        ("28_3_d_contra_FITC", 1.0), ("28_3_d_ipsi_FITC", 2.0),
        ("29_1_f_contra_FITC", 3.0), ("29_1_f_ipsi_FITC", 5.0),
        ("29_1_b_contra_FITC", 4.0),
    ):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
        export_statistics_to_csv(
            {"Total Edge Length (microns)": length}, out / stem / f"{stem}_statistics.csv"
        )
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    argv = [
        "--config", str(config), "--segmented", str(folder), "--output", str(out),
        "--condition-a", "contra", "--condition-b", "ipsi", "--paired", "--compare-only",
    ]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "2 pair(s) by subject (28_3_d, 29_1_f), 1 image(s) with no partner" in printed
    assert "left out with no partner: 29_1_b_contra_FITC" in printed
    assert "smallest possible p 0.5" in printed
    pairs = _summary_rows(out / "batch_condition_pairs.csv")
    assert [row["Subject"] for row in pairs] == ["28_3_d", "29_1_f", "29_1_b"]
    comparison = _summary_rows(out / "batch_condition_comparison.csv")
    assert comparison[0]["N pairs"] == "2"

    assert main(argv[:-4] + ["--paired", "--compare-only"]) == 1  # one condition
    assert "needs both conditions" in capsys.readouterr().out


def test_the_table_names_each_images_subject_or_why_it_has_no_partner(tmp_path):
    from dataclasses import replace

    from haemolynx.gui.batch_run import BatchSetup, table_rows, write_batch_settings
    from haemolynx.statistics.batch_comparison import BatchConditions

    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in ("s1_contra", "s1_ipsi", "s2_contra", "s3_other"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    folders = BatchFolders(folder)
    setup = BatchSetup(
        folders=folders,
        output_root=tmp_path / "out",
        matches=match_batch(folders),
        conditions=BatchConditions.from_text("contra", "ipsi", paired=True),
    )
    assert {row[1]: row[5] for row in table_rows(setup)} == {
        "s1_contra.tif": "contra, subject s1",
        "s1_ipsi.tif": "ipsi, subject s1",
        "s2_contra.tif": "contra: unpaired, no ipsi image of subject s2",
        "s3_other.tif": "neither",
    }
    # Unticking an image unpairs its partner.
    unticked = replace(setup, unticked=frozenset({"s1_ipsi"}))
    cells = {row[1]: row[5] for row in table_rows(unticked)}
    assert cells["s1_contra.tif"] == "contra: unpaired, no ipsi image of subject s1"
    assert cells["s1_ipsi.tif"] == "ipsi"
    out = tmp_path / "out"
    out.mkdir()
    schema = default_schema()
    path = write_batch_settings(schema.defaults(), schema, folders, out, setup.conditions)
    text = path.read_text(encoding="utf-8")
    assert "# batch_comparison: paired" in text


def test_the_command_line_pairs_animals_by_the_median_of_their_images(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml
    from haemolynx.statistics import export_statistics_to_csv

    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    folder = tmp_path / "seg"
    folder.mkdir()
    out = tmp_path / "out"
    for stem, length in (
        ("28_3_d_contra_FITC", 1.0), ("28_3_d_ipsi_FITC", 2.0),
        ("29_1_b_contra_FITC", 3.0), ("29_1_f_contra_FITC", 5.0), ("29_1_f_ipsi_FITC", 6.0),
        ("29_2_f_contra_FITC", 7.0), ("29_2_c_ipsi_FITC", 8.0), ("29_2_d_ipsi_FITC", 9.0),
    ):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
        export_statistics_to_csv(
            {"Total Edge Length (microns)": length}, out / stem / f"{stem}_statistics.csv"
        )
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    argv = [
        "--config", str(config), "--segmented", str(folder), "--output", str(out),
        "--condition-a", "contra", "--condition-b", "ipsi", "--paired",
        "--subject-pattern", r"^(\d+_\d+)", "--paired-aggregate", "median", "--compare-only",
    ]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "3 pair(s) by subject (28_3, 29_1, 29_2)" in printed
    assert "(median of each subject's images in a condition): 3 pair(s)" in printed
    assert "smallest possible p 0.25" in printed
    pairs = {row["Subject"]: row for row in _summary_rows(out / "batch_condition_pairs.csv")}
    assert pairs["29_1"]["contra images"] == "29_1_b_contra_FITC; 29_1_f_contra_FITC"
    comparison = _summary_rows(out / "batch_condition_comparison.csv")
    # Pairs (1, 2), (median 4, 6), (7, median 8.5): differences 1, 2, 1.5.
    assert float(comparison[0]["Median of paired differences (ipsi - contra)"]) == 1.5

    with pytest.raises(SystemExit):  # argparse knows the three ways to combine
        main(argv[:-3] + ["--paired-aggregate", "sum", "--compare-only"])


def _statistics_with_dead_ends(path: Path, length: float, degree: float, dead_ends: float) -> None:
    from haemolynx.statistics import export_statistics_to_csv

    export_statistics_to_csv(
        {
            "Total Edge Length (microns)": length,
            "Average Degree": degree,
            "Dead Ends": {"Dead Ends": dead_ends},
        },
        path,
    )


def test_the_command_line_leaves_a_section_out_of_the_comparison(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml

    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    folder = tmp_path / "seg"
    folder.mkdir()
    out = tmp_path / "out"
    # Length and dead ends part WT from KO (p = 0.1 at 3 against 3); degree has
    # one pair crossed (p = 0.2).
    for stem, length, degree in (
        ("WT_1", 1.0, 1.0), ("WT_2", 2.0, 2.0), ("WT_3", 3.0, 4.0),
        ("KO_1", 4.0, 3.0), ("KO_2", 5.0, 5.0), ("KO_3", 6.0, 6.0),
    ):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
        _statistics_with_dead_ends(out / stem / f"{stem}_statistics.csv", length, degree, length)
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    argv = [
        "--config", str(config), "--segmented", str(folder), "--output", str(out),
        "--condition-a", "WT", "--condition-b", "KO", "--compare-only",
    ]

    def q_values():
        return {
            row["Metric"]: float(row["Q (Benjamini-Hochberg)"])
            for row in _summary_rows(out / "batch_condition_comparison.csv")
        }

    assert main(argv) == 0
    assert "3 metrics tested" in capsys.readouterr().out
    # m = 3, p .1 .1 .2: q = .1 * 3 / 2 for length and dead ends.
    assert q_values() == pytest.approx({"Total Edge Length": 0.15, "Dead Ends": 0.15, "Average Degree": 0.2})

    assert main(argv + ["--exclude-section", "Dead Ends", "--exclude-section", "Loops"]) == 0
    printed = capsys.readouterr().out
    assert "left out: Dead Ends (1 metric)" in printed
    assert "no section named Loops" in printed
    assert "2 metrics tested" in printed
    # m = 2, p .1 .2: q = .2 for both.
    assert q_values() == pytest.approx({"Total Edge Length": 0.2, "Average Degree": 0.2})
    for name in ("batch_statistics_by_image.csv", "batch_statistics_summary.csv"):
        with (out / name).open(encoding="utf-8") as handle:
            assert not any(row[0] == "Dead Ends" for row in csv.reader(handle)), name
    assert calls == []


def test_a_batch_leaves_its_sections_out_and_records_them(tmp_path, schema, monkeypatch):
    from haemolynx.statistics.batch_comparison import BatchConditions

    def run_image(values, schema, *, progress=None, check_cancelled=None, haemorun=None):
        stem = Path(values["input_path"]).stem
        haemorun.parent.mkdir(parents=True, exist_ok=True)
        haemorun.write_bytes(b"run")
        number = float(stem[-1])
        _statistics_with_dead_ends(haemorun.parent / f"{stem}_statistics.csv", number, number, number)
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        return G

    monkeypatch.setattr(batch_run, "run_image", run_image)
    folders, matches = _batch(tmp_path, "WT_1", "WT_2", "KO_3", "KO_4")
    out = tmp_path / "out"
    summary = run_batch(
        _values(schema, tmp_path),
        schema,
        matches,
        out,
        folders=folders,
        conditions=BatchConditions.from_text("WT", "KO"),
        exclude_sections=("Dead Ends",),
    )
    assert summary.statistics.excluded == {"Dead Ends": 1}
    comparison = _summary_rows(out / "batch_condition_comparison.csv")
    assert [row["Section"] for row in comparison] == ["Core Statistics", "Core Statistics"]
    settings_file = (out / "batch_settings.yaml").read_text(encoding="utf-8")
    assert "# batch_exclude_sections: Dead Ends" in settings_file


# -- what the table says before a batch ----------------------------------------


def test_the_table_says_which_images_a_batch_would_skip(tmp_path, schema, monkeypatch):
    from dataclasses import replace

    from haemolynx.gui.batch_run import (
        DONE_BEFORE_RERUN,
        DONE_BEFORE_SKIPPED,
        READY,
        BatchSetup,
        stems_done_before,
        table_rows,
    )

    folders, matches = _batch(tmp_path, "A_seg", "B_seg")
    out = tmp_path / "out"
    done = haemorun_path(out, matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"earlier")
    assert stems_done_before(out, matches) == {"A_seg"}

    setup = BatchSetup(
        folders=folders,
        output_root=out,
        matches=tuple(matches),
        done_before=stems_done_before(out, matches),
    )
    assert [row[-1] for row in table_rows(setup)] == [DONE_BEFORE_SKIPPED, READY]
    assert [row[-1] for row in table_rows(replace(setup, skip_done=False))] == [
        DONE_BEFORE_RERUN, READY
    ]
    # This batch's own news comes first: a failed check, or what it came to.
    checked = replace(setup, checked={"A_seg": "", "B_seg": ""})
    assert [row[-1] for row in table_rows(checked)] == [DONE_BEFORE_SKIPPED, f"{READY} (checked)"]
    ran = replace(setup, checked={"A_seg": DONE, "B_seg": "FAILED to check"})
    assert [row[-1] for row in table_rows(ran)] == [DONE, "FAILED to check"]


def test_the_last_batchs_statuses_are_read_back_from_its_summary(tmp_path, schema, monkeypatch):
    from haemolynx.gui.batch_run import BatchSetup, read_summary_statuses, stems_done_before, table_rows

    monkeypatch.setattr(batch_run, "run_image", _fake_run([], fail={"A_seg"}, cancel={"B_seg"}))
    folders, matches = _batch(tmp_path, "A_seg", "B_seg", "C_seg")
    out = tmp_path / "out"
    run_batch(_values(schema, tmp_path), schema, matches, out)
    statuses = read_summary_statuses(out)
    assert statuses == {
        "A_seg": "failed: ValueError: A_seg is broken",
        "B_seg": CANCELLED,
        "C_seg": NOT_RUN,
    }
    setup = BatchSetup(
        folders=folders,
        output_root=out,
        matches=tuple(matches),
        done_before=stems_done_before(out, matches),
        last_statuses=statuses,
    )
    assert [row[-1] for row in table_rows(setup)] == [
        "last batch: failed: ValueError: A_seg is broken",
        "last batch: cancelled",
        "last batch: not run",
    ]
    assert read_summary_statuses(tmp_path / "nowhere") == {}


def test_an_image_the_last_batch_never_finished_says_so(tmp_path):
    from haemolynx.gui.batch_run import RUNNING, match_status

    _folders, (match,) = _batch(tmp_path, "A_seg")
    assert match_status(match, last=RUNNING) == "last batch: stopped while running"
    # A .haemorun there says more than the summary does.
    assert match_status(match, done_before=True, last=RUNNING).startswith("done before")


def test_a_result_says_its_error_with_its_status(tmp_path):
    from haemolynx.gui.batch_run import BatchResult

    _folders, (match,) = _batch(tmp_path, "A_seg")
    assert BatchResult(match, status=DONE).status_text() == DONE
    assert BatchResult(match, status=FAILED, error="ValueError: x").status_text() == "failed: ValueError: x"


# -- conditions with Statistics off ------------------------------------------------


def test_conditions_with_statistics_off_warn_but_do_not_stop_the_batch(tmp_path, schema):
    from dataclasses import replace

    from haemolynx.gui.batch_run import BatchSetup, check_setup, statistics_warning
    from haemolynx.statistics.batch_comparison import BatchConditions

    folders, matches = _batch(tmp_path, "WT_1", "KO_1")
    setup = BatchSetup(
        folders=folders,
        output_root=tmp_path / "out",
        matches=tuple(matches),
        conditions=BatchConditions.from_text("WT", "KO"),
    )
    values = _values(schema, tmp_path)
    assert values["statistics"] is False
    checked, lines = check_setup(values, schema, setup)
    assert any(line.startswith("    warning: Statistics is off") for line in lines)
    assert "Statistics is off" not in " ".join(checked.checked.values())

    _checked, lines = check_setup({**values, "statistics": True}, schema, setup)
    assert not any("Statistics is off" in line for line in lines)
    no_conditions = replace(setup, conditions=BatchConditions())
    _checked, lines = check_setup(values, schema, no_conditions)
    assert not any("Statistics is off" in line for line in lines)
    assert statistics_warning(values, None) == ""


def test_the_command_line_warns_of_conditions_with_statistics_off(tmp_path, schema, monkeypatch, capsys):
    from haemolynx.pipeline.provenance import write_settings_yaml

    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    folder = tmp_path / "seg"
    folder.mkdir()
    for stem in ("WT_1", "KO_1"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
    config = write_settings_yaml(tmp_path / "config.yaml", _values(schema, tmp_path), schema)
    argv = ["--config", str(config), "--segmented", str(folder), "--dry-run"]
    assert main(argv + ["--condition-a", "WT", "--condition-b", "KO"]) == 0
    assert "warning: Statistics is off" in capsys.readouterr().out
    assert main(argv) == 0
    assert "Statistics is off" not in capsys.readouterr().out


# -- each image's caches -------------------------------------------------------


def _caching_run(memmaps: Path, fail=()):
    """A run that leaves what a real one does: the input decompressed into
    the memmap folder, stage checkpoints and the thick-vessel region beside
    the outputs, and the outputs themselves."""
    from haemolynx.io import load_raw_volume_reusing_cache

    def run_image(values, schema, *, progress=None, check_cancelled=None, haemorun=None):
        stem = Path(values["input_path"]).stem
        image, *_rest = load_raw_volume_reusing_cache(
            str(values["input_path"]), input_format="tif", axis_order="zyx", memmap_directory=memmaps
        )
        del image, _rest
        folder = haemorun.parent
        folder.mkdir(parents=True, exist_ok=True)
        for name in (
            f"{stem}_checkpoint_build_network.pkl",
            f"{stem}_checkpoint_solve.pkl",
            f"{stem}_thick_vessel_region.json",
            f"{stem}_thick_vessel_region.npy",
            f"{stem}_skeleton.npy",
            f"{stem}_graph.pkl",
        ):
            (folder / name).write_bytes(b"x")
        if stem in fail:
            raise ValueError(f"{stem} is broken")
        haemorun.write_bytes(b"run")
        return nx.MultiGraph([(0, 1)])

    return run_image


def _caching_batch(tmp_path: Path, schema, **changes):
    import os
    import time

    folder = tmp_path / "seg"
    folder.mkdir()
    old = time.time_ns() - 60_000_000_000
    for stem in ("A_seg", "B_seg"):
        shutil.copy(FIXTURE, folder / f"{stem}.tif")
        os.utime(folder / f"{stem}.tif", ns=(old, old))
    memmaps = tmp_path / "memmaps"
    values = {
        **_values(schema, tmp_path),
        "use_memmap_loading": True,
        "memmap_directory": memmaps,
        **changes,
    }
    return values, memmaps, match_batch(BatchFolders(folder))


def test_each_images_cached_files_go_once_it_ends_and_its_outputs_stay(tmp_path, schema, monkeypatch):
    values, memmaps, matches = _caching_batch(tmp_path, schema)
    monkeypatch.setattr(batch_run, "run_image", _caching_run(memmaps, fail={"B_seg"}))
    out = tmp_path / "out"
    summary = run_batch(values, schema, matches, out)
    assert [result.status for result in summary.results] == [DONE, FAILED]
    assert list(memmaps.iterdir()) == []
    for stem in ("A_seg", "B_seg"):  # a failed image's too
        left = sorted(path.name for path in (out / stem).iterdir())
        assert f"{stem}_skeleton.npy" in left and f"{stem}_graph.pkl" in left
        assert not [name for name in left if "checkpoint" in name or "thick_vessel_region" in name]
    assert (out / "A_seg" / "A_seg.haemorun").is_file()


def test_caches_stay_when_asked(tmp_path, schema, monkeypatch):
    values, memmaps, matches = _caching_batch(tmp_path, schema)
    monkeypatch.setattr(batch_run, "run_image", _caching_run(memmaps))
    out = tmp_path / "out"
    run_batch(values, schema, matches, out, clear_caches=False)
    assert len([path for path in memmaps.iterdir() if path.suffix == ".memmap"]) == 2
    assert (out / "A_seg" / "A_seg_checkpoint_solve.pkl").is_file()
    assert (out / "B_seg" / "B_seg_thick_vessel_region.npy").is_file()


def test_clearing_an_images_caches_never_fails_the_batch(tmp_path, schema, monkeypatch):
    from haemolynx.gui.batch_run import clear_image_caches

    def broken(*_args, **_kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr("haemolynx.io.discard_cached_raw_volume", broken)
    values, _memmaps, (match, _other) = _caching_batch(tmp_path, schema)
    assert clear_image_caches({**values, "input_path": match.segmented}, schema) == ()


# -- following a batch ----------------------------------------------------------


def test_the_progress_line_counts_the_images_and_estimates_the_rest(tmp_path):
    from haemolynx.gui.batch_run import RUNNING, BatchResult, batch_progress_text

    _folders, matches = _batch(tmp_path, "A", "B", "C", "D", "E")
    results = [BatchResult(match) for match in matches]
    results[0].status = RUNNING
    assert batch_progress_text(results, elapsed_s=30) == (
        "Image 1 of 5: A -- 0 done, 0 failed, under a minute so far"
    )
    results[0].status, results[0].seconds = DONE, 1500
    results[1].status, results[1].seconds = FAILED, 20
    results[2].status = RUNNING
    # 25 min an image: two waiting, and 20 of the running one's 25 left.
    assert batch_progress_text(results, elapsed_s=1800, running_elapsed_s=300) == (
        "Image 3 of 5: C -- 1 done, 1 failed, 30 min so far, about 1 h 10 min left"
    )
    for result in results[2:]:
        result.status, result.seconds = DONE, 1500
    assert batch_progress_text(results, elapsed_s=6000).startswith("5 images -- 4 done, 1 failed, 1 h 40 min")


def test_a_batch_asked_to_stop_finishes_its_image_and_runs_no_more(tmp_path, schema, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "run_image", _fake_run(calls))
    _folders, matches = _batch(tmp_path, "A_seg", "B_seg", "C_seg")
    summary = run_batch(
        _values(schema, tmp_path), schema, matches, tmp_path / "out", stop_requested=lambda: bool(calls)
    )
    assert [result.status for result in summary.results] == [DONE, NOT_RUN, NOT_RUN]
    assert summary.stopped
    assert "Stopped after an image, as asked." in summary.message()
    assert "runs just the ones that failed or did not run" in summary.message()
    rows = _summary_rows(summary.csv_path)
    assert [row["status"] for row in rows] == [DONE, NOT_RUN, NOT_RUN]


def test_a_failure_keeps_its_traceback_and_the_summary_its_first_line(tmp_path, schema, monkeypatch):
    monkeypatch.setattr(batch_run, "run_image", _fake_run([], fail={"A_seg"}))
    _folders, matches = _batch(tmp_path, "A_seg")
    summary = run_batch(_values(schema, tmp_path), schema, matches, tmp_path / "out")
    (result,) = summary.results
    assert result.error == "ValueError: A_seg is broken"
    assert result.error_detail.startswith("Traceback") and "A_seg is broken" in result.error_detail
    (row,) = _summary_rows(summary.csv_path)
    assert row["error"] == "ValueError: A_seg is broken"


def test_time_ram_and_vessels_fill_the_table_from_this_batch_or_the_last(tmp_path, schema, monkeypatch):
    from haemolynx.gui.batch_run import (
        TABLE_COLUMNS,
        BatchSetup,
        figure_cells,
        read_summary_figures,
        table_rows,
    )

    monkeypatch.setattr(batch_run, "run_image", _fake_run([]))
    folders, matches = _batch(tmp_path, "A_seg")
    out = tmp_path / "out"
    run_batch(_values(schema, tmp_path), schema, matches, out)
    (time_s, ram, vessels) = read_summary_figures(out)["A_seg"]
    assert time_s == "under a minute" and vessels == "1"
    assert figure_cells(6600, 3 * 1024**3 + 200 * 1024**2, 12345) == ("1 h 50 min", "3.2 GB", "12,345")
    setup = BatchSetup(
        folders=folders, output_root=out, matches=tuple(matches), figures=read_summary_figures(out)
    )
    (row,) = table_rows(setup)
    assert row[TABLE_COLUMNS.index("Time")] == "under a minute"
    assert row[TABLE_COLUMNS.index("Vessels")] == "1"
    assert len(row) == len(TABLE_COLUMNS)


def test_an_images_peak_memory_is_its_own_stages_not_the_process(tmp_path):
    import json

    from haemolynx.gui.batch_run import _peak_rss

    folder = tmp_path / "A"
    folder.mkdir()
    manifest = {"peak_rss_bytes": 900, "stages": [{"peak_rss_bytes": 100}, {"peak_rss_bytes": 300}, {}]}
    (folder / "A_run_manifest.json").write_text(json.dumps(manifest))
    settings = {"vtk_output_prefix": folder / "run"}
    assert _peak_rss(settings, "A") == 300
    (folder / "A_run_manifest.json").write_text(json.dumps({"peak_rss_bytes": 900, "stages": []}))
    assert _peak_rss(settings, "A") == 900


def test_check_reads_again_which_images_are_done(tmp_path, schema):
    from haemolynx.gui.batch_run import DONE_BEFORE_SKIPPED, BatchSetup, check_setup, table_rows

    folders, matches = _batch(tmp_path, "A_seg")
    out = tmp_path / "out"
    setup = BatchSetup(folders=folders, output_root=out, matches=tuple(matches))
    done = haemorun_path(out, matches[0])
    done.parent.mkdir(parents=True)
    done.write_bytes(b"since")
    checked, _lines = check_setup(_values(schema, tmp_path), schema, setup)
    assert checked.done_before == {"A_seg"}
    assert table_rows(checked)[0][-1] == DONE_BEFORE_SKIPPED


# -- acting on the results --------------------------------------------------


def test_run_again_is_the_one_image_whether_or_not_it_was_done(tmp_path):
    from haemolynx.gui.batch_run import BatchSetup

    folders, matches = _batch(tmp_path, "A_seg", "B_seg", "C_seg")
    setup = BatchSetup(
        folders=folders, output_root=tmp_path / "out", matches=tuple(matches),
        unticked=frozenset({"B_seg"}), checked={"A_seg": "done"},
    )
    again = setup.only("B_seg")
    assert [match.stem for match in again.selected()] == ["B_seg"]
    assert again.skip_done is False and again.checked == {}


def test_the_comparison_shows_its_tested_rows_smallest_q_first(tmp_path):
    from haemolynx.gui.batch_run import comparison_table, comparison_tested
    from haemolynx.statistics.batch_comparison import COMPARISON_FILENAME

    header = [
        "Section", "Metric", "Unit", "N WT", "N KO", "Median WT", "Median KO",
        "Mean WT", "Mean KO", "Difference of medians (KO - WT)", "Mann-Whitney U",
        "P (Mann-Whitney U, two-sided)", "Q (Benjamini-Hochberg)", "Smallest possible P",
        "Welch t", "P (Welch t)", "Note",
    ]
    rows = [
        ["Core", "Length", "um", 3, 3, 10, 12, 10, 12, 2, 1, 0.2, 0.4, 0.1, 1, 0.3, ""],
        ["Core", "Nodes", "count", 3, 3, 5, 9, 5, 9, 4, 0, 0.05, 0.1, 0.1, 3, 0.01, ""],
        ["Core", "Flat", "count", 3, 3, 1, 1, 1, 1, 0, "", "", "", "", "", "", "one value everywhere"],
    ]
    with (tmp_path / COMPARISON_FILENAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    shown, cells = comparison_table(tmp_path)
    assert shown == [
        "Section", "Metric", "Unit", "N WT", "N KO", "Median WT", "Median KO",
        "Difference of medians (KO - WT)", "P (Mann-Whitney U, two-sided)",
        "Q (Benjamini-Hochberg)", "Note",
    ]
    assert [row[1] for row in cells] == ["Nodes", "Length", "Flat"]
    assert cells[0][shown.index("P (Mann-Whitney U, two-sided)")] == "0.05"
    assert comparison_tested(shown, cells) == 2
    assert comparison_table(tmp_path / "none") == ([], [])


# -- setting a batch up ----------------------------------------------------------


def test_a_batch_is_read_back_from_its_settings_file(tmp_path, schema):
    from haemolynx.gui.batch_run import SETTINGS_FILENAME, read_batch_settings, write_batch_settings
    from haemolynx.parsers import load_config
    from haemolynx.statistics.batch_comparison import BatchConditions

    folders = BatchFolders(tmp_path / "seg", large=tmp_path / "large")
    conditions = BatchConditions.from_text(
        "contra", "ipsi, ipsilateral", paired=True, subject_pattern=r"^(\d+_\d+)", aggregate="median"
    )
    values = {**_values(schema, tmp_path), "statistics": True}
    out = tmp_path / "out"
    out.mkdir()
    path = write_batch_settings(values, schema, folders, out, conditions, ("Dead Ends", "Loops"))
    record = read_batch_settings(path)
    assert record.folders == folders
    assert record.output_root == out
    assert record.conditions == conditions
    assert record.exclude_sections == ("Dead Ends", "Loops")
    assert load_config(path, schema)["statistics"] is True  # its body is the settings
    plain = write_batch_settings(values, schema, BatchFolders(tmp_path / "seg"), out)
    assert read_batch_settings(plain).conditions == BatchConditions()
    (tmp_path / SETTINGS_FILENAME).write_text("statistics: true\n")
    assert read_batch_settings(tmp_path / SETTINGS_FILENAME) is None
    assert read_batch_settings(tmp_path / "missing.yaml") is None


def test_a_greyed_mask_row_names_the_switches_that_would_use_it(schema):
    from haemolynx.gui.batch_run import mask_switch_hint
    from haemolynx.gui.tabs import assign_to_stages

    values = schema.defaults()
    tab = assign_to_stages(schema)["use_large_vessel_masks"]
    assert mask_switch_hint(schema, values, "large") == (
        f"not used: tick 'Automated vessel assignment' and 'Use large vessel masks' ({tab})"
    )
    on = {**values, "automated_vessel_assignment": True}
    assert "'Use small vessel masks for boundary assignment'" in mask_switch_hint(schema, on, "small")
    assert "Automated" not in mask_switch_hint(schema, on, "small")
    used = {**on, "use_large_vessel_masks": True}
    assert mask_switch_hint(schema, used, "large") == ""
    assert "ilastik" in mask_switch_hint(
        schema, {**used, "use_ilastik_large_vessel_segmentation": True}, "large"
    )


def test_the_settings_every_image_runs_with_that_differ_from_the_defaults(schema, tmp_path):
    from haemolynx.gui.batch_run import settings_changed

    values = schema.defaults()
    assert settings_changed(values, schema) == []
    changed = {**values, "statistics": True, "input_path": tmp_path / "one.tif"}
    assert settings_changed(changed, schema) == ["statistics"]
