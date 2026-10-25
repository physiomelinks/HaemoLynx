"""The full tier's datasets file, crops and collectors, and the report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from release_review import full, manifests, report, review_doc, scorecard, trends
from release_review.ladder import FAIL, PASS, measured


def test_the_datasets_file_resolves_paths_and_fills_in_crops(tmp_path):
    (tmp_path / "config.yaml").write_text("run:\n  input_path: images/e14.tif\n")
    (tmp_path / "datasets.yaml").write_text(yaml.safe_dump({
        "e14p5": {"config": "config.yaml", "settings": {"statistics": True, "raw_image_path": "raw.tif"},
                  "needs_gib": 8, "shape_zyx": [287, 512, 512]},
        "e14p5_eighth": {"of": "e14p5", "crop_fraction": 0.125},
    }))
    datasets = full.load_datasets(tmp_path / "datasets.yaml")
    whole, eighth = datasets["e14p5"], datasets["e14p5_eighth"]
    assert whole["config"] == str((tmp_path / "config.yaml").resolve())
    assert whole["settings"]["raw_image_path"] == str((tmp_path / "raw.tif").resolve())
    assert whole["voxels"] == 287 * 512 * 512
    assert eighth["settings"]["crop_size_xyz"] == [256, 256, 144]
    assert eighth["settings"]["statistics"] is True and eighth["needs_gib"] == 8
    assert eighth["voxels"] == 256 * 256 * 144
    (missing,) = full.missing_inputs(whole)
    assert Path(missing).resolve() == (tmp_path / "images" / "e14.tif").resolve()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "e14.tif").write_bytes(b"x")
    assert full.missing_inputs(whole) == []
    assert full.load_datasets(tmp_path / "absent.yaml") == {}


def test_a_crop_keeps_the_fraction_of_the_volume():
    for fraction in (0.125, 0.25, 0.5):
        assert full.cropped_voxels((287, 512, 512), fraction) / (287 * 512 * 512) == pytest.approx(fraction, rel=0.02)


def _run(stages=2, status="finished"):
    return manifests.parse({
        "status": status, "wall_seconds": 600.0, "peak_rss_bytes": 5 * 2**30,
        "platform": {"system": "Windows", "total_ram_bytes": 32 * 2**30},
        "stages": [{"stage": f"s{i}", "status": "finished", "wall_seconds": 10.0, "peak_rss_bytes": 2**30}
                   for i in range(stages)],
    })


def test_the_whole_e14_run_decides_speed_2a_and_memory_2a():
    assert full.speed_2a(_run()).status == PASS
    assert full.speed_2a(_run(status="failed")).status == FAIL
    assert full.memory_2a(_run()).evidence.startswith("E14.5 finished, peak 5.0 GiB on a 32.0 GiB Windows")
    # Not run, or no room to run it: the row decides; no room is not a fail.
    assert full.speed_2a(None) is None and full.memory_2a(None) is None


def test_every_reference_dataset_needs_its_table():
    runs = {name: _run() for name in full.REFERENCE}
    assert full.per_stage_check("speed.6a", runs, {}).status == PASS
    del runs["mason_sm"]
    result = full.per_stage_check("memory.6a", runs, {"mason_sm": "input not on this machine: E:/x.tif"})
    assert result.status == FAIL and "Mason SM: input not on this machine" in result.evidence


def test_a_table_recorded_at_an_earlier_commit_counts():
    rows = manifests.benchmark_rows(_run(), dataset="mason_sm", commit="abc12345", date="2026-10-20")
    assert full.measured_before(rows, "mason_sm") == "abc12345"
    assert full.measured_before(rows, "brain") is None
    runs = {name: _run() for name in full.REFERENCE if name != "mason_sm"}
    result = full.per_stage_check("speed.6a", runs, {}, rows)
    assert result.status == PASS and "Mason SM (at abc12345)" in result.evidence
    failed = manifests.benchmark_rows(_run(status="failed"), dataset="mason_sm", commit="def", date="2026-10-21")
    assert full.measured_before(failed, "mason_sm") is None


def test_the_solved_share_must_be_explained_on_every_dataset():
    graph = {"solved_fraction": 0.5, "length_um": 100.0, "length_um_by_flow_state": {"solved": 50.0, "blind_end": 50.0}}
    results = {name: {"graph": graph} for name in full.REFERENCE}
    assert full.flow_6c(results, {}).status == PASS
    results["nerve"] = {"graph": {"solved_fraction": 0.0, "length_um_by_flow_state": {"not solved": 9.0}}}
    result = full.flow_6c(results, {})
    assert result.status == FAIL and "Nerve: no solve" in result.evidence


def test_the_lumen_report_counts_on_a_real_stack_only():
    report = {"text": "Lumen artefacts: 17 pair(s)...", "counts": {"duplicate_pair_count": 17}}
    assert full.graph_6a({"e14p5": {"lumen_artefacts": report}}).status == PASS
    assert full.graph_6a({"seven_vessel": {"lumen_artefacts": report}}) is None
    missing = full.graph_6a({"e14p5": {"status": "failed"}})
    assert missing.status == FAIL and "no report for E14.5" in missing.evidence


def test_a_slowdown_check_needs_an_earlier_commit():
    assert full.speed_8b([], None) is None
    slower = trends.Slowdown("e14p5", "build_network", "abc", 100.0, 130.0)
    result = full.speed_8b([slower], "abc")
    assert result.status == PASS and "e14p5 build_network 100 -> 130 s" in result.evidence


def test_scaling_needs_three_sizes_and_e14_within_sixteen_gib():
    rows = [{"commit": "a", "dataset": name, "stage": stage, "status": "finished", "voxels": voxels,
             "wall_seconds": voxels / 1e6, "peak_rss_mib": peak}
            for name, voxels in (("e14p5", 8e6), ("e14p5_half", 4e6), ("e14p5_quarter", 2e6))
            for stage, peak in (("graph", voxels / 1e4), ("total", 5000.0 * voxels / 8e6))]
    datasets = {"e14p5": {}, "e14p5_half": {"of": "e14p5"}, "e14p5_quarter": {"of": "e14p5"}}
    assert full.scaling_check("speed.10a", rows, datasets).status == PASS
    memory = full.scaling_check("memory.10a", rows, datasets)
    assert memory.status == PASS and "against 16 GiB" in memory.evidence
    assert full.scaling_check("speed.10a", rows[:4], datasets).status == FAIL  # two sizes
    assert full.scaling_check("speed.10a", [], datasets) is None
    # A stage reached on the whole stack alone (each crop stopped before it)
    # is not measured against volume.
    solve = {"commit": "a", "dataset": "e14p5", "stage": "solve", "status": "finished", "voxels": 8e6,
             "wall_seconds": 3.0, "peak_rss_mib": 800.0}
    stopped = full.scaling_check("speed.10a", rows + [solve], datasets)
    assert stopped.status == FAIL and "fewer than 3 sizes: solve" in stopped.evidence
    # Rows of another commit are not mixed in.
    other = [{**row, "commit": "b", "wall_seconds": 99.0} for row in rows]
    assert full.scaling_check("speed.10a", rows + other, datasets, commit="a").status == PASS


DOCUMENT = review_doc.parse("""\
## Scorecard

| # | Aspect | Score | What holds it |
|---|---|---|---|
| 1 | [Graph accuracy](#1-graph-accuracy--graph) | **4** | x |

## The ladders

### 1. Graph accuracy — `graph`

| ID | By | Passes when | Now | Evidence or gap |
|---|---|---|---|---|
| graph.2a | A | A graph builds. | ✓ | `outputs/x/REPORT.md` |
""")


def test_the_report_says_what_moved_and_where_it_disagrees(tmp_path):
    card = scorecard.build(DOCUMENT, {"graph.2a": measured("graph.2a", False, evidence="it did not")}, {},
                           commit="new", date="2026-10-26", tiers=["quick"])
    previous = {"commit": "old", "date": "2026-10-25", "scores": {"graph": 2},
                "results": {"graph.2a": {"status": "pass"}}}
    moved = report.changes(previous, card)
    assert moved["scores"][0] == ("graph", 2, 0) and ("graph.2a", "pass", "fail") in moved["checks"]
    assert moved["since"] == "2026-10-25 at old"
    text = report.markdown(card, moved=moved, sections=[("Extra", "detail")])
    assert "# Release review scorecard, 2026-10-26 at `new`" in text
    assert "- graph: 2 → 0" in text and "`graph.2a`: the row says pass, here fail (measured)" in text
    assert "| 1 | Graph accuracy | **0** | 4 | graph.2a |" in text
    assert "## Extra" in text
    assert report.changes(None, card)["since"] is None


def test_the_latest_earlier_scorecard_is_found(tmp_path):
    for name in ("2026-10-20_aaa", "2026-10-24_bbb", "2026-10-25_ccc"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "scorecard.json").write_text(json.dumps({"commit": name}))
    assert report.latest_previous(tmp_path, tmp_path / "2026-10-25_ccc")["commit"] == "2026-10-24_bbb"
    assert report.latest_previous(tmp_path / "nothing", tmp_path) is None
