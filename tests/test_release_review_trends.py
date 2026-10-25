"""Stage times and memory from run manifests, the benchmark rows over time,
a stage flagged slower (`speed.8b`) and growth with volume (`speed.10a`)."""
from __future__ import annotations

import json

import pytest

from release_review import manifests, trends

MANIFEST = {
    "status": "finished",
    "wall_seconds": 30.0,
    "peak_rss_bytes": 3 * 2**30,
    "haemolynx_version": "0.1.0",
    "git": None,
    "platform": {"system": "Windows", "total_ram_bytes": 32 * 2**30},
    "inputs": [{"path": "x.tif", "checksum": "sha256:ab"}],
    "settings": {"input_path": "x.tif"},
    "stages": [
        {"stage": "segment", "status": "finished", "wall_seconds": 2.0, "peak_rss_bytes": 2**30,
         "peak_rss_method": "process_peak", "resumed": False},
        {"stage": "skeletonise", "status": "finished", "wall_seconds": 20.0, "peak_rss_bytes": 3 * 2**30,
         "peak_rss_method": "process_peak", "resumed": False},
        {"stage": "build_network", "status": "failed", "wall_seconds": 8.0, "peak_rss_bytes": None,
         "peak_rss_method": None, "resumed": False},
    ],
}


def test_a_manifest_is_read_into_stage_records(tmp_path):
    path = tmp_path / "x_run_manifest.json"
    path.write_text(json.dumps(MANIFEST))
    run = manifests.read(path)
    assert run.finished and run.platform == "Windows" and run.machine_ram_bytes == 32 * 2**30
    assert [stage.stage for stage in run.stages] == ["segment", "skeletonise", "build_network"]
    assert run.stage("skeletonise").wall_seconds == 20.0
    assert manifests.find(tmp_path) == [path]


def test_benchmark_rows_skip_resumed_stages_and_add_the_whole_run():
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["stages"][0]["resumed"] = True
    rows = manifests.benchmark_rows(manifests.parse(manifest), dataset="e14p5", commit="abc", date="2026-10-25", voxels=10)
    assert [row["stage"] for row in rows] == ["skeletonise", "build_network", "total"]
    assert rows[0]["peak_rss_mib"] == 3072.0 and rows[-1]["wall_seconds"] == 30.0
    assert set(rows[0]) == set(manifests.BENCHMARK_COLUMNS)


def test_the_stage_table_says_how_each_peak_was_read():
    table = manifests.stage_table(manifests.parse(MANIFEST))
    assert "| skeletonise | finished | 20.0 | 3072.0 | process_peak |" in table
    assert "| build_network | failed | 8.0 | - | - |" in table


def test_a_dataset_counts_only_with_every_stage_timed_and_its_peak():
    complete = manifests.parse({**MANIFEST, "stages": MANIFEST["stages"][:2]})
    gappy = manifests.parse(MANIFEST)
    done, missing = manifests.per_dataset({"a": complete, "b": gappy, "c": None}, ["a", "b", "c", "d"])
    assert (done, missing) == (["a"], ["b", "c", "d"])


def test_the_solved_share_and_why_the_rest_carries_no_flow():
    edges = [
        {"length": 10.0, "flow_state": "solved"},
        {"length": 5.0, "flow_state": "blind_end"},
        {"length": 5.0, "flow_state": "no_boundary"},
    ]
    share = manifests.solved_share(edges)
    assert share["solved_fraction"] == 0.5 and share["unsolved_vessels"] == 2
    assert share["unsolved_um_by_state"] == {"blind_end": 5.0, "no_boundary": 5.0}
    assert manifests.solved_share([{"length": 3.0}])["unsolved_um_by_state"] == {"not solved": 3.0}


def _row(commit, stage, seconds, dataset="e14p5", status="finished", voxels=None, peak=None):
    return {"commit": commit, "dataset": dataset, "stage": stage, "wall_seconds": seconds,
            "status": status, "voxels": voxels, "peak_rss_mib": peak}


def test_a_stage_more_than_twenty_percent_slower_than_the_last_commit_is_flagged():
    rows = [
        _row("a", "skeletonise", 100.0), _row("a", "graph", 200.0), _row("a", "segment", 0.5),
        _row("b", "skeletonise", 110.0), _row("b", "graph", 150.0),
        _row("c", "skeletonise", 135.0), _row("c", "graph", 260.0), _row("c", "segment", 0.9),
    ]
    flagged = trends.slower_stages(rows, "c")
    # Against b, the latest earlier commit with each stage: skeletonise
    # 110 -> 135 (+23%), graph 150 -> 260; segment is under a second.
    assert [(each.stage, each.before_commit) for each in flagged] == [("graph", "b"), ("skeletonise", "b")]
    assert flagged[0].ratio == pytest.approx(260 / 150)
    assert trends.slower_stages(rows, "b") == []
    assert trends.slower_stages(rows, "z") == []


def test_repeated_runs_compare_by_their_fastest_and_failed_stages_not_at_all():
    rows = [_row("a", "graph", 100.0), _row("b", "graph", 500.0), _row("b", "graph", 115.0),
            _row("b", "solve", 300.0, status="failed"), _row("a", "solve", 10.0)]
    assert trends.slower_stages(rows, "b") == []


def test_rows_append_with_one_header(tmp_path):
    path = tmp_path / "benchmarks.csv"
    trends.append_rows(path, [_row("a", "graph", 1.0)])
    trends.append_rows(path, [_row("b", "graph", 2.0)])
    read = trends.read_rows(path)
    assert [row["commit"] for row in read] == ["a", "b"]
    assert path.read_text().count("date,commit") == 1


def test_growth_with_volume():
    linear = [(1e6, 10.0), (2e6, 20.0), (4e6, 40.0)]
    assert trends.growth_exponent(linear) == pytest.approx(1.0)
    squared = [(1e6, 1.0), (2e6, 4.0), (4e6, 16.0)]
    assert trends.growth_exponent(squared) == pytest.approx(2.0)
    assert trends.growth_exponent([(1e6, 1.0), (1e6, 2.0)]) is None
    rows = [_row("a", "graph", seconds, dataset=name, voxels=voxels)
            for name, voxels, seconds in (("e", 8e6, 80.0), ("e8", 1e6, 10.0), ("e4", 2e6, 20.0))]
    rows += [_row("a", "solve", seconds, dataset=name, voxels=voxels)
             for name, voxels, seconds in (("e", 8e6, 64.0), ("e8", 1e6, 1.0), ("e4", 2e6, 4.0))]
    found = trends.scaling(rows, ["e", "e8", "e4"])
    assert found["sizes"] == 3 and found["faster_than_volume"] == ["solve"]
