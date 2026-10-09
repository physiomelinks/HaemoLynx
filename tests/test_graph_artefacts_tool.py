"""The graph-artefact comparison tool (``scripts/graph_artefacts.py``): its
measuring and tables, the build script each code version runs, and the one
settings-to-arguments mapping it shares with the pipeline. The full
comparison is not a test and never runs in CI."""
from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from graph_artefacts import build_side
from graph_artefacts.report import COLUMNS, comparison_table, measure, steps_table
from haemolynx.graph import build_graph_from_skeleton
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import graph_build_arguments
from haemolynx.preprocessing import MaskSupport

from lumen_artefact_fixtures import VOXEL_SIZE, loop_round_tissue, two_strands_in_one_lumen

REPO = Path(__file__).resolve().parent.parent


def test_every_setting_mapped_to_graph_building_is_one_it_takes():
    arguments = graph_build_arguments(default_schema().defaults())
    taken = inspect.signature(build_graph_from_skeleton).parameters
    assert set(arguments) <= set(taken)
    assert arguments["min_stub_length_radius_multiple"] == default_schema()["min_stub_length_radius_multiple"].default


def test_an_older_version_is_built_without_the_settings_it_lacks():
    def old(skeleton, voxel_size=(1, 1, 1), min_stub_length=10.0):
        return None

    taken, dropped = build_side.accepted_arguments(old, {"min_stub_length": 5.0, "facing_dead_end_max_gap_um": 10.0})
    assert taken == {"min_stub_length": 5.0}
    assert dropped == ["facing_dead_end_max_gap_um"]
    taken, dropped = build_side.accepted_arguments(lambda **kw: None, {"anything": 1})
    assert taken == {"anything": 1} and dropped == []


def test_a_row_counts_what_the_report_finds():
    mask, G = two_strands_in_one_lumen()
    row = measure(G, MaskSupport(mask, VOXEL_SIZE), coverage=True)
    assert row["pairs"] == 1 and row["forks"] == 0
    assert row["edges"] == 2 and row["components"] == 2
    assert 0.0 < row["mask_covered"] <= 1.0
    mask, G = loop_round_tissue()
    row = measure(G, MaskSupport(mask, VOXEL_SIZE))
    assert (row["loops"], row["loops_in_lumen"]) == (1, 0)
    assert "mask_covered" not in row


def test_the_tables_have_a_column_per_measure_and_the_change_between_two_sides():
    first = {key: 10 for key, _ in COLUMNS} | {"mask_covered": 0.647}
    second = {key: 4 for key, _ in COLUMNS} | {"mask_covered": 0.628}
    table = comparison_table({"5fab427": first, "working tree": second})
    lines = table.splitlines()
    assert lines[0] == "| measure | 5fab427 | working tree | change |"
    assert "| pairs in one lumen | 10 | 4 | -6 |" in lines
    assert "| mask covered | 64.7% | 62.8% | -1.9 points |" in lines
    steps = steps_table([("a", {"edges": 3, "pairs": 1}), ("b", {"edges": 2, "pairs": 0})]).splitlines()
    assert steps[0] == "| step | edges | pairs in one lumen |"
    assert steps[2:] == ["| a | 3 | 1 |", "| b | 2 | 0 |"]


@pytest.mark.slow
def test_a_side_builds_with_its_own_code_and_leaves_every_step(tmp_path):
    """The script each version runs, on a small T with its mask, importing
    this checkout's src as a ref's would be."""
    skeleton = np.zeros((40, 40, 40), dtype=bool)
    skeleton[2:38, 20, 20] = True
    skeleton[20, 21:38, 20] = True
    from scipy import ndimage

    mask = ndimage.binary_dilation(skeleton, iterations=2)
    np.save(tmp_path / "skeleton.npy", skeleton)
    np.save(tmp_path / "mask.npy", mask)
    arguments = graph_build_arguments(default_schema().defaults())
    job = {
        "src": str(REPO / "src"), "out": str(tmp_path / "side"),
        "skeleton": str(tmp_path / "skeleton.npy"), "mask": str(tmp_path / "mask.npy"),
        "voxel_size_zyx": [1.0, 1.0, 1.0],
        "build": arguments | {"a_setting_from_the_future": 1},
        "smoothing": {"method": "taubin", "iterations": 10, "max_deviation": 1.0},
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    build_side.main(str(tmp_path / "job.json"))
    info = json.loads((tmp_path / "side" / "side.json").read_text())
    from haemolynx.graph import STEP_LABELS

    assert info["steps"] == list(STEP_LABELS)
    assert info["arguments_not_taken"] == ["a_setting_from_the_future"]
    for label in STEP_LABELS:
        assert (tmp_path / "side" / "steps" / f"{label}.pkl").exists()
    with open(tmp_path / "side" / "final.pkl", "rb") as f:
        G = pickle.load(f)
    assert G.number_of_edges() == 3


def test_a_side_given_a_thick_vessel_region_splits_the_graph_as_build_network_saves_it(tmp_path):
    """With the thick-vessel region in the job, the graph measured is the
    graph saved: the branch opening into the fat trunk is split at the
    region's edge, its part inside a zero-resistance bridge. Before the split
    is kept too."""
    from scipy import ndimage

    from haemolynx.graph import IS_ZERO_RESISTANCE

    skeleton = np.zeros((40, 40, 40), dtype=bool)
    skeleton[2:38, 20, 20] = True
    skeleton[20, 21:38, 20] = True
    mask = ndimage.binary_dilation(skeleton, iterations=2)
    trunk = np.zeros_like(skeleton)
    trunk[2:38, 20, 20] = True
    thick = ndimage.binary_dilation(trunk, iterations=4)
    for name, volume in (("skeleton", skeleton), ("mask", mask), ("thick", thick)):
        np.save(tmp_path / f"{name}.npy", volume)
    job = {
        "src": str(REPO / "src"), "out": str(tmp_path / "side"),
        "skeleton": str(tmp_path / "skeleton.npy"), "mask": str(tmp_path / "mask.npy"),
        "voxel_size_zyx": [1.0, 1.0, 1.0],
        "build": graph_build_arguments(default_schema().defaults()),
        "smoothing": None,
        "thick_vessel_mask": str(tmp_path / "thick.npy"),
    }
    (tmp_path / "job.json").write_text(json.dumps(job))

    build_side.main(str(tmp_path / "job.json"))

    side = tmp_path / "side"
    assert json.loads((side / "side.json").read_text())["thick_split"] is True
    with open(side / "before_thick_split.pkl", "rb") as f:
        before = pickle.load(f)
    with open(side / "final.pkl", "rb") as f:
        saved = pickle.load(f)
    bridges = [d for *_, d in saved.edges(data=True) if d.get(IS_ZERO_RESISTANCE)]
    assert len(bridges) == 1
    assert not any(d.get(IS_ZERO_RESISTANCE) for *_, d in before.edges(data=True))
    assert saved.number_of_edges() == before.number_of_edges() + 1


def test_a_row_counts_the_pairs_a_thick_vessel_bridge_is_in():
    from haemolynx.graph import IS_ZERO_RESISTANCE

    mask, G = two_strands_in_one_lumen()
    assert measure(G, MaskSupport(mask, VOXEL_SIZE))["bridge_pairs"] == 0
    u, v, key = next(iter(G.edges(keys=True)))
    G.edges[u, v, key][IS_ZERO_RESISTANCE] = True

    row = measure(G, MaskSupport(mask, VOXEL_SIZE))

    assert (row["pairs"], row["bridge_pairs"], row["bridges"]) == (1, 1, 1)
