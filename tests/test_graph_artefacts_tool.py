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
from graph_artefacts.report import (
    COLUMNS, bridge_shapes, bridge_shapes_table, comparison_table, measure, steps_table,
)
from haemolynx.graph import build_graph_from_skeleton
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import graph_build_arguments, thick_junction_bridge_arguments
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
    # Two full-length strands: two long runs, all of the doubling mid-edge.
    assert row["doubled_long_runs"] == 2
    assert row["doubled_mid_um"] == pytest.approx(row["doubled_um"], rel=0.05)
    assert row["doubled_chain_um"] == row["doubled_fork_um"] == row["doubled_bridge_um"] == 0
    mask, G = loop_round_tissue()
    row = measure(G, MaskSupport(mask, VOXEL_SIZE))
    assert (row["loops"], row["loops_in_lumen"]) == (1, 0)
    assert (row["loops_doubled_side"], row["loops_through_bridge"], row["loops_triangle"]) == (0, 0, 0)
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
        "split": thick_junction_bridge_arguments(default_schema().defaults()) | {"a_setting_from_the_future": 1},
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
    # Its shape, measured by this checkout against the region's tree.
    report = bridge_shapes(saved, thick, mask, (1.0, 1.0, 1.0))
    assert report["bridge_count"] == 1
    assert report["counts"]["radius_shaped"] + report["counts"]["longer_than_radius"] == 1


def test_a_side_splits_with_the_runs_thick_junction_choices(tmp_path, monkeypatch):
    """The split takes the run's settings (``thick_junction_bridge_arguments``),
    as build_network does, not the function's defaults: those this version
    takes."""
    from haemolynx import graph

    seen = {}

    def split(G, thick, *, voxel_size_zyx, segmentation=None, extend_thick_ends=False, bridge_forks=True):
        seen.update(extend_thick_ends=extend_thick_ends, bridge_forks=bridge_forks)
        return G

    monkeypatch.setattr(graph, "insert_thick_vessel_junction_nodes", split)
    skeleton = np.zeros((20, 20, 20), dtype=bool)
    skeleton[2:18, 10, 10] = True
    from scipy import ndimage

    for name, volume in (("skeleton", skeleton), ("mask", ndimage.binary_dilation(skeleton, iterations=2))):
        np.save(tmp_path / f"{name}.npy", volume)
    np.save(tmp_path / "thick.npy", np.zeros_like(skeleton))
    job = {
        "src": str(REPO / "src"), "out": str(tmp_path / "side"),
        "skeleton": str(tmp_path / "skeleton.npy"), "mask": str(tmp_path / "mask.npy"),
        "voxel_size_zyx": [1.0, 1.0, 1.0],
        "build": graph_build_arguments(default_schema().defaults()),
        "smoothing": None,
        "thick_vessel_mask": str(tmp_path / "thick.npy"),
        "split": {"extend_thick_ends": True, "bridge_forks": False, "fill_tree_gaps": True},
    }
    (tmp_path / "job.json").write_text(json.dumps(job))

    build_side.main(str(tmp_path / "job.json"))

    assert seen == {"extend_thick_ends": True, "bridge_forks": False}


def test_the_bridge_shape_table_has_a_row_per_shape_and_a_column_per_side():
    from haemolynx.graph.thick_vessel_bridge_shapes import BRIDGE_SHAPES

    empty = {shape: 0 for shape in BRIDGE_SHAPES}
    none = {shape: None for shape in BRIDGE_SHAPES}
    before = {"bridge_count": 3, "counts": empty | {"wall_to_wall": 3}, "median_length_um": none | {"wall_to_wall": 5.2},
              "length_over_radius": {"median": 0.9, "p90": 1.4, "max": 2.0}, "drawn_path_count": 0}
    after = {"bridge_count": 2, "counts": empty | {"radius_shaped": 2}, "median_length_um": none | {"radius_shaped": 7.24},
             "length_over_radius": {"median": None, "p90": None, "max": None}, "drawn_path_count": 1}

    lines = bridge_shapes_table({"5fab427": before, "working tree": after}).splitlines()

    assert lines[0] == "| bridge shape (count, median length) | 5fab427 | working tree |"
    assert "| radius-shaped | 0 | 2 (7.2 um) |" in lines
    assert "| wall to wall through the margin | 3 (5.2 um) | 0 |" in lines
    assert "| all bridges | 3 | 2 |" in lines
    assert "| length / lumen radius, median (90th pct) | 0.90 (1.40) |  |" in lines
    assert lines[-1] == "| drawn along their path | 0 | 1 |"
    assert len(lines) == 2 + len(BRIDGE_SHAPES) + 3


def test_a_row_counts_the_pairs_a_thick_vessel_bridge_is_in():
    from haemolynx.graph import IS_ZERO_RESISTANCE

    mask, G = two_strands_in_one_lumen()
    assert measure(G, MaskSupport(mask, VOXEL_SIZE))["bridge_pairs"] == 0
    u, v, key = next(iter(G.edges(keys=True)))
    G.edges[u, v, key][IS_ZERO_RESISTANCE] = True

    row = measure(G, MaskSupport(mask, VOXEL_SIZE))

    assert (row["pairs"], row["bridge_pairs"], row["bridges"]) == (1, 1, 1)


# --- the dead-end census (PLAN_dead_ends.md, Phase 0) ------------------------------


def test_the_census_counts_each_fixture_by_kind_place_and_what_lies_ahead():
    from dead_end_fixtures import CASES, SHAPE
    from graph_artefacts.dead_end_census import census, census_markdown
    from haemolynx.graph.dead_ends import DEAD_END_KINDS, image_extent_um

    extent = image_extent_um(SHAPE, VOXEL_SIZE)
    for make in CASES:
        case = make()
        row = census(case.graph, support=MaskSupport(case.mask, VOXEL_SIZE), extent_um=extent)
        assert row["dead_ends"] == sum(row["by_kind"].values())
        assert tuple(row["by_kind"]) == DEAD_END_KINDS
        assert row["by_kind"][case.kind] >= 1, case.name
        assert sum(sum(c.values()) for c in row["by_face_band"].values()) == row["dead_ends"]
    broken = CASES[4]()
    row = census(broken.graph, support=MaskSupport(broken.mask, VOXEL_SIZE), extent_um=extent)
    assert row["interior_with_a_vessel_ahead"] == {"within 5 um": 0, "within 10 um": 1}
    # The stopped-short tip reaches the face through the mask, 5-10 um out.
    face = CASES[1]()
    row = census(face.graph, support=MaskSupport(face.mask, VOXEL_SIZE), extent_um=extent)
    assert row["by_face_band"]["5-10"] == {"reaches face": 1}
    assert "| leaves_image | 3 | 3 |" in census_markdown({"a": row, "b": row})


def test_each_final_dead_end_is_traced_to_the_step_that_first_left_it():
    import networkx as nx
    from graph_artefacts.dead_end_census import made_by_step

    def chain(*points):
        G = nx.MultiGraph()
        for p in points:
            G.add_node(p, pos=np.asarray(p, dtype=float))
        for a, b in zip(points[:-1], points[1:]):
            G.add_edge(a, b, length=float(np.linalg.norm(np.subtract(b, a))))
        return G

    first = chain((0, 0, 0), (0, 0, 10))
    second = chain((0, 0, 0), (0, 0, 10), (0, 0, 29.5))
    final = chain((0, 0, 0), (0, 0, 10), (0, 0, 20), (0, 0, 30))
    # (0, 0, 0) was a dead end from the first step on, (0, 0, 30) from the
    # second (within a micron of 29.5).
    assert made_by_step(final, [("first", first), ("second", second)]) == {
        "first": 1, "second": 1, "after the steps": 0,
    }
    assert made_by_step(final, [("first", first)]) == {"first": 1, "after the steps": 1}


def test_flow_coverage_says_why_each_unsolved_piece_has_no_flow():
    import networkx as nx
    from graph_artefacts.dead_end_census import flow_coverage
    from haemolynx.graph import FLOW_SOLVED
    from haemolynx.graph.dead_ends import DEAD_END_KIND, LEAVES_IMAGE, UNKNOWN

    G = nx.MultiGraph()
    pos = {0: (5, 5, 0), 1: (5, 5, 10), 2: (5, 5, 20), 3: (5, 0, 10), 4: (5, 9, 20), 5: (5, 5, 50), 6: (5, 5, 60)}
    for n, p in pos.items():
        G.add_node(n, pos=np.asarray(p, dtype=float))
    for u, v, solved in ((0, 1, True), (1, 2, True), (1, 3, False), (2, 4, False), (5, 6, False)):
        G.add_edge(u, v, length=10.0, **{FLOW_SOLVED: solved})
    G.nodes[3][DEAD_END_KIND] = LEAVES_IMAGE   # node 3 on the y = 0 face
    G.nodes[4][DEAD_END_KIND] = UNKNOWN
    G.graph["image_extent_um"] = (10.0, 10.0, 100.0)
    report = flow_coverage(G, inlets=[0], outlets=[2])
    assert report["solved_share"] == pytest.approx(0.4)
    assert report["unsolved_um_by_reason"] == {
        "behind a closed face end": 10.0, "behind interior dead ends": 10.0, "no boundary": 10.0,
    }
    # Opening node 3 as a boundary puts 0-1-3 on a path: 1-3 joins the flow.
    assert report["through_designated_um"] == pytest.approx(20.0)
    assert report["through_with_open_ends_um"] == pytest.approx(30.0)
    assert report["boundary_face_distance_um"] == {0: 0.0, 2: pytest.approx(5.0)}
