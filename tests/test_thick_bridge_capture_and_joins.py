"""The capture of one run's steps (``scripts/thick_bridge_review/capture_steps.py``),
the thin-arm join's paths (``join_paths.py``) and their census grid
(``join_census.py``), on a synthetic thick vessel with a capillary opening
into it: what is captured, what the join drew, how its arm meets it, which
paths the census offers, and the capture traced back by ``connection_steps.py``."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from thick_bridge_review.connection_steps import StepIndex, _unpacked
from thick_bridge_review.join_census import census_cases, other_way_um
from thick_bridge_review.join_paths import approach, arm_voxels, join_paths


def _tube(shape, a, b, radius):
    p = np.stack(np.indices(shape), axis=-1).astype(float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1) <= radius


def _line(z, y, xs):
    return [(z, y, x) for x in xs]


def _skeletons(arm):
    """A tree along y at x = 10 and a Lee *arm*, before the join; after it,
    the join from x = 11 to 15 along y = 20."""
    shape = (12, 40, 40)
    tree = np.zeros(shape, bool)
    tree[6, 2:38, 10] = True
    before = tree.copy()
    before[tuple(np.array(arm).T)] = True
    after = before.copy()
    after[tuple(np.array(_line(6, 20, range(11, 16))).T)] = True
    return before, after, tree


def test_each_join_runs_from_the_arm_it_joined_to_the_tree():
    before, after, tree = _skeletons(_line(6, 20, range(16, 35)))
    (path,) = join_paths(before, after, tree)
    assert tuple(path["start"]) == (6, 20, 16)
    assert tuple(path["end"]) in {(6, y, 10) for y in (19, 20, 21)}
    assert [tuple(v) for v in path["path"]] == _line(6, 20, range(15, 10, -1))


def test_an_arm_heading_in_meets_its_join_end_on_and_one_running_past_meets_it_side_on():
    vs = (1.0, 1.0, 1.0)
    before, after, tree = _skeletons(_line(6, 20, range(16, 35)))
    (path,) = join_paths(before, after, tree)
    heading_in = approach(path, before & ~tree, vs)
    assert heading_in["approach_cos"] == pytest.approx(1.0, abs=0.02)
    assert heading_in["start_degree"] == 1
    assert heading_in["chord_um"] == pytest.approx(6.0, abs=1.1)

    before, after, tree = _skeletons([(6, y, 16) for y in range(8, 33)])
    (path,) = join_paths(before, after, tree)
    running_past = approach(path, before & ~tree, vs)
    assert running_past["approach_cos"] < 0.1
    assert running_past["start_degree"] == 2

    before, after, tree = _skeletons([(6, 20, 16)])
    (path,) = join_paths(before, after, tree)
    speck = approach(path, before & ~tree, vs)
    assert speck["start_degree"] == 0 and np.isnan(speck["approach_cos"])
    # The arm is walked along its own skeleton, not taken from a box round the start.
    lee = np.zeros((12, 40, 40), bool)
    lee[6, 20, 16:30] = True
    lee[6, 23, 16:30] = True
    assert len(arm_voxels(lee, (6, 20, 16), vs, reach_um=5.0)) == 6


def _edge(G, u, v, a, b, **attrs):
    count = int(np.ceil(np.linalg.norm(np.subtract(b, a)))) + 1
    path = [tuple(float(c) for c in q) for q in np.linspace(a, b, count)]
    G.add_node(u, pos=path[0])
    G.add_node(v, pos=path[-1])
    G.add_edge(u, v, voxels=path, length=float(np.linalg.norm(np.subtract(b, a))), **attrs)


def test_the_other_way_is_the_loop_a_connection_closes():
    G = nx.MultiGraph()
    _edge(G, "u", "v", (0, 0, 0), (0, 0, 5))
    _edge(G, "u", "a", (0, 0, 0), (0, 10, 0))
    _edge(G, "a", "v", (0, 10, 0), (0, 0, 5))
    assert other_way_um(G, ("u", "v", 0)) == pytest.approx(10 + np.hypot(10, 5))
    G.remove_edge("a", "v")
    assert other_way_um(G, ("u", "v", 0)) == np.inf
    _edge(G, "u", "v", (0, 0, 0), (0, 0, 5))
    assert other_way_um(G, ("u", "v", 0)) == pytest.approx(5.0)
    assert np.isnan(other_way_um(G, None))


def test_the_census_offers_the_fresh_paths_the_saved_graph_still_joins():
    """Two capillaries joined to the tree at y = 20 and y = 30: the saved
    graph still joins the first (its bridge), not the second (its vessel
    stops at x = 20). With an earlier grid's piece on the first, none is
    fresh."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    shape = (12, 40, 40)
    tree = np.zeros(shape, bool)
    tree[6, 2:38, 10] = True
    before = tree.copy()
    after = tree.copy()
    for y in (20, 30):
        before[6, y, 16:35] = True
        after[6, y, 11:35] = True
    G = nx.MultiGraph()
    _edge(G, "T0", "T1", (6, 2, 10), (6, 37, 10))
    _edge(G, "A0", "A1", (6, 20, 34), (6, 20, 16))
    _edge(G, "A1", "C", (6, 20, 16), (6, 20, 10), **{IS_ZERO_RESISTANCE: True})
    _edge(G, "B0", "B1", (6, 30, 34), (6, 30, 20))
    final = StepIndex(G)
    vs = (1.0, 1.0, 1.0)
    tree_points = np.argwhere(tree).astype(float)

    cases = census_cases(join_paths(before, after, tree), before & ~tree, final, vs, tree_points, None)

    (case,) = cases
    assert case["id"] == "J01" and case["rule"] == "join"
    assert case["start_vox"] == (6, 20, 16) and case["end_vox"][2] == 10
    assert case["final_bridge_share"] > 0.5 and case["other_way_um"] == np.inf
    assert case["view"] == "down z" and case["approach_cos"] == pytest.approx(1.0, abs=0.02)
    earlier = np.array([[6.0, 20.0, 13.0]])
    assert census_cases(join_paths(before, after, tree), before & ~tree, final, vs, tree_points, earlier) == []


@pytest.mark.slow
def test_a_capture_keeps_every_step_and_traces_the_bridge_back_to_the_join(tmp_path: Path):
    """A thick vessel along y and a capillary opening into it along x: the
    capture keeps the skeleton before and after the join (the join drew the
    way in), every topology step and every step after it in order, and the
    graph the split was handed; the tracer puts the run's bridge down to the
    join."""
    from haemolynx.pipeline import default_schema, resolve_settings
    from thick_bridge_review.capture_steps import CAPTURE_OVERRIDES, capture_run
    from thick_bridge_review.connection_steps import main as trace_main

    # A trunk 8 um in radius and a capillary of 2: under a 5 um gate the
    # capillary stays thin, and Lee draws it to 5 um from the trunk's region.
    shape = (28, 72, 92)
    mask = _tube(shape, (14, 2, 14), (14, 69, 14), 8.0) | _tube(shape, (14, 36, 14), (14, 36, 74), 2.0)
    tifffile.imwrite(tmp_path / "vessels.tif", mask.astype(np.uint8))
    schema = default_schema()
    values = schema.defaults()
    values.update({
        "input_path": tmp_path / "vessels.tif", "vtk_output_prefix": tmp_path / "out" / "run",
        "plot_dir": tmp_path / "plots", "voxel_size_override_xyz": [1.0, 1.0, 1.0],
        "use_thick_vessel_skeletonisation": True, "skeleton_thick_vessel_min_radius_um": 5.0,
        "skeleton_thick_vessel_restrict_to_mask": "off", **CAPTURE_OVERRIDES,
    })
    settings = resolve_settings(values, schema=schema, config_path=None)
    run = tmp_path / "run"
    capture_run(settings, schema, run)

    before = _unpacked(run / "skel_before_join.npz")
    after = _unpacked(run / "skel_after_join.npz")
    assert (after >= before).all() and (after & ~before).any()
    assert _unpacked(run / "thick.npz").any() and len(np.load(run / "tree.npy"))
    labels = [p.stem.split("_", 1)[1] for p in sorted((run / "steps").glob("*.pkl"))]
    assert labels[0] == "build_graph_segment_skan_stitched_loops"
    assert labels.index("consolidate_lumen") < labels.index("finish_thick_split")
    assert "finish_smoothing" in labels
    with open(run / "split_call.pkl", "rb") as f:
        assert pickle.load(f)["G"].number_of_edges()
    with open(run / "graph_built.pkl", "rb") as f:
        G = pickle.load(f)
    assert any(d.get("is_zero_resistance") for _u, _v, d in G.edges(data=True))

    out = tmp_path / "trace"
    trace_main(["--run", str(run), "--grid", str(tmp_path), "--all-bridges", "--out", str(out)])
    rows = (out / "trace.csv").read_text(encoding="utf-8").splitlines()
    capillary = [r for r in rows[1:] if ",bridge," in r and ",join," in r]
    assert capillary, rows
    assert json.loads((run / "voxel_size_zyx.json").read_text()) == [1.0, 1.0, 1.0]
    assert (_unpacked(run / "mask.npz") == mask).all()

    from thick_bridge_review.join_census import main as census_main

    grid = tmp_path / "census"
    census_main(["--run", str(run), "--out", str(grid)])
    key = (grid / "key.csv").read_text(encoding="utf-8").splitlines()
    assert key[0].startswith("id,join,") and key[1].startswith("J01,")
    data = json.loads((grid / "page.json").read_text(encoding="utf-8"))
    (sheet,) = data["sheets"]
    assert [m["key"] for m in sheet["marks"]] == ["J", "C", "N", "?"]
    assert len(sheet["cases"]) == len(key) - 1 and sheet["cases"][0]["piece"]
    assert "Thick-vessel grid J" in (grid / "grid_J.html").read_text(encoding="utf-8")
