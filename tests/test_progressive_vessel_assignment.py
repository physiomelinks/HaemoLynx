"""Tests for progressive vessel-mask assignment dilation."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph import (
    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation,
    select_terminal_nodes_from_large_vessel_masks_progressive_dilation,
)
from haemolynx.graph.automated_vessel_assignment import _build_dilation_schedule_microns
from haemolynx.pipeline import default_schema


def test_build_dilation_schedule_includes_zero_and_exact_max():
    assert _build_dilation_schedule_microns(
        max_dilation_microns=0.0, dilation_step_microns=5.0
    ) == [0.0]
    assert _build_dilation_schedule_microns(
        max_dilation_microns=10.0, dilation_step_microns=5.0
    ) == [0.0, 5.0, 10.0]
    assert _build_dilation_schedule_microns(
        max_dilation_microns=12.0, dilation_step_microns=5.0
    ) == [0.0, 5.0, 10.0, 12.0]


def test_progressive_dilation_assignment_locks_earlier_nodes():
    """Nodes assigned early remain fixed across later dilation steps.

    Positions and masks use canonical physical (z, y, x).
    """
    G = nx.MultiGraph()
    # Terminal near arteriole and venule volumes with staged overlap behavior.
    G.add_node(0, pos=np.array([1.0, 1.0, 1.0]))  # arteriole at 0 µm
    G.add_node(1, pos=np.array([1.0, 1.0, 7.0]))  # venule at +5, arteriole at +10
    G.add_node(2, pos=np.array([1.0, 1.0, 9.0]))  # venule at 0 µm
    G.add_node(10, pos=np.array([1.0, 1.0, 2.0]))
    G.add_node(11, pos=np.array([1.0, 1.0, 7.0]))
    G.add_edge(0, 10, length=1.0)
    G.add_edge(10, 11, length=5.0)
    G.add_edge(11, 2, length=2.0)
    G.add_edge(1, 10, length=5.0)

    arteriole_mask = np.zeros((12, 12, 12), dtype=bool)
    venule_mask = np.zeros((12, 12, 12), dtype=bool)
    arteriole_mask[1, 1, 1] = True
    venule_mask[1, 1, 9] = True

    start_nodes, out_nodes = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G,
        large_arteriole_mask=arteriole_mask,
        large_venule_mask=venule_mask,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=10.0,
        dilation_step_microns=5.0,
        allow_overlap=False,
    )
    assert start_nodes == [0]
    assert out_nodes == [1, 2]


def test_progressive_small_vessel_zero_max_matches_single_shot_shape():
    """A 0 µm schedule labels art/ven edges and finds capillary transitions."""
    G = nx.MultiGraph()
    # art -- art/cap -- capillary -- ven/cap -- ven -- capillary past venule
    for node, z in enumerate((0.0, 2.0, 4.0, 6.0, 8.0, 10.0)):
        G.add_node(node, pos=np.array([z, 3.0, 3.0]))
    for node in range(5):
        z0, z1 = float(node * 2), float(node * 2 + 2)
        G.add_edge(
            node,
            node + 1,
            voxels=[[z, 3.0, 3.0] for z in (z0, z1)],
        )

    art = np.zeros((12, 8, 8), dtype=bool)
    ven = np.zeros((12, 8, 8), dtype=bool)
    # Each mask covers most of the length of the edges it labels -- the
    # overlap is a fraction of an edge's length, sampled every micron.
    art[0:4, 3, 3] = True
    # Cover the venule segments (z=4..8) but leave the distal edge (z=8..10)
    # unlabeled so a capillary transition exists at the venule/capillary boundary.
    ven[5:8, 3, 3] = True

    result = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G,
        small_arteriole_mask=art,
        small_venule_mask=ven,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        max_dilation_microns=0.0,
        minimum_overlap_fraction=0.5,
        allow_overlap=False,
    )
    assert result["arteriole_boundary_nodes"] == [2]
    assert result["venule_boundary_nodes"] == [4]
    assert result["arteriole_nodes"] == [0, 1, 2]
    assert result["venule_nodes"] == [3, 4]
    assert result["arteriole_edge_count"] == 2
    assert result["venule_edge_count"] == 2
    assert set(result["arteriole_nodes"]).isdisjoint(result["venule_nodes"])
    for node_id in result["arteriole_boundary_nodes"]:
        assert G.nodes[node_id].get("mask_vessel_type") == "arteriole"
    for node_id in result["venule_boundary_nodes"]:
        assert G.nodes[node_id].get("mask_vessel_type") == "venule"


def test_progressive_dilation_schema_settings_and_requires():
    schema = default_schema()
    load = schema["large_vessel_mask_dilation_microns"]
    assign = schema["large_vessel_assignment_max_dilation_microns"]
    small = schema["small_vessel_mask_dilation_microns"]

    load_help = load.help.lower()
    assert "one-shot" in load_help
    assert "load-time" in load_help
    assert "progressive" in assign.help.lower()
    assert "progressive" in small.help.lower()
    assert "load-time one-shot" in assign.help.lower()
    assert assign.requires == ("use_large_vessel_masks", "automated_vessel_assignment")
    assert small.requires == (
        "use_small_vessel_masks_for_boundary_assignment",
        "automated_vessel_assignment",
    )
    assert assign.default == 0.0
    assert small.default == 0.0


# --- read from each terminal's distance, not from whole-volume step masks ------


def _star_with_masks(seed):
    """A star of terminals around a hub, and random blobs for the two masks."""
    rng = np.random.default_rng(seed)
    shape = (20, 24, 24)
    G = nx.MultiGraph()
    G.add_node("hub", pos=np.array([10.0, 12.0, 12.0]))
    for i in range(14):
        pos = np.array([rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(0, shape[2])], dtype=float)
        G.add_node(i, pos=pos)
        G.add_edge("hub", i)
    zz, yy, xx = np.indices(shape)
    art = np.zeros(shape, dtype=bool)
    ven = np.zeros(shape, dtype=bool)
    for mask in (art, ven):
        for _ in range(2):
            c = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(0, shape[2])
            mask |= (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= int(rng.integers(2, 5)) ** 2
    ven &= ~art  # the masks touch but never overlap
    return G, art, ven


def _reference_first_steps(G, mask, schedule, voxel_size):
    """The step-mask algorithm: grow the mask step by step, note the first
    step that reaches each terminal."""
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~mask, sampling=voxel_size)
    first = {}
    for k, dilation in enumerate(schedule):
        grown = mask | (distance <= dilation)
        for node in G.nodes:
            if node == "hub" or node in first:
                continue
            index = tuple(int(v) for v in np.rint(np.asarray(G.nodes[node]["pos"]) / voxel_size))
            if grown[index]:
                first[node] = k
    return first


@pytest.mark.parametrize("seed", range(6))
def test_progressive_assignment_matches_growing_the_masks_step_by_step(seed):
    """Every terminal the two masks reach at different steps is assigned just
    as growing the masks step by step would assign it."""
    G, art, ven = _star_with_masks(seed)
    voxel_size = (2.0, 0.5, 0.5)
    for node in G.nodes:  # positions are voxel indices above; make them microns
        G.nodes[node]["pos"] = G.nodes[node]["pos"] * np.asarray(voxel_size)
    schedule = [0.0, 1.5, 3.0, 4.5, 5.0]

    inputs, outputs = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=voxel_size, max_dilation_microns=5.0, dilation_step_microns=1.5,
    )

    art_first = _reference_first_steps(G, art, schedule, voxel_size)
    ven_first = _reference_first_steps(G, ven, schedule, voxel_size)
    for node in (n for n in G.nodes if n != "hub"):
        a, v = art_first.get(node), ven_first.get(node)
        if a is not None and (v is None or a < v):
            assert node in inputs
        elif v is not None and (a is None or v < a):
            assert node in outputs
        elif a is None and v is None:
            assert node not in inputs and node not in outputs


def test_progressive_assignment_builds_no_whole_volume_mask_per_step(monkeypatch):
    """Regression: two volume-sized masks were built at every step, to read a
    handful of terminal voxels from."""
    import haemolynx.graph.automated_vessel_assignment as module

    def refuse(*_args, **_kwargs):
        raise AssertionError("a whole-volume step mask was built")

    monkeypatch.setattr(module, "_dilated_mask_from_cached_distance", refuse)
    monkeypatch.setattr(module, "distance_transform_edt", refuse)
    G, art, ven = _star_with_masks(0)

    inputs, outputs = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=(1.0, 1.0, 1.0), max_dilation_microns=10.0,
    )

    assert inputs or outputs


def test_a_terminal_both_masks_reach_at_one_step_goes_to_the_nearer_mask():
    """Regression: a same-step tie was settled by whole-mask midpoints; a big
    arteriole whose middle is far away beat a venule right beside the
    terminal."""
    shape = (12, 12, 60)
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([6.0, 6.0, 30.0]))  # the terminal
    G.add_node(1, pos=np.array([6.0, 6.0, 20.0]))
    G.add_edge(0, 1, voxels=[[6.0, 6.0, 30.0], [6.0, 6.0, 20.0]])
    art = np.zeros(shape, dtype=bool)
    ven = np.zeros(shape, dtype=bool)
    art[:, :, 34:60] = True  # 4 um away, a long vessel
    ven[5:8, 9:12, 29:32] = True  # 3 um away, a small one

    inputs, outputs = select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=(1.0, 1.0, 1.0), max_dilation_microns=5.0,
        dilation_step_microns=5.0,
    )

    assert outputs == [0] and 0 not in inputs


def test_a_masks_long_axis_is_measured_in_microns():
    """Regression (audit): the long axis was the one with most voxels, so a
    mask 98 um long in z (50 slices of 2 um) and 74.5 um across x (150 voxels
    of 0.5 um) read as running along x, and the cross-section that resolves an
    inlet/outlet overlap was taken from the wrong slice."""
    from haemolynx.graph.automated_vessel_assignment import (
        _cross_section_midpoint_physical,
        _mask_principal_axis,
    )

    spacing = (2.0, 0.5, 0.5)
    mask = np.zeros((50, 6, 150), dtype=bool)
    mask[:, 2:4, :] = True

    assert _mask_principal_axis(mask, spacing) == 0
    assert _mask_principal_axis(mask) == 2  # counted in voxels
    midpoint = _cross_section_midpoint_physical(mask, spacing, np.array([40.0, 1.5, 10.0]))
    # The cross-section through z = 40 um, centred in y and x.
    assert midpoint[0] == pytest.approx(40.0)
    assert midpoint[2] == pytest.approx(74.5 / 2)


# --- each edge keeps the first step that reaches it ------------------------------


def _arteriole_then_capillaries():
    """An arteriole edge in the mask, then edges 0-11 um and 11-21 um past it."""
    art = np.zeros((5, 5, 40), dtype=bool)
    art[2, 2, 0:10] = True
    ven = np.zeros_like(art)
    G = nx.MultiGraph()
    for node, x in enumerate((0.0, 9.0, 20.0, 30.0)):
        G.add_node(node, pos=(2.0, 2.0, x))
    for node, (x0, x1) in enumerate(((0, 9), (9, 20), (20, 30))):
        G.add_edge(node, node + 1, voxels=[(2.0, 2.0, float(x)) for x in range(x0, x1 + 1)])
    return G, art, ven


@pytest.mark.parametrize("max_dilation, labelled, boundary", [
    (0.0, ["arteriole", None, None], [1]),
    (5.0, ["arteriole", "arteriole", None], [2]),   # half of edge 1-2 within 5 um
    (15.0, ["arteriole", "arteriole", None], [2]),  # 5 of 11 samples of 2-3 within 15 um
    (20.0, ["arteriole", "arteriole", "arteriole"], []),  # no transition left
])
def test_small_vessel_dilation_labels_the_edges_each_step_reaches(max_dilation, labelled, boundary):
    """Regression: every edge's label was locked at the first step that
    labelled *any* edge, the unlabelled ones included, so once the 0 um step
    found one edge no dilation ever labelled another. The boundary follows
    the final labels rather than accumulating over the steps."""
    G, art, ven = _arteriole_then_capillaries()

    result = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=(1.0, 1.0, 1.0), max_dilation_microns=max_dilation,
        dilation_step_microns=5.0, minimum_overlap_fraction=0.5,
    )

    assert [G.edges[u, v, k].get("mask_vessel_type") for u, v, k in G.edges(keys=True)] == labelled
    assert result["arteriole_boundary_nodes"] == boundary
    assert result["arteriole_edge_count"] == sum(label == "arteriole" for label in labelled)


def _chains_with_masks(seed):
    """Short chains hanging off a hub, each leaf's chain reaching no other
    labelled chain, and two random blobs per mask that never overlap."""
    rng = np.random.default_rng(seed)
    shape = (16, 24, 24)
    G = nx.MultiGraph()
    hub = 1000
    G.add_node(hub, pos=np.array([8.0, 12.0, 12.0]))
    node = 0
    for _chain in range(8):
        previous = hub
        p0 = np.asarray(G.nodes[hub]["pos"], dtype=float)
        for _edge in range(2):
            p1 = np.array([rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(0, shape[2])], dtype=float)
            G.add_node(node, pos=p1)
            steps = max(2, int(np.linalg.norm(p1 - p0)))
            G.add_edge(previous, node, voxels=[tuple(p0 + (p1 - p0) * t) for t in np.linspace(0, 1, steps)])
            previous, p0, node = node, p1, node + 1
    zz, yy, xx = np.indices(shape)
    art = np.zeros(shape, dtype=bool)
    ven = np.zeros(shape, dtype=bool)
    for mask in (art, ven):
        for _ in range(2):
            c = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(0, shape[2])
            mask |= (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= int(rng.integers(2, 5)) ** 2
    ven &= ~art
    return G, art, ven


def _reference_labels_growing_the_masks(G, art, ven, schedule, voxel_size, threshold):
    """Grow each mask step by step with a whole-volume transform; an edge's
    label is the first step at which the grown mask covers *threshold* of
    it (both masks: the one covering more, the arteriole on a tie)."""
    from scipy.ndimage import distance_transform_edt

    from haemolynx.graph.automated_vessel_assignment import (
        _edge_sample_points_from_data,
        _sample_overlap_fraction,
    )

    d_art = distance_transform_edt(~art, sampling=voxel_size)
    d_ven = distance_transform_edt(~ven, sampling=voxel_size)
    labels = {}
    for dilation in schedule:
        grown_art = art | (d_art <= dilation)
        grown_ven = ven | (d_ven <= dilation)
        for u, v, key, data in G.edges(keys=True, data=True):
            if (u, v, key) in labels:
                continue
            pu, pv = (np.asarray(G.nodes[n]["pos"], dtype=float) for n in (u, v))
            samples = _edge_sample_points_from_data(data, (pu, pv), step_um=float(min(voxel_size)))
            fa = _sample_overlap_fraction(samples, grown_art, voxel_size_zyx=voxel_size)
            fv = _sample_overlap_fraction(samples, grown_ven, voxel_size_zyx=voxel_size)
            if fa >= threshold and fv >= threshold:
                labels[(u, v, key)] = "arteriole" if fa >= fv else "venule"
            elif fa >= threshold:
                labels[(u, v, key)] = "arteriole"
            elif fv >= threshold:
                labels[(u, v, key)] = "venule"
    return labels


@pytest.mark.parametrize("seed", range(5))
def test_small_vessel_progressive_labels_match_growing_the_masks_step_by_step(seed):
    """Reading each sample's distance once gives every edge the label and
    step that growing the masks with a whole-volume transform gives it."""
    G, art, ven = _chains_with_masks(seed)
    voxel_size = (2.0, 0.5, 0.5)
    for node in G.nodes:
        G.nodes[node]["pos"] = np.asarray(G.nodes[node]["pos"], dtype=float) * np.asarray(voxel_size)
    for _u, _v, data in G.edges(data=True):
        data["voxels"] = [tuple(np.asarray(p) * np.asarray(voxel_size)) for p in data["voxels"]]
    schedule = [0.0, 1.0, 1.5, 3.0, 4.5, 5.0]
    expected = _reference_labels_growing_the_masks(G, art, ven, schedule, voxel_size, 0.5)

    infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=voxel_size, max_dilation_microns=5.0,
        dilation_step_microns=1.5, minimum_overlap_fraction=0.5,
    )

    # Leaves' chains never join two labelled edges of one type, so the gap
    # fill cannot fire and the labels are the per-edge first steps alone.
    got = {(u, v, k): d.get("mask_vessel_type") for u, v, k, d in G.edges(keys=True, data=True)}
    assert {e: l for e, l in got.items() if l is not None} == expected


@pytest.mark.parametrize("max_dilation", [0.0, 10.0])
def test_small_vessel_progressive_labelling_builds_no_whole_volume_transform(monkeypatch, max_dilation):
    """Regression: two float64 distance transforms of the whole volume were
    computed before the first step, whatever the schedule, and two volume
    sized masks built per step."""
    import haemolynx.graph.automated_vessel_assignment as module

    def refuse(*_args, **_kwargs):
        raise AssertionError("a whole-volume transform or step mask was built")

    monkeypatch.setattr(module, "distance_transform_edt", refuse)
    monkeypatch.setattr(module, "_dilated_mask_from_cached_distance", refuse)
    G, art, ven = _arteriole_then_capillaries()

    result = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(
        G, art, ven, voxel_size_zyx=(1.0, 1.0, 1.0), max_dilation_microns=max_dilation,
    )

    assert result["arteriole_edge_count"] >= 1


@pytest.mark.parametrize("seed", range(4))
def test_low_ram_progressive_assignments_read_the_same_distances(seed):
    """Under the low-RAM option each mask's distances come from tiles of its
    surface (``FeatureDistance(by_block=True)``) instead of one tree over all
    of it: the same terminals, labels and boundary nodes."""
    import copy

    voxel_size = (2.0, 0.5, 0.5)
    G, art, ven = _star_with_masks(seed)
    for node in G.nodes:
        G.nodes[node]["pos"] = G.nodes[node]["pos"] * np.asarray(voxel_size)
    kwargs = dict(voxel_size_zyx=voxel_size, max_dilation_microns=5.0, dilation_step_microns=1.5)
    assert select_terminal_nodes_from_large_vessel_masks_progressive_dilation(
        G, art, ven, low_memory=True, **kwargs
    ) == select_terminal_nodes_from_large_vessel_masks_progressive_dilation(G, art, ven, **kwargs)

    G, art, ven = _chains_with_masks(seed)
    for node in G.nodes:
        G.nodes[node]["pos"] = np.asarray(G.nodes[node]["pos"], dtype=float) * np.asarray(voxel_size)
    for _u, _v, data in G.edges(data=True):
        data["voxels"] = [tuple(np.asarray(p) * np.asarray(voxel_size)) for p in data["voxels"]]
    H = copy.deepcopy(G)
    kwargs.update(minimum_overlap_fraction=0.5)
    plain = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(G, art, ven, **kwargs)
    low = infer_boundary_nodes_from_small_vessel_masks_progressive_dilation(H, art, ven, low_memory=True, **kwargs)
    assert repr(low) == repr(plain)
    assert [d.get("mask_vessel_type") for *_e, d in H.edges(keys=True, data=True)] == [
        d.get("mask_vessel_type") for *_e, d in G.edges(keys=True, data=True)
    ]
