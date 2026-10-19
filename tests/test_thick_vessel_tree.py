"""The large-stack centreline tree (``skeleton_thick_vessel_large_stack_tree``,
``preprocessing/thick_vessel_tree.py``) and the thin-arm join's bounded last
resort.

Harvey's patella mask at a 3 um gate has one fat component of 111 M voxels
crossing the stack (2026-10-18), and the tree's 26-neighbour CSR graph alone
was ~40 GB of it. Pinned here: the in-RAM tree (and the low-RAM one without
the setting) stays bit for bit what it was; a component small enough for the
exact walk gets that same tree with the setting; a larger one is still a
connected tree reaching every arm, past the old 500-arm cap, with its
shortest-path distances scipy's own, in tens of bytes a voxel; and the join's
last resort, bounded, finds the same bridge where it fits and leaves an arm
unjoined, logged, where it does not.
"""
from __future__ import annotations

import hashlib
import logging
import tracemalloc

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.measure import euler_number

import haemolynx.preprocessing.thick_vessel_tree as tree_module
import haemolynx.preprocessing.thick_vessels as thick_vessels_module
from haemolynx.preprocessing import (
    THICK_VESSEL_MIN_RADIUS_UM,
    skeletonize_edt_ridge,
    skeletonize_thickness_gated,
)
from haemolynx.preprocessing.thick_vessel_tree import (
    ComponentVoxels,
    component_tree,
    shortest_path_parents,
)
from haemolynx.preprocessing.thick_vessels import (
    _OFFSETS_26,
    _build_dijkstra_graph_from_coords,
    _component_tree,
    _coords_of,
    _join_thin_arms_to_fat_ridge,
    _keys_of,
    _principal_endpoints,
    _relative_spacing,
    _step_factor,
)
from test_thick_vessel_skeletonisation import (
    _NETWORK_CASES,
    _capsule_into,
    _fat_and_thin_network,
    _network_case,
    plasma_labelled_cross,
    plasma_labelled_object,
    plasma_labelled_y,
)

_FULL = generate_binary_structure(3, 3)


def _digest(volume) -> str:
    volume = np.asarray(volume, dtype=bool)
    h = hashlib.sha256(repr(volume.shape).encode())
    h.update(np.packbits(volume.reshape(-1)).tobytes())
    return h.hexdigest()[:16]


def _random_network(seed, shape=(50, 130, 130), tubes=20, radius=(2.0, 6.0)):
    """Capsules chained end to end at random: one fused fat network."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    extent = np.asarray(shape, float)
    points = [rng.uniform(0.2, 0.8, 3) * extent]
    for _ in range(tubes):
        a = points[int(rng.integers(len(points)))]
        b = np.clip(a + rng.normal(size=3) * extent * 0.25, 0, extent - 1)
        _capsule_into(mask, (1.0, 1.0, 1.0), a, b, rng.uniform(*radius))
        points.append(b)
    labelled, _n = label(mask, structure=_FULL)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(np.argmax(sizes))


def _component(mask, spacing=(1.0, 1.0, 1.0)):
    """``(keys, edt_at, relative)`` of a one-component mask, the radius map as
    ``skeletonize_edt_ridge`` measures it."""
    relative = _relative_spacing(spacing)
    keys = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    edt = distance_transform_edt(mask, sampling=tuple(relative)).reshape(-1)[keys]
    return keys, edt, relative


def _in_ram_tree(mask, keys, edt, relative):
    coords = _coords_of(keys, mask.shape)
    return np.sort(_keys_of(_component_tree(coords, edt, mask.shape, relative), mask.shape))


def _brush(teeth: int, *, spacing=2, reach=16, trunk_radius=3.0, tooth_radius=2.5, side=40):
    """A trunk along x with *teeth* fat side branches every *spacing* voxels,
    turning a quarter each time: one component with one arm per tooth.
    Returns ``(mask, tips)``."""
    length = teeth * spacing + 20
    shape = (side, side, length + 10)
    c = side // 2
    zz, yy = np.indices(shape[:2])
    mask = np.zeros(shape, dtype=bool)
    mask[:, :, 5:length + 5] = (((zz - c) ** 2 + (yy - c) ** 2) <= trunk_radius**2)[:, :, None]
    disc = [(a, b) for a in range(-3, 4) for b in range(-3, 4) if a * a + b * b <= tooth_radius**2]
    turns = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    tips = []
    for k in range(teeth):
        x0 = 15 + spacing * k
        dz, dy = turns[k % 4]
        for t in range(reach + 1):
            z, y = c + dz * t, c + dy * t
            for a, b in disc:
                if dz:
                    mask[z, y + a, x0 + b] = True
                else:
                    mask[z + a, y, x0 + b] = True
        tips.append((c + dz * reach, c + dy * reach, x0))
    return mask, np.array(tips)


# --- the voxels' neighbours and the shortest paths ---


@pytest.mark.parametrize("shape", [(7, 9, 11), (1, 6, 8), (5, 1, 7), (4, 6, 1), (2, 2, 2)])
@pytest.mark.parametrize("fill", [0.2, 0.6, 1.0])
def test_each_voxels_neighbours_are_the_26_round_it(shape, fill):
    """Looked up three to a row, a row's x-1 at x = 0 wrapping onto the row
    before must not be taken for a neighbour, nor stop the row's x and x+1
    being found."""
    rng = np.random.default_rng(int(fill * 10) + sum(shape))
    volume = rng.random(shape) < fill
    keys = np.flatnonzero(volume.reshape(-1)).astype(np.int64)
    voxels = ComponentVoxels(keys, shape)
    found = voxels.neighbours(np.arange(len(keys)))
    for i, (z, y, x) in enumerate(_coords_of(keys, shape)):
        for j, (dz, dy, dx) in enumerate(_OFFSETS_26):
            q = (z + dz, y + dy, x + dx)
            there = all(0 <= a < s for a, s in zip(q, shape)) and volume[q]
            expected = int(np.searchsorted(keys, np.ravel_multi_index(q, shape))) if there else -1
            assert found[i, j] == expected, (i, (z, y, x), (dz, dy, dx))


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (1.5, 0.7, 0.7)])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_search_finds_scipys_shortest_distances(spacing, seed, monkeypatch):
    """The same distances as scipy's Dijkstra over the CSR graph, bit for bit
    -- each a sum of the same steps in the same order -- and every voxel's
    parent a step its distance came by. Small batches and frontiers, so
    buckets, bands and batches are all crossed."""
    monkeypatch.setattr(tree_module, "_MIN_FRONTIER", 64)
    mask = _random_network(seed, shape=(40, 90, 90), tubes=12)
    keys, edt, relative = _component(mask, spacing)
    coords = _coords_of(keys, mask.shape)
    root = int(np.argmax(edt))
    cost = 1.0 / (np.square(edt) + 1e-6)
    graph, _fg, _index = _build_dijkstra_graph_from_coords(coords, mask.shape, cost, relative)
    expected = dijkstra(graph, directed=True, indices=root)

    parent, distance = shortest_path_parents(
        ComponentVoxels(keys, mask.shape), edt, root, relative, batch=500, return_distances=True
    )

    assert np.array_equal(distance, expected)
    assert parent[root] == root and np.all(parent >= 0)
    others = np.flatnonzero(np.arange(len(keys)) != root)
    step = coords[others] - coords[parent[others]]
    factor = np.array([_step_factor(tuple(s), relative) for s in step])
    assert np.all(np.abs(step).max(axis=1) == 1)
    assert np.array_equal(distance[parent[others]] + cost[others] * factor, distance[others])


# --- the tree ---


def _fixture_components():
    cases = {
        "y": (plasma_labelled_y(8.0)[0], (1.0, 1.0, 1.0)),
        "cross": (plasma_labelled_cross(8.0)[0], (1.0, 1.0, 1.0)),
        "flat_trunk": (plasma_labelled_object(8.0)[0], (1.0, 1.0, 1.0)),
        "random": (_random_network(3), (1.0, 1.0, 1.0)),
    }
    for name in ("anisotropic", "cube"):
        spacing, shape, fat_radius, _extra = _NETWORK_CASES[name]
        _mask, fat = _fat_and_thin_network(sorted(_NETWORK_CASES).index(name), shape, spacing, fat_radius)
        labelled, _n = label(fat, structure=_FULL)
        cases[name] = (labelled == 1, spacing)
    return cases


@pytest.mark.parametrize("case", sorted(_fixture_components()))
def test_a_component_small_enough_for_the_exact_walk_gets_the_in_ram_tree(case):
    mask, spacing = _fixture_components()[case]
    keys, edt, relative = _component(mask, spacing)

    assert np.array_equal(component_tree(keys, edt, mask.shape, relative), _in_ram_tree(mask, keys, edt, relative))


@pytest.mark.parametrize("case", ["y", "cross", "random", "anisotropic", "cube"])
def test_the_bucketed_tree_lies_on_the_in_ram_one(case):
    """Past the exact walk's size the tree may take another route of exactly
    the same cost, and an arm judged on a route a voxel aside can come out a
    few voxels longer or shorter: on these fixtures at least 97% of either
    tree lies within two voxels of the other (on "random", 98.2%: one piece
    of 6 voxels more), and their sizes agree within 5%. (Not on a flattened
    trunk, whose ridge is a band of near-equal cost -- that is the case
    below.)"""
    mask, spacing = _fixture_components()[case]
    keys, edt, relative = _component(mask, spacing)
    plain = _coords_of(_in_ram_tree(mask, keys, edt, relative), mask.shape)
    bucketed = _coords_of(component_tree(keys, edt, mask.shape, relative, exact_max_voxels=0), mask.shape)

    for a, b in ((plain, bucketed), (bucketed, plain)):
        d, _ = cKDTree(b).query(a, p=np.inf)
        assert np.mean(d <= 2) >= 0.97, np.mean(d <= 2)
    assert abs(len(bucketed) - len(plain)) <= 0.05 * len(plain)


def test_on_a_flattened_trunk_the_bucketed_tree_still_runs_end_to_end():
    """A plasma column flattened five to one: its ridge is a band 64 voxels
    across where every route costs about the same, and the two searches pick
    different ones (about half the voxels more than two apart). Each is a
    tree of one piece reaching both ends of the trunk."""
    mask, _fat = plasma_labelled_object(8.0)
    keys, edt, relative = _component(mask)
    root, far = _principal_endpoints(_coords_of(keys, mask.shape), relative)
    tree = np.zeros(mask.shape, dtype=bool)
    tree.reshape(-1)[component_tree(keys, edt, mask.shape, relative, exact_max_voxels=0)] = True

    assert label(tree, structure=_FULL)[1] == 1 and euler_number(tree, connectivity=3) == 1
    assert tree[root] and tree[far]


@pytest.mark.parametrize("exact_max_voxels", [None, 0])
def test_a_component_with_600_arms_gets_every_one(exact_max_voxels, caplog):
    """_component_tree stopped at 500 arms (and reached 304 of these 600
    teeth); the large-stack tree has no cap. One piece, no loop, every tooth's
    tip within two of the component's widest radii of it."""
    mask, tips = _brush(600)
    keys, edt, relative = _component(mask)
    kwargs = {} if exact_max_voxels is None else {"exact_max_voxels": exact_max_voxels}
    with caplog.at_level(logging.WARNING):
        tree_keys = component_tree(keys, edt, mask.shape, relative, **kwargs)
    tree = np.zeros(mask.shape, dtype=bool)
    tree.reshape(-1)[tree_keys] = True

    assert label(tree, structure=_FULL)[1] == 1
    assert euler_number(tree, connectivity=3) == 1
    d, _ = cKDTree(np.argwhere(tree)).query(tips)
    assert np.all(d <= 2.0 * edt.max()), (int((d > 2.0 * edt.max()).sum()), d.max())
    assert "arm cap" not in caplog.text


def test_the_bucketed_tree_holds_tens_of_bytes_a_voxel(monkeypatch):
    """Traced, the bucketed tree's peak beyond its inputs stays under 80
    bytes a voxel of the component -- the exact walk's CSR graph alone is
    about 360. Batches small, so the per-voxel arrays are what is measured."""
    monkeypatch.setattr(tree_module, "_SEARCH_BATCH", 4096)
    monkeypatch.setattr(tree_module, "_CHUNK", 1 << 15)
    mask = _random_network(5, shape=(60, 240, 240), tubes=80, radius=(3.0, 8.0))
    keys, edt, relative = _component(mask)
    coords = _coords_of(keys, mask.shape)
    assert len(keys) > 300_000

    tracemalloc.start()
    component_tree(keys, edt, mask.shape, relative, exact_max_voxels=0)
    _now, bucketed = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    _build_dijkstra_graph_from_coords(coords, mask.shape, 1.0 / (np.square(edt) + 1e-6), relative)
    _now, csr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert bucketed / len(keys) < 80, bucketed / len(keys)
    assert csr / len(keys) > 300


# --- the setting, end to end ---

#: The in-RAM thickness-gated skeleton ("gated-") and EDT-ridge tree
#: ("ridge-") of the fixtures, as 4c96e9c5 drew them (outputs/lrt/digests.py,
#: 2026-10-19). The in-RAM path must not move: a change here is a change to
#: every run without the low-RAM option.
_IN_RAM_DIGESTS = {
    "gated-anisotropic": "6086aea587c577c9",
    "ridge-anisotropic": "98e92614262822d9",
    "gated-bridge_cap": "27bfb78e23014dad",
    "ridge-bridge_cap": "d5b5c3c81e793509",
    "gated-cube": "10154e123718c1ee",
    "ridge-cube": "c7629b436a365f8d",
    "gated-restricted": "afa75b997177bae3",
    "ridge-restricted": "1300f9e6ae69a46a",
    "gated-short_arms": "873b811026c101f2",
    "ridge-short_arms": "ce6882e1f13a4923",
    "gated-object": "f6903b0c364d3805",
    "ridge-y": "908f302a98e0fc5e",
    "ridge-cross": "5873e3d93c570434",
}


@pytest.mark.parametrize("large_stack_tree", [False, True])
def test_the_in_ram_skeleton_and_tree_are_what_they_were(large_stack_tree):
    """Without the low-RAM option the setting does nothing."""
    found = {}
    for name in sorted(_NETWORK_CASES):
        mask, kwargs = _network_case(name)
        skeleton, thick = skeletonize_thickness_gated(mask, large_stack_tree=large_stack_tree, **kwargs)
        found[f"gated-{name}"] = _digest(skeleton)
        found[f"ridge-{name}"] = _digest(
            skeletonize_edt_ridge(thick, voxel_size_zyx=kwargs["voxel_size_zyx"], large_stack_tree=large_stack_tree)
        )
    mask, _fat = plasma_labelled_object(8.0)
    found["gated-object"] = _digest(
        skeletonize_thickness_gated(
            mask, min_radius_um=THICK_VESSEL_MIN_RADIUS_UM, large_stack_tree=large_stack_tree
        )
    )
    for name, (mask, _ends) in (("y", plasma_labelled_y(8.0)), ("cross", plasma_labelled_cross(8.0))):
        found[f"ridge-{name}"] = _digest(skeletonize_edt_ridge(mask, large_stack_tree=large_stack_tree))

    assert found == _IN_RAM_DIGESTS


@pytest.mark.parametrize("case", sorted(_NETWORK_CASES))
def test_with_the_setting_a_small_network_keeps_its_low_ram_skeleton(tmp_path, monkeypatch, case):
    """Every fat component here fits the exact walk and has well under 500
    arms, and every join a corridor or the bounded box can make: the same
    skeleton as the low-RAM option without the setting, so as in RAM."""
    from haemolynx.preprocessing import memmap_support

    mask, kwargs = _network_case(case)
    plain, plain_thick = skeletonize_thickness_gated(mask, **kwargs)
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 30_000)
    low_ram, low_ram_thick = skeletonize_thickness_gated(
        mask, use_memmap=True, memmap_directory=tmp_path, large_stack_tree=True, **kwargs
    )

    assert np.array_equal(low_ram_thick, plain_thick)
    assert np.array_equal(low_ram, plain)
    assert _digest(plain) == _IN_RAM_DIGESTS[f"gated-{case}"]


def test_with_the_setting_a_large_component_takes_the_bucketed_walk(tmp_path, monkeypatch):
    """Over the exact walk's size, the skeleton still holds a connected tree
    through the fat region: every fat component's tree in one piece, with
    no loop of its own."""
    monkeypatch.setattr(tree_module, "EXACT_TREE_MAX_VOXELS", 0)
    walks = []
    real = tree_module.shortest_path_parents
    monkeypatch.setattr(
        tree_module, "shortest_path_parents", lambda *a, **k: walks.append(1) or real(*a, **k)
    )
    mask, kwargs = _network_case("cube")
    _skeleton, thick = skeletonize_thickness_gated(mask, **kwargs)
    ridge = skeletonize_edt_ridge(
        thick, voxel_size_zyx=kwargs["voxel_size_zyx"], use_memmap=True, large_stack_tree=True
    )

    assert walks
    labelled, n = label(thick, structure=_FULL)
    for component in range(1, n + 1):
        piece = ridge & (labelled == component)
        assert label(piece, structure=_FULL)[1] == 1
        assert euler_number(piece, connectivity=3) == 1


# --- the thin-arm join's bounded last resort ---


def _horseshoe():
    """The low-RAM last-resort test's horseshoe: an arm across the gap from
    the ridge, joined only the long way round."""
    import math

    shape = (3, 200, 200)
    yy, xx = np.indices(shape[1:])
    dist2 = (yy - 100) ** 2 + (xx - 100) ** 2
    angle = np.degrees(np.arctan2(yy - 100, xx - 100))
    allowed = np.zeros(shape, dtype=bool)
    allowed[1] = (dist2 <= 80**2) & (dist2 >= 70**2) & ~((angle > -6) & (angle < 6))
    allowed[1, 98:103, 98:103] = True
    thick = np.zeros(shape, dtype=bool)
    thick[1] = allowed[1] & (angle < -8) & (angle > -60) & (dist2 >= 70**2)
    skeleton = np.zeros(shape, dtype=bool)
    for lo, hi in ((-58.0, -9.0), (9.0, 40.0)):
        for degrees in np.arange(lo, hi, 0.25):
            rad = math.radians(degrees)
            skeleton[1, int(round(100 + 75 * math.sin(rad))), int(round(100 + 75 * math.cos(rad)))] = True
    return skeleton, thick, allowed


def test_the_bounded_last_resort_draws_the_unbounded_bridge_where_it_fits(monkeypatch):
    """The horseshoe's structure fits the budgets once the box has grown to
    it: the same bridge, voxel for voxel, from a graph never larger than the
    budget's voxels."""
    from haemolynx.preprocessing import memmap_support

    skeleton, thick, allowed = _horseshoe()
    plain = _join_thin_arms_to_fat_ridge(
        skeleton, thick, allowed, voxel_size_zyx=(1.0, 1.0, 1.0), min_arm_extent_um=4.0
    )
    assert (plain & ~skeleton).sum() > 100, "the bridge goes the long way round"

    sizes = []
    real = thick_vessels_module._build_dijkstra_graph_from_coords
    monkeypatch.setattr(
        thick_vessels_module,
        "_build_dijkstra_graph_from_coords",
        lambda coords, *a, **k: sizes.append(len(coords)) or real(coords, *a, **k),
    )
    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 5_000)
    bounded = _join_thin_arms_to_fat_ridge(
        skeleton, thick, allowed, voxel_size_zyx=(1.0, 1.0, 1.0), min_arm_extent_um=4.0,
        use_memmap=True, bounded_fallback=True,
    )

    assert np.array_equal(bounded, plain)
    assert sizes and max(sizes) <= thick_vessels_module._BOUNDED_FALLBACK_MAX_VOXELS


@pytest.mark.parametrize("budget", ["box", "voxels"])
def test_an_arm_past_the_bounded_search_is_left_unjoined_and_logged(monkeypatch, caplog, budget):
    """Budgets smaller than the horseshoe: no bridge, a warning naming the
    arm, and neither budget passed by what was labelled or searched."""
    skeleton, thick, allowed = _horseshoe()
    box_budget, voxel_budget = (10_000, 2_000_000) if budget == "box" else (64_000_000, 1_000)
    monkeypatch.setattr(thick_vessels_module, "_BOUNDED_FALLBACK_MAX_BOX_VOXELS", box_budget)
    monkeypatch.setattr(thick_vessels_module, "_BOUNDED_FALLBACK_MAX_VOXELS", voxel_budget)
    labelled, searched = [], []
    real_label = thick_vessels_module.label
    real_build = thick_vessels_module._build_dijkstra_graph_from_coords
    real_bounded = thick_vessels_module._bounded_structure_path

    def bounded(*args, **kwargs):
        monkeypatch.setattr(
            thick_vessels_module, "label", lambda v, **k: labelled.append(np.size(v)) or real_label(v, **k)
        )
        monkeypatch.setattr(
            thick_vessels_module,
            "_build_dijkstra_graph_from_coords",
            lambda coords, *a, **k: searched.append(len(coords)) or real_build(coords, *a, **k),
        )
        try:
            return real_bounded(*args, **kwargs)
        finally:
            monkeypatch.setattr(thick_vessels_module, "label", real_label)
            monkeypatch.setattr(thick_vessels_module, "_build_dijkstra_graph_from_coords", real_build)

    monkeypatch.setattr(thick_vessels_module, "_bounded_structure_path", bounded)
    with caplog.at_level(logging.INFO, logger=thick_vessels_module.logger.name):
        joined = _join_thin_arms_to_fat_ridge(
            skeleton, thick, allowed, voxel_size_zyx=(1.0, 1.0, 1.0), min_arm_extent_um=4.0,
            use_memmap=True, bounded_fallback=True,
        )

    assert np.array_equal(joined, skeleton), "no bridge is drawn"
    assert "1 thin arm(s) left unjoined by the bounded" in caplog.text
    assert labelled and max(labelled) <= box_budget
    assert max(searched, default=0) <= voxel_budget


def test_without_the_setting_the_low_ram_last_resort_is_unbounded(monkeypatch):
    """The low-RAM option alone still walks the whole structure."""
    skeleton, thick, allowed = _horseshoe()
    monkeypatch.setattr(thick_vessels_module, "_BOUNDED_FALLBACK_MAX_BOX_VOXELS", 1)
    called = []
    monkeypatch.setattr(
        thick_vessels_module, "_bounded_structure_path", lambda *a, **k: called.append(1) or ([], "")
    )
    joined = _join_thin_arms_to_fat_ridge(
        skeleton, thick, allowed, voxel_size_zyx=(1.0, 1.0, 1.0), min_arm_extent_um=4.0, use_memmap=True
    )

    assert not called
    assert (joined & ~skeleton).sum() > 100


# --- the pipeline reads the setting ---


def _settings(tmp_path, **overrides):
    from haemolynx.pipeline import default_schema, resolve_settings

    schema = default_schema()
    values = schema.defaults()
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "use_thick_vessel_skeletonisation": True,
            **overrides,
        }
    )
    return resolve_settings(values, schema=schema, config_path=None)


@pytest.mark.parametrize(
    "low_ram, setting, expected", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_the_setting_applies_under_the_low_ram_option_only(tmp_path, monkeypatch, low_ram, setting, expected):
    """Skeletonisation, graph building and post processing draw the tree the
    same way: each asks stages.large_stack_thick_tree."""
    from haemolynx import graph
    from haemolynx.pipeline import stages

    settings = _settings(
        tmp_path, use_memmap_loading=low_ram, skeleton_thick_vessel_large_stack_tree=setting,
        memmap_directory=tmp_path / "mm",
    )
    assert stages.large_stack_thick_tree(settings) is expected

    seen = {}

    def skeletonise(binary, **kwargs):
        seen["skeletonise"] = kwargs["large_stack_tree"]
        return np.zeros(np.shape(binary), dtype=bool), None

    monkeypatch.setattr(stages.preprocessing, "skeletonize_thickness_gated", skeletonise)
    stages._skeletonize_loaded_mask(np.ones((4, 4, 4), np.uint8), settings, (1.0, 1.0, 1.0))
    assert seen["skeletonise"] is expected

    def centreline(thick, voxel_size_zyx, *, use_memmap=False, large_stack_tree=False):
        seen["centreline"] = large_stack_tree
        return np.zeros(np.shape(thick), dtype=bool)

    monkeypatch.setattr(graph.thick_vessel_junctions, "thick_vessel_centreline", centreline)
    monkeypatch.setattr(graph, "insert_thick_vessel_junction_nodes", lambda G, *a, **k: G)
    G = nx.MultiGraph()
    G.add_edge(0, 1, **{graph.post_processing.EDITED: True})
    stages._bridge_edited_vessels(settings, G, None, np.ones((4, 4, 4), dtype=bool))
    assert seen["centreline"] is expected
