"""What ``build_network()`` does when graph building leaves no vessels.

Every later stage needs a vessel, and each failed on something else when there
was none: the 3D drawing on missing node positions, boundary assignment on
missing terminals, while the 2D drawing went through and saved the empty
graph. So the stage stops there and says which step took the last vessel.

Once it has the network it also warns when the run's dense solvers would not
fit in the free memory.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage

from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.stages import SkeletonisedVolume, build_network

SCHEMA = default_schema()


def _settings(tmp_path: Path, **overrides) -> dict:
    values = SCHEMA.defaults()
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            **overrides,
        }
    )
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def _cross(size: int, start: int, stop: int) -> np.ndarray:
    """Two straight skeleton lines crossing at the volume's centre: four arms."""
    skeleton = np.zeros((size, size, size), dtype=bool)
    centre = size // 2
    skeleton[start:stop, centre, centre] = True
    skeleton[centre, centre, start:stop] = True
    return skeleton


def _mask_under(skeleton: np.ndarray) -> np.ndarray:
    return ndimage.binary_dilation(skeleton, iterations=2).astype(np.uint8)


def _volume(tmp_path: Path, skeleton: np.ndarray, image: np.ndarray) -> SkeletonisedVolume:
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    return SkeletonisedVolume(
        image=image,
        skeleton=skeleton,
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path / "out",
    )


def _graph_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "input_graph.pkl"


#: 3 um arms on 1 um voxels, all shorter than the default 5 um
#: cluster_collapse_distance: the four vessels collapse into one point.
SHORT_CROSS = _cross(10, 2, 8)
#: 15 um arms: longer than the collapse distance and the stub prune's reach.
LONG_CROSS = _cross(40, 5, 35)


@pytest.mark.parametrize(
    "render",
    [
        {"final_render_mode": "3d"},
        {"final_render_mode": "2d"},
        {"visualize_results": False},
    ],
    ids=["3d drawing", "2d drawing", "no drawing"],
)
def test_a_skeleton_shorter_than_the_collapse_distance_stops_the_stage_and_says_why(
    tmp_path, render
):
    settings = _settings(tmp_path, **render)
    volume = _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS))

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    message = str(raised.value)
    assert "built 4 vessel(s), and the last went at the collapse_node_clusters step" in message
    assert "cluster_collapse_distance (5.0 um)" in message
    # The image is not the cause here, and the message says so.
    assert "11 of those 11 voxels lie in the image's foreground" in message
    # Stopped before anything was saved or drawn from the empty graph.
    assert not _graph_path(tmp_path).exists()
    assert list((tmp_path / "plots").glob("final_graph*")) == []


def test_a_run_with_a_thick_vessel_region_stops_with_the_same_reason(tmp_path):
    """The thick-vessel split, and the clean-up after it, run on whatever graph
    building left before the check: an empty build passes through them and is
    still stopped with the step that took its last vessel."""
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS))
    volume.thick_vessel_mask = _mask_under(SHORT_CROSS).astype(bool)

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    assert "the last went at the collapse_node_clusters step" in str(raised.value)
    assert not _graph_path(tmp_path).exists()


def test_an_image_that_does_not_cover_the_skeleton_is_named_as_the_cause(tmp_path):
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, LONG_CROSS, np.zeros(LONG_CROSS.shape, dtype=np.uint8))

    with pytest.raises(ValueError, match="Graph building left no vessels") as raised:
        build_network(settings, volume, SCHEMA)

    message = str(raised.value)
    assert "the last went at the prune_vascular_stubs step" in message
    assert "0 of those 59 voxels lie in the image's foreground" in message
    assert "must be the segmentation the skeleton was made from" in message


def test_the_same_cross_with_longer_arms_builds_its_four_vessels(tmp_path):
    """The check only fires on no vessels: the cross above, in a mask, with
    arms long enough to keep."""
    settings = _settings(tmp_path)
    network = build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)

    assert network.graph.number_of_edges() == 4
    assert _graph_path(tmp_path).exists()
    assert (tmp_path / "plots" / "final_graph_3d.html").exists()


@pytest.mark.parametrize("run_haemodynamics", [True, False])
def test_the_built_network_warns_when_its_dense_solve_will_not_fit(
    tmp_path, monkeypatch, caplog, run_haemodynamics
):
    """As soon as the node count is known, so a run can be stopped before its
    diameters are measured -- and only when the run will solve."""
    from haemolynx.haemodynamics import resistance

    monkeypatch.setattr(resistance, "available_memory_bytes", lambda: 1000)
    settings = _settings(tmp_path, run_haemodynamics=run_haemodynamics, visualize_results=False)
    with caplog.at_level(logging.WARNING, logger="haemolynx.haemodynamics.resistance"):
        network = build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)

    nodes = f"{network.graph.number_of_nodes():,} nodes"
    memory = [r.getMessage() for r in caplog.records if "of memory free now" in r.getMessage()]
    if run_haemodynamics:
        assert len(memory) == 2 and all(nodes in message for message in memory)
    else:
        assert memory == []


def _spy_on_graph_building(monkeypatch) -> dict:
    """The keyword arguments graph building, the lumen clean-up and the
    thick-vessel split are called with, by name, in call order."""
    from haemolynx import graph

    seen: dict = {}

    def spy(name, real):
        def wrapper(*args, **kwargs):
            seen.setdefault(name, []).append(kwargs)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(graph, "build_graph_from_skeleton", spy("build", graph.build_graph_from_skeleton))
    monkeypatch.setattr(graph.assemble, "lumen_cleanup", spy("cleanup", graph.assemble.lumen_cleanup))
    monkeypatch.setattr(
        graph, "insert_thick_vessel_junction_nodes", spy("split", graph.insert_thick_vessel_junction_nodes)
    )
    return seen


def test_a_thick_vessel_regions_centreline_reaches_every_clean_up_and_the_split(tmp_path, monkeypatch):
    """Every clean-up keeps the strand on the thick vessels' centreline tree
    of two in one lumen (judged by the wall alone it broke them, E14.5,
    2026-10-15): graph building's own, the one after smoothing and the one
    after the split are all handed it, and the split the same tree, found
    once."""
    seen = _spy_on_graph_building(monkeypatch)
    settings = _settings(tmp_path)
    volume = _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS))
    volume.thick_vessel_mask = _mask_under(LONG_CROSS).astype(bool)

    build_network(settings, volume, SCHEMA)

    along = seen["build"][0]["along_thick_centreline"]
    assert along is not None
    assert len(seen["cleanup"]) == 2  # graph building's, then build_network's
    assert all(call["along_thick_centreline"] is along for call in seen["cleanup"])
    arm = np.array([[20.0, 20.0, float(x)] for x in range(8, 33)])
    assert along(arm) > 20.0
    (split,) = seen["split"]
    # The tree as its voxels (skeletonisation's, or drawn here): a volume
    # of zeros the region's shape is not made to hold it.
    tree = split["centreline"]
    assert tree.ndim == 2 and tree.shape[1] == 3 and len(tree)
    from haemolynx.graph import thick_vessel_junctions

    assert np.array_equal(
        tree, np.argwhere(thick_vessel_junctions.thick_vessel_centreline(volume.thick_vessel_mask, (1.0, 1.0, 1.0)))
    )


def test_the_tree_skeletonisation_drew_is_used_not_drawn_again(tmp_path, monkeypatch):
    """``SkeletonisedVolume.thick_vessel_centreline``: the split and the
    clean-ups get skeletonisation's tree as it is, and nothing draws it."""
    from haemolynx.graph import thick_vessel_junctions

    seen = _spy_on_graph_building(monkeypatch)
    volume = _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS))
    volume.thick_vessel_mask = _mask_under(LONG_CROSS).astype(bool)
    drawn = thick_vessel_junctions.thick_vessel_centreline_voxels(volume.thick_vessel_mask, (1.0, 1.0, 1.0))
    volume.thick_vessel_centreline = drawn

    def refuse(*_args, **_kwargs):
        raise AssertionError("the tree was drawn again")

    monkeypatch.setattr(thick_vessel_junctions, "thick_vessel_centreline_voxels", refuse)
    monkeypatch.setattr(thick_vessel_junctions, "thick_vessel_centreline", refuse)
    build_network(_settings(tmp_path), volume, SCHEMA)

    (split,) = seen["split"]
    assert split["centreline"] is drawn


def _thick_tee(size: int = 48):
    """A thick vessel 5 um in radius along x, a thin one 1.5 um in radius
    leaving it along y to the image's edge, their skeletons, and the thick
    region (the segmentation within 3 um of the thick vessel)."""
    p = np.stack(np.indices((size, size, size)), axis=-1).astype(float)
    centre = size // 2
    thick_tube = np.hypot(p[..., 0] - centre, p[..., 1] - centre) <= 5.0
    thin_tube = (np.hypot(p[..., 0] - centre, p[..., 2] - centre) <= 1.5) & (p[..., 1] >= centre)
    image = thick_tube | thin_tube
    skeleton = np.zeros(image.shape, dtype=bool)
    skeleton[centre, centre, 2 : size - 2] = True
    skeleton[centre, centre:, centre] = True
    region = image & (ndimage.distance_transform_edt(~thick_tube) <= 3.0)
    return skeleton, image.astype(np.uint8), region


def test_the_built_network_logs_the_shape_of_its_thick_vessel_bridges(tmp_path, caplog):
    """graph.diagnose_thick_vessel_bridges, with the graph checks: the one
    vessel opening into the thick one has one bridge, from the region's wall
    to the centreline."""
    skeleton, image, region = _thick_tee()
    volume = _volume(tmp_path, skeleton, image)
    volume.thick_vessel_mask = region
    with caplog.at_level(logging.INFO, logger="haemolynx.pipeline.stages"):
        network = build_network(_settings(tmp_path, visualize_results=False), volume, SCHEMA)

    from haemolynx.graph import IS_ZERO_RESISTANCE

    bridges = [d for *_e, d in network.graph.edges(data=True) if d.get(IS_ZERO_RESISTANCE)]
    (line,) = [r for r in caplog.records if r.getMessage().startswith("Thick-vessel bridges: ")
               and r.name == "haemolynx.pipeline.stages"]
    assert len(bridges) == 1
    assert line.getMessage().startswith("Thick-vessel bridges: 1 bridge(s): 1 ")
    assert line.levelno == logging.INFO, "a bridge from the wall to the centreline is what the split means to make"


def test_the_clean_ups_judge_the_strand_on_the_thick_centreline_by_the_runs_tolerance(tmp_path):
    """``thick_bridge_centreline_tolerance_um`` reaches what every clean-up
    keeps the strand on the thick centreline by, as it reaches the split: a
    strand 3 um off the thick vessel's axis follows its centreline at a 4 um
    tolerance, not at the default."""
    from haemolynx.pipeline.stages import network_context

    _skeleton, image, region = _thick_tee()
    strand = np.array([[24.0, 27.0, float(x)] for x in range(8, 40)])
    lengths = {}
    for tolerance in (1.5, 4.0):
        settings = _settings(tmp_path, thick_bridge_centreline_tolerance_um=tolerance)
        context = network_context(settings, image.astype(bool), (1.0, 1.0, 1.0), region)
        lengths[tolerance] = context.along_thick_centreline(strand)
    assert lengths[1.5] == 0.0
    assert lengths[4.0] > 25.0


def test_marking_dead_ends_records_the_thick_centreline_the_wall_is_judged_on(tmp_path):
    """``mark_dead_end_kinds`` keeps the thick vessels' centreline tree on the
    graph it marks (PLAN_bridging.md 1.3), so the dead end stopping at the
    thick vessel's wall is ``thick_wall`` there and to every later reader of
    the graph -- post processing's re-judging, the review lists -- without
    the mask's thick region at hand."""
    from dead_end_fixtures import SHAPE as FIXTURE_SHAPE
    from dead_end_fixtures import THICK_AXIS, THICK_RADIUS, at_a_thick_wall
    from dead_end_fixtures import VOXEL_SIZE as FIXTURE_VOXELS
    from lumen_artefact_fixtures import capsule

    from haemolynx.graph import dead_ends as de
    from haemolynx.pipeline.stages import mark_dead_end_kinds, network_context

    case = at_a_thick_wall(recorded=False)
    body = capsule(np.zeros(FIXTURE_SHAPE, dtype=bool), (10, 7, -5), (10, 7, 85), THICK_RADIUS)
    region = case.mask & (ndimage.distance_transform_edt(~body) <= 3.0)
    settings = _settings(tmp_path, thick_bridge_centreline_tolerance_um=1.2)

    mark_dead_end_kinds(settings, case.graph, network_context(settings, case.mask, FIXTURE_VOXELS, region))

    record = case.graph.graph[de.THICK_CENTRELINE]
    points = record["points_um"]
    # The tree the split bridges to, inside the thick vessel along its length.
    off_axis = np.hypot(points[:, 0] - THICK_AXIS[0][0], points[:, 1] - THICK_AXIS[0][1])
    assert np.all(off_axis <= THICK_RADIUS) and np.ptp(points[:, 2]) > 60.0
    assert len(record["radius_um"]) == len(points)
    assert np.all(np.isfinite(record["radius_um"]) & (record["radius_um"] <= THICK_RADIUS + 1.0))
    assert record["tolerance_um"] == 1.2
    assert case.graph.nodes[case.tip][de.DEAD_END_KIND] == de.THICK_WALL

    # A run with no thick region keeps no record, a stale one included.
    mark_dead_end_kinds(settings, case.graph, network_context(settings, case.mask, FIXTURE_VOXELS))
    assert de.THICK_CENTRELINE not in case.graph.graph
    assert case.graph.nodes[case.tip][de.DEAD_END_KIND] == de.UNKNOWN


def test_the_built_network_carries_its_thick_centreline(tmp_path):
    from haemolynx.graph import dead_ends as de

    skeleton, image, region = _thick_tee()
    volume = _volume(tmp_path, skeleton, image)
    volume.thick_vessel_mask = region
    network = build_network(_settings(tmp_path, visualize_results=False), volume, SCHEMA)

    points = network.graph.graph[de.THICK_CENTRELINE]["points_um"]
    assert len(points) > 30 and np.all(np.hypot(points[:, 0] - 24.0, points[:, 1] - 24.0) <= 5.0)

    plain = build_network(_settings(tmp_path), _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert de.THICK_CENTRELINE not in plain.graph.graph


def test_a_run_without_a_thick_vessel_region_leaves_the_choice_to_the_wall(tmp_path, monkeypatch):
    seen = _spy_on_graph_building(monkeypatch)
    build_network(_settings(tmp_path), _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert seen["build"][0]["along_thick_centreline"] is None
    assert all(call["along_thick_centreline"] is None for call in seen["cleanup"])
    assert "split" not in seen


def test_an_empty_build_leaves_the_graph_a_run_saved_before(tmp_path):
    (tmp_path / "out").mkdir()
    saved = nx.MultiGraph()
    saved.add_edge(0, 1)
    _graph_path(tmp_path).write_bytes(pickle.dumps(saved))
    before = _graph_path(tmp_path).read_bytes()

    with pytest.raises(ValueError, match="Graph building left no vessels"):
        build_network(
            _settings(tmp_path),
            _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS)),
            SCHEMA,
        )

    assert _graph_path(tmp_path).read_bytes() == before


def test_a_saved_graph_with_no_vessels_is_refused_when_loaded(tmp_path):
    (tmp_path / "out").mkdir()
    _graph_path(tmp_path).write_bytes(pickle.dumps(nx.MultiGraph()))

    with pytest.raises(ValueError, match="has no vessels") as raised:
        build_network(
            _settings(tmp_path, do_graph_building=False),
            _volume(tmp_path, SHORT_CROSS, _mask_under(SHORT_CROSS)),
            SCHEMA,
        )

    assert str(_graph_path(tmp_path)) in str(raised.value)
    assert "do_graph_building" in str(raised.value)


def _spy_on_retrace(monkeypatch) -> dict:
    """The skeleton re-trace's calls, and the edges smoothing is asked to
    smooth afterwards; the re-trace marks the first vessel as drawn back."""
    from haemolynx import graph

    seen: dict = {"retrace": [], "smoothed": []}

    def retrace(G, skeleton, support, **kwargs):
        seen["retrace"].append(dict(support=support, **kwargs))
        u, v, key = next(iter(G.edges(keys=True)))
        G.edges[u, v, key][graph.skeleton_retrace.RETRACED] = True
        return G

    real_smooth = graph.smooth_graph_centrelines

    def smooth(G, skeleton, **kwargs):
        if kwargs.get("edges") is not None:
            seen["smoothed"].append(list(kwargs["edges"]))
        return real_smooth(G, skeleton, **kwargs)

    monkeypatch.setattr(graph.skeleton_retrace, "retrace_lost_skeleton", retrace)
    monkeypatch.setattr(graph, "smooth_graph_centrelines", smooth)
    return seen


def test_the_skeleton_re_trace_runs_by_default_and_not_when_turned_off(tmp_path, monkeypatch):
    """On by default since Harvey's grid N2 (2026-10-18: 19 of the 21 vessels it
    drew on E14.5 kept)."""
    seen = _spy_on_retrace(monkeypatch)
    assert SCHEMA.defaults()["retrace_lost_skeleton"] is True
    build_network(_settings(tmp_path), _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert len(seen["retrace"]) == 1

    seen["retrace"].clear()
    off = tmp_path / "off"
    off.mkdir()
    build_network(_settings(off, retrace_lost_skeleton=False), _volume(off, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert seen["retrace"] == []


def test_the_skeleton_re_trace_reads_the_clean_ups_mask_and_the_short_loop_rules_hole(tmp_path, monkeypatch):
    """After the last clean-up, so no rule takes what it draws again; judged
    by the clean-up's own mask, at the short-loop rule's hole; and what it
    draws is smoothed like every other vessel."""
    from haemolynx.preprocessing.bridge_mask_support import MaskSupport

    seen = _spy_on_retrace(monkeypatch)
    settings = _settings(tmp_path, retrace_lost_skeleton=True, short_loop_keep_hole_over_um=12.5)
    network = build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)

    (call,) = seen["retrace"]
    assert isinstance(call["support"], MaskSupport)
    assert call["support"].max_background_gap_um == settings["bridge_max_background_gap_um"]
    assert call["keep_hole_over_um"] == 12.5
    (smoothed,) = seen["smoothed"]
    assert len(smoothed) == 1
    u, v, key = smoothed[0]
    assert network.graph.edges[u, v, key]["centreline_smoothing"] in {"smoothed", "relaxed", "kept_raw", "too_short"}


def test_the_skeleton_re_trace_needs_mask_supported_bridging(tmp_path, monkeypatch):
    seen = _spy_on_retrace(monkeypatch)
    settings = _settings(tmp_path, retrace_lost_skeleton=True, bridge_require_mask_support=False)
    build_network(settings, _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS)), SCHEMA)
    assert seen["retrace"] == []


def test_one_mask_support_serves_every_step_that_reads_the_mask(tmp_path, monkeypatch):
    """Graph building, both clean-ups, the thick-vessel split and the dead-end
    marks are handed one ``MaskSupport``: the mask is packed to bits once and
    one surface tree is built a run. Each used to make its own -- up to four
    packings, and a second tree for the split's join at the walls -- which on
    a large disk-backed stack is a pass over the volume each (2026-10-18). The
    split is handed the tree's radius ``ThickCentreline`` measured, too. In
    low-RAM mode, where the mask is a memmap and the in-RAM graph checks
    (which read the image again) are skipped."""
    import importlib

    from haemolynx.pipeline import stages

    bridge_mask_support = importlib.import_module("haemolynx.preprocessing.bridge_mask_support")
    pointwise_distance = importlib.import_module("haemolynx.preprocessing.pointwise_distance")

    seen = _spy_on_graph_building(monkeypatch)
    marked = []
    real_marks = stages._mark_dead_end_kinds
    monkeypatch.setattr(
        stages, "_mark_dead_end_kinds", lambda *a, **k: marked.append(k.get("support")) or real_marks(*a, **k)
    )
    packs, trees = [], []
    real_pack = bridge_mask_support.PackedMask.from_mask.__func__
    monkeypatch.setattr(
        bridge_mask_support.PackedMask, "from_mask",
        classmethod(lambda cls, mask, **kw: packs.append(1) or real_pack(cls, mask, **kw)),
    )
    monkeypatch.setattr(bridge_mask_support, "PACK_AFTER_POINTS", 1)
    real_init = pointwise_distance.FeatureDistance.__init__

    def counting_init(self, *args, **kwargs):
        trees.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(pointwise_distance.FeatureDistance, "__init__", counting_init)
    volume = _volume(tmp_path, LONG_CROSS, _mask_under(LONG_CROSS))
    volume.thick_vessel_mask = _mask_under(LONG_CROSS).astype(bool)

    build_network(_settings(tmp_path, use_memmap_loading=True, memmap_directory=tmp_path / "mm"), volume, SCHEMA)

    support = seen["build"][0]["mask_support"]
    assert support is not None
    assert all(call["mask_support"] is support for call in seen["cleanup"])
    (split,) = seen["split"]
    assert split["mask_support"] is support
    assert split["tree_radius_um"] is not None and len(split["tree_radius_um"]) == len(split["centreline"])
    assert marked == [support]
    assert len(packs) == 1
    assert len(trees) == 1


def _cross_and_island() -> np.ndarray:
    """LONG_CROSS with a 25 um line of its own 12 um from it: farther than
    the orphan reconnect reaches, and not heading for it."""
    skeleton = LONG_CROSS.copy()
    skeleton[8, 5:31, 8] = True
    return skeleton


def test_pieces_without_a_junction_stay_by_default(tmp_path):
    skeleton = _cross_and_island()
    network = build_network(_settings(tmp_path), _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)

    G = network.graph
    assert G.number_of_edges() == 5
    assert "pieces_without_junction_removed" not in G.graph
    assert sum(1 for n in G if G.nodes[n].get("dead_end_kind") == "island_end") == 2


def test_pieces_without_a_junction_go_at_the_end_of_graph_building_when_asked(tmp_path):
    skeleton = _cross_and_island()
    settings = _settings(tmp_path, remove_pieces_without_junction=True)
    network = build_network(settings, _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)

    G = network.graph
    assert G.number_of_edges() == 4
    assert all(abs(G.nodes[n]["pos"][1] - 20) < 1e-6 for n in G)    # only the cross is left
    assert G.graph["pieces_without_junction_removed"] == 1
    assert G.graph["pieces_without_junction_removed_um"] == pytest.approx(25.0, abs=2.0)
    assert not any(G.nodes[n].get("dead_end_kind") == "island_end" for n in G)
    with _graph_path(tmp_path).open("rb") as f:
        assert pickle.load(f).number_of_edges() == 4    # the saved graph is the trimmed one


def test_a_build_left_only_with_pieces_without_a_junction_says_the_setting_took_them(tmp_path):
    skeleton = np.zeros((40, 40, 40), dtype=bool)
    skeleton[8, 5:31, 8] = True
    settings = _settings(tmp_path, remove_pieces_without_junction=True)
    with pytest.raises(ValueError, match="remove_pieces_without_junction") as caught:
        build_network(settings, _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)
    assert "removal of pieces with no junction step" in str(caught.value)


# --- remove_interior_dead_ends --------------------------------------------------


def _ring_and_spur() -> np.ndarray:
    """A square ring 30 um a side round a 26 um hole, and a 15 um spur off it
    ending inside the image, 34 um from the nearest face."""
    skeleton = np.zeros((20, 100, 100), dtype=bool)
    skeleton[10, 20, 20:51] = skeleton[10, 50, 20:51] = True
    skeleton[10, 20:51, 20] = skeleton[10, 20:51, 50] = True
    skeleton[10, 51:66, 35] = True
    return skeleton


def test_interior_dead_ends_stay_with_the_setting_off(tmp_path):
    skeleton = _ring_and_spur()
    settings = _settings(tmp_path, remove_interior_dead_ends=False)
    network = build_network(settings, _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)

    G = network.graph
    assert "interior_dead_ends_removed" not in G.graph
    assert sum(1 for n in G if G.degree(n) == 1) == 1


def test_interior_dead_ends_go_and_the_ring_stays_when_asked(tmp_path):
    skeleton = _ring_and_spur()
    settings = _settings(tmp_path, remove_interior_dead_ends=True)
    network = build_network(settings, _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)

    G = network.graph
    assert not any(G.degree(n) == 1 for n in G)
    assert G.graph["interior_dead_ends_removed"] == 1
    assert G.graph["interior_dead_ends_removed_um"] == pytest.approx(15.0, abs=3.0)
    ring = sum(float(d["length"]) for *_e, d in G.edges(data=True))
    assert ring == pytest.approx(120.0, rel=0.1)
    with _graph_path(tmp_path).open("rb") as f:
        assert pickle.load(f).graph["interior_dead_ends_removed"] == 1


def test_a_tree_with_no_vessel_leaving_the_image_goes_whole_and_the_error_says_why(tmp_path):
    cross = _cross(80, 25, 55)  # its arms end 25 um from the faces
    settings = _settings(tmp_path, remove_interior_dead_ends=True)
    with pytest.raises(ValueError, match="remove_interior_dead_ends"):
        build_network(settings, _volume(tmp_path, cross, _mask_under(cross)), SCHEMA)


def test_the_clean_up_after_the_recovery_reads_its_own_hole_limit(tmp_path, monkeypatch):
    import haemolynx.graph.assemble as assemble

    holes = []
    real = assemble.consolidate_lumen

    def spy(G, cleanup, **kwargs):
        holes.append(cleanup.short_loop_keep_hole_um)
        return real(G, cleanup, **kwargs)

    monkeypatch.setattr(assemble, "consolidate_lumen", spy)
    skeleton = _ring_and_spur()
    settings = _settings(tmp_path, remove_interior_dead_ends=True, recovery_short_loop_keep_hole_over_um=4.0)
    build_network(settings, _volume(tmp_path, skeleton, _mask_under(skeleton)), SCHEMA)

    assert holes[-1] == 4.0
    assert set(holes[:-1]) == {settings["short_loop_keep_hole_over_um"]}
    assert SCHEMA["recovery_short_loop_keep_hole_over_um"].default == 5.3
    assert SCHEMA["remove_interior_dead_ends"].default is False


def test_the_low_ram_option_removes_and_recovers_the_same(tmp_path):
    """The step reads the mask through the run's one MaskSupport, made low
    memory under use_memmap_loading: the same graph either way."""
    skeleton = _ring_and_spur()
    graphs = []
    for low in (False, True):
        run = tmp_path / ("low" if low else "ram")
        (run / "mm").mkdir(parents=True)
        settings = _settings(run, use_memmap_loading=low, memmap_directory=run / "mm", remove_interior_dead_ends=True)
        graphs.append(build_network(settings, _volume(run, skeleton, _mask_under(skeleton)), SCHEMA).graph)

    ram, low = graphs
    assert ram.graph["interior_dead_ends_removed"] == low.graph["interior_dead_ends_removed"] == 1
    assert sorted(ram.nodes) == sorted(low.nodes)
    assert sorted(ram.edges(keys=True)) == sorted(low.edges(keys=True))
    for edge in ram.edges(keys=True):
        assert np.array_equal(np.asarray(ram.edges[edge]["voxels"]), np.asarray(low.edges[edge]["voxels"]))
