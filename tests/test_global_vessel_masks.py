"""Global vessel masks: one mask entered in a role's arteriole/global row and
no venule mask beside it.

Such a mask holds every vessel of its size, arteriole or venule, without
saying which. So the run cannot tell inlets from outlets by it, cut at it or
count branch orders from it. It can still say how large each vessel is, and
every vessel is classed Unknown_Large, Unknown_Small or Unknown_Cap instead.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx import graph
from haemolynx.graph.boundaries import (
    assigns_large_vessel_branch_orders,
    inlets_outlets_from_vessel_masks,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics.constriction import is_capillary_branch_order
from haemolynx.haemodynamics.poiseuille import (
    UNKNOWN_VESSEL_CLASS_LABELS,
    build_diameter_by_branch_order,
)
from haemolynx.io import (
    as_global_vessel_mask,
    clear_vessel_mask_cache,
    global_vessel_mask_roles,
    load_and_validate_vessel_masks,
    vessel_mask_role_is_global,
    vessel_masks_are_global,
)
from haemolynx.parsers import check_settings
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.checks import (
    check_global_vessel_masks,
    check_large_vessel_cut_when_masks_enabled,
)
from haemolynx.pipeline.stages import (
    SkeletonisedVolume,
    VesselNetwork,
    _assign_branch_orders,
    assign_boundaries,
)
from haemolynx.statistics.bifurcation import branch_order_category

VOXEL_SIZE_ZYX = (1.0, 1.0, 1.0)
SHAPE = (12, 12, 30)


@pytest.fixture(autouse=True)
def _forget_loaded_masks():
    clear_vessel_mask_cache()
    yield
    clear_vessel_mask_cache()


def _settings(**overrides):
    settings = default_schema().defaults()
    settings.update(
        {
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "large_arteriole_mask_path": Path("large_global.tif"),
            "large_venule_mask_path": None,
        }
    )
    settings.update(overrides)
    return settings


def _write(path: Path, volume: np.ndarray) -> Path:
    tifffile.imwrite(path, volume.astype(np.uint8))
    return path


# --- which runs are global ---------------------------------------------------


def test_a_role_with_no_venule_mask_is_global():
    assert vessel_mask_role_is_global(_settings(), "large")
    assert vessel_masks_are_global(_settings())
    assert global_vessel_mask_roles(_settings()) == ("large",)


def test_a_role_with_both_masks_is_not_global():
    settings = _settings(large_venule_mask_path=Path("large_venule.tif"))
    assert not vessel_mask_role_is_global(settings, "large")
    assert not vessel_masks_are_global(settings)


def test_a_mapping_that_never_names_the_venule_mask_reads_as_a_pair():
    """A run's settings name every setting; a hand-written mapping without the
    venule one means the masks as they were before a global one existed."""
    partial = {"automated_vessel_assignment": True, "use_large_vessel_masks": True}
    assert not vessel_masks_are_global(partial)
    assert vessel_masks_are_global({**partial, "large_venule_mask_path": None})


def test_masks_out_of_use_are_never_global():
    """The switch ticked under an unticked automated_vessel_assignment, as the
    panel leaves it, loads nothing, so it classes nothing either."""
    assert not vessel_masks_are_global(_settings(automated_vessel_assignment=False))
    assert not vessel_masks_are_global(_settings(use_large_vessel_masks=False))


def test_the_small_role_on_its_own_can_be_global():
    settings = _settings(
        use_large_vessel_masks=False,
        use_small_vessel_masks_for_boundary_assignment=True,
        small_venule_mask_path=None,
    )
    assert global_vessel_mask_roles(settings) == ("small",)


def test_under_ilastik_the_venule_classifier_decides():
    """The venule mask path is not read under ilastik; the classifier is."""
    settings = _settings(
        use_ilastik_large_vessel_segmentation=True,
        large_venule_mask_path=Path("large_venule.tif"),
        ilastik_venule_classifier_path=None,
    )
    assert vessel_mask_role_is_global(settings, "large")
    settings["ilastik_venule_classifier_path"] = Path("venule.ilp")
    assert not vessel_mask_role_is_global(settings, "large")


def test_the_venule_paths_are_unset_by_default():
    schema = default_schema()
    for name in (
        "large_venule_mask_path",
        "small_venule_mask_path",
        "ilastik_venule_classifier_path",
        "ilastik_small_venule_classifier_path",
    ):
        assert schema[name].default is None
        assert schema[name].optional_path


def test_global_masks_choose_no_inlets_and_no_large_tier():
    settings = _settings(
        inlets_outlets_from_vessel_masks=True,
        assign_large_vessel_branch_orders=True,
        cut_network_at_large_vessel_volumes=False,
    )
    assert not inlets_outlets_from_vessel_masks(settings)
    assert not assigns_large_vessel_branch_orders(settings)
    settings["large_venule_mask_path"] = Path("large_venule.tif")
    assert inlets_outlets_from_vessel_masks(settings)
    assert assigns_large_vessel_branch_orders(settings)


def test_the_panel_names_the_arteriole_rows_arteriole_or_global():
    from haemolynx.gui.form import label_for

    assert label_for("large_arteriole_mask_path") == "Large arteriole/global mask path"
    assert label_for("small_arteriole_mask_path") == "Small arteriole/global mask path"
    assert "global" in default_schema()["large_venule_mask_path"].placeholder


def test_as_global_vessel_mask_joins_a_pair():
    arteriole = np.zeros(SHAPE, dtype=bool)
    venule = np.zeros(SHAPE, dtype=bool)
    arteriole[0, 0, 0] = venule[1, 1, 1] = True
    joined = as_global_vessel_mask(arteriole, venule)
    assert joined[0, 0, 0] and joined[1, 1, 1] and int(joined.sum()) == 2
    assert as_global_vessel_mask(arteriole, None) is not None
    assert as_global_vessel_mask(None, None) is None


# --- loading -------------------------------------------------------------------


def test_a_global_mask_loads_cleaned_with_no_venule_mask(tmp_path):
    """Dilation and the component-volume filter apply; the pair-only cleanups
    have no second class to read and are skipped, not failed."""
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[5, 5, 5:20] = 1  # a 15-voxel vessel
    volume[10, 10, 28] = 1  # a 1-voxel speck, a few voxels once dilated
    path = _write(tmp_path / "large_global.tif", volume)

    mask, venule, voxel_size, venule_voxel_size = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=path,
        venule_mask_path=None,
        image_shape=SHAPE,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
        min_component_volume_um3=20.0,
        dilation_microns=1.0,
        swap_minority_components=True,
        remove_small_opposite_attached_components=True,
        exclude_smaller_overlapping_volumes=True,
    )

    assert venule is None and venule_voxel_size is None
    assert voxel_size == (1.0, 1.0, 1.0)
    assert mask.dtype == bool and mask.shape == SHAPE
    assert not mask[10, 10, 28], "the speck below the volume threshold is removed"
    assert mask[5, 5, 10] and mask[5, 6, 10], "dilated by a micron"


def test_a_global_mask_is_loaded_once_and_handed_out_fresh(tmp_path, monkeypatch):
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[5, 5, 5:20] = 1
    path = _write(tmp_path / "large_global.tif", volume)
    # Old enough that the cache trusts its mtime and size.
    old = time.time() - 60
    os.utime(path, (old, old))
    arguments = dict(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=path,
        venule_mask_path=None,
        image_shape=SHAPE,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    first, *_ = load_and_validate_vessel_masks(**arguments)
    first[:] = False

    import haemolynx.io.automated_vessel_assignment as loader

    monkeypatch.setattr(
        loader, "_load_mask_image", lambda *a, **k: pytest.fail("loaded the file again")
    )
    second, venule, *_ = load_and_validate_vessel_masks(**arguments)
    assert venule is None
    assert int(second.sum()) == 15, "a caller's copy changed what the next one got"


def test_a_global_mask_of_the_wrong_shape_is_refused(tmp_path):
    path = _write(tmp_path / "large_global.tif", np.zeros((4, 5, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="large_global_mask shape does not match"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=path,
            venule_mask_path=None,
            image_shape=SHAPE,
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


def test_ilastik_with_no_venule_classifier_segments_one_global_mask(tmp_path, monkeypatch):
    import haemolynx.io.automated_vessel_assignment as loader

    calls = []

    def _segment(*, input_image_path, classifier_path, output_path, **_kwargs):
        calls.append((Path(classifier_path).name, Path(output_path).name))
        volume = np.zeros(SHAPE, dtype=np.uint8)
        volume[5, 5, 5:20] = 1
        return _write(Path(output_path), volume)

    monkeypatch.setattr(loader, "run_ilastik_headless_segmentation", _segment)
    raw = _write(tmp_path / "raw.tif", np.zeros(SHAPE, dtype=np.uint8))

    mask, venule, *_ = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=True,
        arteriole_mask_path=None,
        venule_mask_path=None,
        ilastik_unsegmented_arteriole_path=raw,
        ilastik_unsegmented_venule_path=None,
        ilastik_arteriole_classifier_path=tmp_path / "vessels.ilp",
        ilastik_venule_classifier_path=None,
        ilastik_output_dir=tmp_path,
        image_shape=SHAPE,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert calls == [("vessels.ilp", "raw_large_global_segmented.tif")]
    assert venule is None and int(mask.sum()) == 15


# --- tagging and classes -----------------------------------------------------


def _chain(xs, *, bridge_between=None) -> nx.MultiGraph:
    """Nodes along x at (z, y) = (5, 5), one edge between each pair."""
    G = nx.MultiGraph()
    for node, x in enumerate(xs):
        G.add_node(node, pos=(5.0, 5.0, float(x)))
    for node in range(len(xs) - 1):
        start, end = xs[node], xs[node + 1]
        voxels = [(5.0, 5.0, float(x)) for x in range(start, end + 1)]
        G.add_edge(node, node + 1, voxels=voxels, length=float(end - start))
        if bridge_between == node:
            G.edges[node, node + 1, 0][IS_ZERO_RESISTANCE] = True
    return G


def _classes(G):
    return [G.edges[u, v, k].get(graph.GLOBAL_MASK_CLASS) for u, v, k in sorted(G.edges(keys=True))]


def test_vessels_are_tagged_by_the_global_mask_they_lie_in():
    G = _chain([0, 6, 12, 18, 28])
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, 0:8] = True
    small = np.zeros(SHAPE, dtype=bool)
    small[4:7, 4:7, 12:20] = True

    counts = graph.tag_vessels_in_global_masks(
        G, large_mask=large, small_mask=small, voxel_size_zyx=VOXEL_SIZE_ZYX
    )

    # x 6-12 is 2/7 large and 1/7 small, and joins a large edge to a small
    # one, so no gap fill: a capillary between the two.
    assert _classes(G) == ["large", None, "small", None]
    assert counts["large_edge_count"] == 1 and counts["small_edge_count"] == 1
    assert counts["capillary_edge_count"] == 2


def test_an_edge_in_both_masks_goes_to_the_one_covering_more():
    G = _chain([0, 10])
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, 0:7] = True  # 7 of 11 samples
    small = np.zeros(SHAPE, dtype=bool)
    small[4:7, 4:7, 0:11] = True  # all 11
    graph.tag_vessels_in_global_masks(
        G, large_mask=large, small_mask=small, voxel_size_zyx=VOXEL_SIZE_ZYX
    )
    assert _classes(G) == ["small"]


def test_an_under_covered_edge_inside_one_large_vessel_keeps_its_class():
    """A short gap in the mask is not the vessel ending twice."""
    G = _chain([0, 8, 16, 24])
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, 0:25] = True
    large[4:7, 4:7, 10:16] = False  # x 8-16: 3 of 9 samples in the mask

    counts = graph.tag_vessels_in_global_masks(
        G, large_mask=large, voxel_size_zyx=VOXEL_SIZE_ZYX
    )

    assert _classes(G) == ["large", "large", "large"]
    assert counts["gap_filled_edge_count"] == 1


def test_a_bridge_is_never_tagged():
    G = _chain([0, 6, 12], bridge_between=1)
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, :] = True
    graph.tag_vessels_in_global_masks(G, large_mask=large, voxel_size_zyx=VOXEL_SIZE_ZYX)
    assert _classes(G) == ["large", None]


def test_tagging_only_some_edges_leaves_the_others_as_they_were():
    G = _chain([0, 6, 12])
    G.edges[0, 1, 0][graph.GLOBAL_MASK_CLASS] = "small"
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, :] = True
    graph.tag_vessels_in_global_masks(
        G, large_mask=large, voxel_size_zyx=VOXEL_SIZE_ZYX, edges=[(2, 1, 0)]
    )
    assert _classes(G) == ["small", "large"]


def test_unknown_classes_replace_branch_orders():
    G = _chain([0, 6, 12, 18, 24], bridge_between=3)
    G.edges[0, 1, 0][graph.GLOBAL_MASK_CLASS] = "large"
    G.edges[1, 2, 0][graph.GLOBAL_MASK_CLASS] = "small"
    G.edges[2, 3, 0]["branch_order"] = "B03"  # left by an earlier run
    G.edges[3, 4, 0]["branch_order"] = "B04"

    summary = graph.assign_unknown_vessel_classes(G)

    orders = [G.edges[u, v, k].get("branch_order") for u, v, k in sorted(G.edges(keys=True))]
    assert orders == ["Unknown_Large", "Unknown_Small", "Unknown_Cap", None]
    assert summary["mode"] == "unknown" and summary["bridges"] == 1
    assert summary["counts"] == {"Unknown_Large": 1, "Unknown_Small": 1, "Unknown_Cap": 1}


def test_the_unknown_classes_are_spelled_alike_everywhere():
    assert UNKNOWN_VESSEL_CLASS_LABELS == graph.UNKNOWN_VESSEL_CLASSES
    for label in graph.UNKNOWN_VESSEL_CLASSES:
        # The label-prefix tests read B, Art and Ven as tiers.
        assert not label.startswith(("B", "Art", "Ven", "Large_"))


@pytest.mark.parametrize("all_diams_const", [False, True])
def test_the_diameter_table_has_an_entry_for_each_unknown_class(all_diams_const):
    table = build_diameter_by_branch_order(
        all_diams_const=all_diams_const,
        default_diameter=4.0,
        manual_capillary_diameter_by_branch_order={"B01": 6.0},
    )
    expected_small = 4.0 if all_diams_const else 6.0
    assert table["Unknown_Large"] == 4.0
    assert table["Unknown_Small"] == expected_small
    assert table["Unknown_Cap"] == 4.0


def test_an_unknown_capillary_is_a_capillary_and_the_others_are_classed_by_size():
    """Large and small vessels are on neither side of the bed, and are
    compared apart by every breakdown by vessel type."""
    assert is_capillary_branch_order("Unknown_Cap")
    assert not is_capillary_branch_order("Unknown_Large")
    assert branch_order_category("Unknown_Cap") == "capillary"
    assert branch_order_category("Unknown_Large") == "large_vessel"
    assert branch_order_category("Unknown_Small") == "small_vessel"


# --- preflight -----------------------------------------------------------------


def test_preflight_asks_for_no_venule_mask(tmp_path):
    arteriole = _write(tmp_path / "large_global.tif", np.zeros(SHAPE, dtype=np.uint8))
    settings = _settings(large_arteriole_mask_path=arteriole)
    report = check_settings(default_schema(), settings)
    assert not [error for error in report.errors if "venule" in error]


def test_preflight_says_what_global_masks_do_and_warns_of_unread_settings():
    report = check_global_vessel_masks(_settings())
    assert any("Unknown_Large" in detail for _label, detail in report.passed)
    assert not report.warnings

    report = check_global_vessel_masks(
        _settings(
            use_small_vessel_masks_for_boundary_assignment=True,
            small_venule_mask_path=Path("small_venule.tif"),
            strict_branch_order_assignment=True,
        )
    )
    assert any("read as one" in warning and "small-vessel" in warning for warning in report.warnings)
    assert any("strict_branch_order_assignment" in warning for warning in report.warnings)


def test_preflight_does_not_ask_global_masks_for_the_cut():
    report = check_large_vessel_cut_when_masks_enabled(
        _settings(cut_network_at_large_vessel_volumes=False)
    )
    assert not report.warnings


def test_preflight_is_silent_without_global_masks():
    report = check_global_vessel_masks(_settings(large_venule_mask_path=Path("v.tif")))
    assert not report.passed and not report.warnings


# --- the Boundaries and Diameters stages ---------------------------------------


def _network(G, large=None, small=None, tmp_path=None) -> VesselNetwork:
    image = np.zeros(SHAPE, dtype=np.uint8)
    volume = SkeletonisedVolume(
        image=image,
        skeleton=image.copy(),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        output_dir=tmp_path,
    )
    return VesselNetwork(
        graph=G, volume=volume, large_arteriole_mask=large, small_arteriole_mask=small
    )


def _stage_settings(tmp_path, **overrides):
    return _settings(
        plot_dir=tmp_path,
        visualize_results=False,
        boundary_axis=2,
        inlet_nodes=[],
        outlet_nodes=[],
        arteriole_boundary_nodes=[],
        venule_boundary_nodes=[],
        large_vessel_inlet_nodes=[],
        large_vessel_outlet_nodes=[],
        large_arteriole_boundary_nodes=[],
        large_venule_boundary_nodes=[],
        **overrides,
    )


def test_assign_boundaries_with_a_global_mask_cuts_nothing_and_tags_vessels(
    tmp_path, monkeypatch
):
    G = _chain([0, 6, 12, 18, 29])
    large = np.zeros(SHAPE, dtype=bool)
    large[4:7, 4:7, 0:8] = True
    small = np.zeros(SHAPE, dtype=bool)
    small[4:7, 4:7, 12:20] = True
    network = _network(G, large, small, tmp_path)
    settings = _stage_settings(
        tmp_path,
        use_small_vessel_masks_for_boundary_assignment=True,
        small_venule_mask_path=None,
        cut_network_at_large_vessel_volumes=True,
        inlets_outlets_from_vessel_masks=True,
    )
    for name in (
        "cut_graph_at_large_vessel_volumes",
        "select_terminal_nodes_from_large_vessel_masks_progressive_dilation",
        "infer_boundary_nodes_from_small_vessel_masks_progressive_dilation",
    ):
        monkeypatch.setattr(
            f"haemolynx.graph.{name}",
            lambda *a, _name=name, **k: pytest.fail(f"{_name} reads arterioles and venules apart"),
        )

    boundaries = assign_boundaries(settings, network)

    assert boundaries.graph.number_of_edges() == 4, "nothing was cut"
    # edge_percent along x: the two ends of the chain.
    assert boundaries.inlet_nodes == [0] and boundaries.outlet_nodes == [4]
    assert boundaries.arteriole_boundary_nodes == []
    assert _classes(boundaries.graph) == ["large", None, "small", None]

    _assign_branch_orders(settings, boundaries.graph)

    orders = [
        boundaries.graph.edges[u, v, k]["branch_order"]
        for u, v, k in sorted(boundaries.graph.edges(keys=True))
    ]
    assert orders == ["Unknown_Large", "Unknown_Cap", "Unknown_Small", "Unknown_Cap"]


FIXTURE = Path(__file__).resolve().parent / "data" / "seven_vessel_noisy_3d.tif"


@pytest.mark.slow
@pytest.mark.integration
def test_a_run_with_one_global_large_mask_classes_every_vessel(tmp_path, caplog):
    """The whole pipeline, with one large mask and no venule mask: it runs to
    the end, the thick-vessel skeletonisation is restricted to that mask, and
    every vessel comes out of unknown class, the solve included."""
    import logging

    from scipy import ndimage

    from haemolynx.io.load import _to_binary_volume_for_skeletonization
    from haemolynx.pipeline import resolve_settings, run_pipeline_stages

    binary = np.asarray(_to_binary_volume_for_skeletonization(tifffile.imread(FIXTURE)), dtype=bool)
    core = ndimage.distance_transform_edt(binary) >= 2
    labels, _count = ndimage.label(core)
    thickest = labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    global_mask = ndimage.binary_dilation(thickest, iterations=3) & binary
    mask_path = _write(tmp_path / "large_global.tif", global_mask)

    schema = default_schema()
    values = {setting.name: setting.default for setting in schema}
    values.update(
        {
            "input_path": FIXTURE,
            "vtk_output_prefix": tmp_path / "run",
            "plot_dir": tmp_path / "plots",
            "statistics": False,
            "show_plots_in_ide": False,
            "interactive_plots": False,
            "automated_vessel_assignment": True,
            "use_large_vessel_masks": True,
            "large_arteriole_mask_path": mask_path,
            "large_venule_mask_path": None,
            "use_thick_vessel_skeletonisation": True,
            "skeleton_thick_vessel_restrict_to_mask": "large",
        }
    )
    settings = resolve_settings(values, schema=schema, config_path=None)

    with caplog.at_level(logging.INFO):
        G = run_pipeline_stages(settings, schema)

    vessels = [data for _u, _v, data in G.edges(data=True) if not data.get(IS_ZERO_RESISTANCE)]
    orders = {data.get("branch_order") for data in vessels}
    assert orders <= set(graph.UNKNOWN_VESSEL_CLASSES), orders
    assert {"Unknown_Large", "Unknown_Cap"} <= orders
    assert all(data.get("diameter_um", 0) > 0 for data in vessels)
    assert any(
        data.get("flow_state") == "solved" and float(data.get("flow_abs") or 0.0) > 0
        for data in vessels
    ), "nothing solved"
    assert "large_arteriole_mask" in caplog.text, "the thick-vessel restriction read no mask"
    assert "Global vessel masks" in caplog.text


def test_a_pair_beside_a_global_mask_is_read_as_one_mask(tmp_path):
    """Its arteriole and venule are not told apart: one global mask makes every
    vessel's class unknown."""
    G = _chain([0, 6, 12, 29])
    arteriole = np.zeros(SHAPE, dtype=bool)
    arteriole[4:7, 4:7, 0:7] = True
    venule = np.zeros(SHAPE, dtype=bool)
    venule[4:7, 4:7, 6:13] = True
    small = np.zeros(SHAPE, dtype=bool)
    network = _network(G, arteriole, small, tmp_path)
    network.large_venule_mask = venule
    settings = _stage_settings(
        tmp_path,
        large_venule_mask_path=Path("large_venule.tif"),
        use_small_vessel_masks_for_boundary_assignment=True,
        small_venule_mask_path=None,
        inlets_outlets_from_vessel_masks=False,
    )

    boundaries = assign_boundaries(settings, network)

    assert _classes(boundaries.graph) == ["large", "large", None]
