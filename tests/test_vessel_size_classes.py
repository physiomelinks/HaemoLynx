"""The size convention: vessels classed by measured width against two stated
thresholds (``classify_vessels_by_size``, ``graph/vessel_size_classes.py``).

The rule and its names; the tagging, which rides on the global masks' tags so
every reader of the ``Unknown_*`` labels works unchanged; the Diameters
stage's order with the box on (measure, class, branch orders, then the class
medians and the table) and off; post processing, which re-classes edited
vessels by the thresholds the network was run with; preflight; and the
outputs that name the classes by their widths.
"""
from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph
from haemolynx.graph import (
    GLOBAL_MASK_CLASS,
    VESSEL_CLASSES_SOURCE,
    VESSEL_SIZE_CLASS,
    add_vessel_between,
    assign_unknown_vessel_classes,
    check_size_thresholds,
    clear_vessel_size_classes,
    size_class_range,
    size_class_title,
    size_convention_of,
    tag_vessels_by_size,
    vessel_classes_by_size,
    vessel_size_class,
)
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.graph.vessel_size_classes import VESSEL_SIZE_FROM
from haemolynx.haemodynamics.poiseuille import (
    DEFAULT_CLASS_MEDIAN_MIN_EDGES,
    build_diameter_by_branch_order,
    stamp_edge_diameters,
)
from haemolynx.parsers.schema import IneffectiveSettingWarning
from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.checks import check_vessel_size_classes
from haemolynx.pipeline.stages import (
    BoundaryNodes,
    SkeletonisedVolume,
    VesselNetwork,
    assign_diameters,
    post_process,
)
from haemolynx.statistics.vessel_classes import compute_vessel_class_statistics
from vessel_class_fixtures import CLASSES, NODE_X, tube_chain, tube_chain_mask

SCHEMA = default_schema()


# --- the rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    "width, expected",
    [
        (0.5, "narrow"),
        (5.999, "narrow"),
        (6.0, "mid"),  # each threshold is the least width of its class
        (14.999, "mid"),
        (15.0, "wide"),
        (120.0, "wide"),
    ],
)
def test_each_threshold_is_the_least_width_of_its_class(width, expected):
    assert vessel_size_class(width, 6.0, 15.0) == expected


@pytest.mark.parametrize("width", [None, float("nan"), float("inf"), 0.0, -3.0, "wide"])
def test_a_vessel_with_no_usable_width_has_no_class(width):
    assert vessel_size_class(width, 6.0, 15.0) is None


@pytest.mark.parametrize(
    "mid, wide, words",
    [
        (None, 15.0, "vessel_size_mid_min_diameter_um is not set"),
        (6.0, None, "vessel_size_wide_min_diameter_um is not set"),
        (15.0, 15.0, "must be below"),
        (16.0, 15.0, "must be below"),
        (-1.0, 15.0, "0 or more"),
        (6.0, float("nan"), "finite"),
        ("six", 15.0, "a number of microns"),
    ],
)
def test_thresholds_must_both_be_stated_with_mid_below_wide(mid, wide, words):
    with pytest.raises(ValueError, match=words):
        check_size_thresholds(mid, wide)


def test_thresholds_read_as_numbers():
    assert check_size_thresholds("6", 15) == (6.0, 15.0)


def test_classes_are_named_by_the_widths_they_hold():
    assert [size_class_range(name, 6.0, 15.0) for name in CLASSES] == [
        "at least 15 µm", "6-15 µm", "under 6 µm",
    ]
    assert [size_class_title(name, 8.4, 16.5) for name in CLASSES] == [
        "Vessels at least 16.5 µm wide", "Vessels 8.4-16.5 µm wide", "Vessels under 8.4 µm wide",
    ]
    with pytest.raises(ValueError):
        size_class_range("capillary", 6.0, 15.0)


def test_the_box_is_in_effect_only_without_masks_on_an_image():
    assert vessel_classes_by_size({"classify_vessels_by_size": True})
    assert not vessel_classes_by_size({"classify_vessels_by_size": False})
    assert not vessel_classes_by_size(
        {"classify_vessels_by_size": True, "automated_vessel_assignment": True}
    )
    assert not vessel_classes_by_size(
        {"classify_vessels_by_size": True, "network_source": "synthetic"}
    )


# --- tagging -------------------------------------------------------------------


def _star(widths: dict[tuple[int, int], float | None]) -> nx.MultiGraph:
    """Edges keyed (u, v) with each one's ``diameter_um`` (None: no width)."""
    G = nx.MultiGraph()
    for (u, v), width in widths.items():
        for node in (u, v):
            G.add_node(node, pos=(0.0, 0.0, float(node)))
        G.add_edge(u, v, **({} if width is None else {"diameter_um": width}))
    return G


def _attr(G, name):
    return {(u, v): data.get(name) for u, v, data in G.edges(data=True)}


def test_vessels_are_classed_by_width_on_the_global_masks_own_tags():
    G = _star({(0, 1): 20.0, (1, 2): 9.0, (2, 3): 4.0})

    tag_vessels_by_size(G, 6.0, 15.0)

    assert _attr(G, VESSEL_SIZE_CLASS) == {(0, 1): "wide", (1, 2): "mid", (2, 3): "narrow"}
    assert _attr(G, GLOBAL_MASK_CLASS) == {(0, 1): "large", (1, 2): "small", (2, 3): None}
    assert set(_attr(G, VESSEL_SIZE_FROM).values()) == {"width"}
    assign_unknown_vessel_classes(G)
    assert _attr(G, "branch_order") == {
        (0, 1): "Unknown_Large", (1, 2): "Unknown_Small", (2, 3): "Unknown_Cap",
    }


def test_a_bridge_is_never_classed():
    G = _star({(0, 1): 20.0, (1, 2): 20.0})
    G.edges[1, 2, 0].update({IS_ZERO_RESISTANCE: True, VESSEL_SIZE_CLASS: "wide", GLOBAL_MASK_CLASS: "large"})

    record = tag_vessels_by_size(G, 6.0, 15.0)

    bridge = G.edges[1, 2, 0]
    assert VESSEL_SIZE_CLASS not in bridge and GLOBAL_MASK_CLASS not in bridge
    assert record["counts"] == {"wide": 1, "mid": 0, "narrow": 0}


def test_a_vessel_with_no_width_takes_the_class_the_vessels_at_both_its_ends_agree_on():
    # 1-2 has no width between two wide vessels; 4-5 between a wide and a mid
    # one; 6-7 has a wide vessel at one end and nothing at the other.
    G = _star({
        (0, 1): 20.0, (1, 2): None, (2, 3): 18.0,
        (3, 4): 20.0, (4, 5): None, (5, 9): 8.0,
        (8, 6): 30.0, (6, 7): None,
    })

    record = tag_vessels_by_size(G, 6.0, 15.0)

    assert G.edges[1, 2, 0][VESSEL_SIZE_CLASS] == "wide"
    assert G.edges[1, 2, 0][VESSEL_SIZE_FROM] == "neighbours"
    assert G.edges[1, 2, 0][GLOBAL_MASK_CLASS] == "large"
    for u, v in ((4, 5), (6, 7)):
        assert G.edges[u, v, 0][VESSEL_SIZE_CLASS] == "narrow"
        assert G.edges[u, v, 0][VESSEL_SIZE_FROM] == "no_width"
        assert GLOBAL_MASK_CLASS not in G.edges[u, v, 0]
    assert record["unsized"] == {"neighbours": 1, "no_width": 2}


def test_a_vessel_waiting_for_a_fallback_has_no_width_whatever_it_still_carries():
    """A diameter left from an earlier stamp is not a measurement."""
    G = _star({(0, 1): 4.0, (1, 2): 20.0, (2, 3): 4.0})

    tag_vessels_by_size(G, 6.0, 15.0, unsized=[(2, 1, 0)])

    assert G.edges[1, 2, 0][VESSEL_SIZE_CLASS] == "narrow"
    assert G.edges[1, 2, 0][VESSEL_SIZE_FROM] == "neighbours"


def test_the_record_counts_classes_bases_and_the_vessels_with_no_width():
    G = _star({(0, 1): 20.0, (1, 2): 9.0, (2, 3): 9.0, (3, 4): None})
    G.edges[1, 2, 0]["diameter_basis"] = "endothelial"

    record = tag_vessels_by_size(G, 6.0, 15.0, default_basis="plasma_column")

    assert record == {
        "source": "size_convention",
        "mid_min_um": 6.0,
        "wide_min_um": 15.0,
        "counts": {"wide": 1, "mid": 2, "narrow": 1},
        "basis_counts": {"endothelial": 1, "plasma_column": 2},
        "unsized": {"neighbours": 0, "no_width": 1},
    }
    assert size_convention_of(G) == record
    assert G.graph[VESSEL_CLASSES_SOURCE] == record


def test_the_mask_width_is_on_the_runs_basis():
    G = _star({(0, 1): None})
    G.edges[0, 1, 0].update(edt_diameter_um=9.0, diameter_basis="endothelial")

    record = tag_vessels_by_size(
        G, 6.0, 15.0, width_attribute="edt_diameter_um", default_basis="plasma_column"
    )

    assert G.edges[0, 1, 0][VESSEL_SIZE_CLASS] == "mid"
    assert record["basis_counts"] == {"plasma_column": 1}


def test_classing_some_vessels_leaves_the_others_and_counts_them_all():
    G = _star({(0, 1): 20.0, (1, 2): 9.0})
    tag_vessels_by_size(G, 6.0, 15.0)
    G.edges[0, 1, 0]["diameter_um"] = 3.0  # not re-classed: not asked for
    G.add_edge(2, 3, diameter_um=4.0)

    record = tag_vessels_by_size(G, 6.0, 15.0, edges=[(3, 2, 0)])

    assert _attr(G, VESSEL_SIZE_CLASS) == {(0, 1): "wide", (1, 2): "mid", (2, 3): "narrow"}
    assert record["counts"] == {"wide": 1, "mid": 1, "narrow": 1}


def test_a_graph_without_the_convention_has_no_record():
    G = _star({(0, 1): 20.0})
    assert size_convention_of(G) is None
    G.graph[VESSEL_CLASSES_SOURCE] = {"source": "global_masks"}
    assert size_convention_of(G) is None


@pytest.mark.parametrize("keep_mask_class", [False, True])
def test_clearing_takes_the_convention_off(keep_mask_class):
    G = _star({(0, 1): 20.0, (1, 2): 4.0})
    tag_vessels_by_size(G, 6.0, 15.0)

    assert clear_vessel_size_classes(G, keep_mask_class=keep_mask_class)

    assert size_convention_of(G) is None
    assert set(_attr(G, VESSEL_SIZE_CLASS).values()) == {None}
    assert set(_attr(G, VESSEL_SIZE_FROM).values()) == {None}
    assert G.edges[0, 1, 0].get(GLOBAL_MASK_CLASS) == ("large" if keep_mask_class else None)
    assert not clear_vessel_size_classes(G, keep_mask_class=keep_mask_class)


# --- the diameter chain's hook ---------------------------------------------------


def _measured_chain() -> nx.MultiGraph:
    """Three FWHM-measured wide vessels and, between two of them, one FWHM
    failed on; a mid vessel off the end."""
    G = _star({(0, 1): None, (1, 2): None, (2, 3): None, (3, 4): None, (4, 5): None})
    for (u, v), width in {(0, 1): 20.0, (1, 2): 22.0, (3, 4): 24.0, (4, 5): 9.0}.items():
        G.edges[u, v, 0]["fwhm_diameter_um"] = width
    G.edges[2, 3, 0]["diameter_um"] = 99.0  # stale, from an earlier stamp
    return G


def test_the_hook_runs_once_between_the_measured_widths_and_the_fallbacks():
    G = _measured_chain()
    calls = []

    def hook(graph_seen, waiting):
        calls.append((graph_seen, sorted(waiting)))
        assert graph_seen.edges[0, 1, 0]["diameter_source"] == "measured"
        assert "diameter_source" not in graph_seen.edges[2, 3, 0], "fallbacks ran first"

    stamp_edge_diameters(G, {}, before_fallbacks=hook)

    assert len(calls) == 1 and calls[0][0] is G
    assert calls[0][1] == [(2, 3, 0)]


def test_the_class_median_follows_the_size_classes():
    """The unmeasured vessel between wide vessels is wide, so Unknown_Large,
    and takes the median of the wide vessels' widths -- not the table's."""
    G = _measured_chain()

    def classify(graph_seen, waiting):
        tag_vessels_by_size(graph_seen, 6.0, 15.0, unsized=waiting)
        assign_unknown_vessel_classes(graph_seen)

    table = build_diameter_by_branch_order(all_diams_const=False, default_diameter=4.0)
    counts = stamp_edge_diameters(
        G, table, class_median_min_edges=DEFAULT_CLASS_MEDIAN_MIN_EDGES, before_fallbacks=classify
    )

    unmeasured = G.edges[2, 3, 0]
    assert unmeasured["branch_order"] == "Unknown_Large"
    assert unmeasured["diameter_source"] == "class_median"
    assert unmeasured["diameter_um"] == 22.0
    assert counts["class_median"] == 1


def test_without_a_hook_the_stamp_is_what_it_was():
    with_hook, without = _measured_chain(), _measured_chain()
    for G in (with_hook, without):
        for _u, _v, data in G.edges(data=True):
            data["branch_order"] = "B01"
    table = build_diameter_by_branch_order(all_diams_const=False, default_diameter=4.0)

    stamp_edge_diameters(with_hook, table, class_median_min_edges=3, before_fallbacks=lambda *_: None)
    stamp_edge_diameters(without, table, class_median_min_edges=3)

    assert [d for *_e, d in with_hook.edges(data=True)] == [d for *_e, d in without.edges(data=True)]


# --- the Diameters stage on tubes of known width ---------------------------------


def _network(tmp_path, G=None, mask=None) -> VesselNetwork:
    image = tube_chain_mask() if mask is None else mask
    volume = SkeletonisedVolume(
        image=image,
        skeleton=np.zeros_like(image),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    return VesselNetwork(graph=tube_chain() if G is None else G, volume=volume)


def _settings(tmp_path, **overrides):
    values = {
        **SCHEMA.defaults(),
        "visualize_results": False,
        "automated_vessel_assignment": False,
        "classify_vessels_by_size": True,
        "vessel_size_mid_min_diameter_um": 6.0,
        "vessel_size_wide_min_diameter_um": 15.0,
        "use_edt_diameter_crosscheck": True,
        "measure_diameters_from_raw_image": False,
        **overrides,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IneffectiveSettingWarning)
        settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    # What the Boundaries stage would have left.
    settings.update(
        plot_dir=tmp_path,
        inlet_nodes=[0],
        outlet_nodes=[3],
        arteriole_boundary_nodes=[],
        venule_boundary_nodes=[],
        large_vessel_inlet_nodes=[],
        large_vessel_outlet_nodes=[],
        large_arteriole_boundary_nodes=[],
        large_venule_boundary_nodes=[],
    )
    return settings


def _diameters(settings, network):
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[3], graph=network.graph)
    return assign_diameters(settings, network, boundaries, SCHEMA), boundaries


def _trunk(G, name):
    return [G.edges[node, node + 1, 0].get(name) for node in range(len(NODE_X) - 1)]


def test_the_stage_measures_then_classes_then_orders_then_fills_in(tmp_path, caplog):
    network = _network(tmp_path)

    with caplog.at_level(logging.INFO):
        model, _boundaries = _diameters(_settings(tmp_path), network)

    G = model.graph
    assert _trunk(G, VESSEL_SIZE_CLASS) == list(CLASSES)
    assert _trunk(G, "branch_order") == ["Unknown_Large", "Unknown_Small", "Unknown_Cap"]
    assert _trunk(G, "diameter_source") == ["edt_mask"] * 3
    widths = _trunk(G, "diameter_um")
    assert widths[0] >= 15.0 > widths[1] >= 6.0 > widths[2], widths
    assert [vessel_size_class(w, 6.0, 15.0) for w in widths] == list(CLASSES)
    record = size_convention_of(G)
    assert record["counts"] == {"wide": 1, "mid": 1, "narrow": 1}
    assert record["basis_counts"] == {"plasma_column": 3}
    assert "Size convention: vessels classed by their measured width" in caplog.text
    assert "under 6 µm wide" in caplog.text


def test_a_vessel_the_mask_gives_no_width_is_filled_in_by_its_class(tmp_path):
    """The narrow piece's mask is taken away, so it has no width. A mid
    vessel at one end and nothing at the other do not agree on a class: it
    is narrow by default, Unknown_Cap, and takes the table's width."""
    mask = tube_chain_mask()
    mask[:, :, 60:] = 0
    model, _ = _diameters(_settings(tmp_path), _network(tmp_path, mask=mask))

    G = model.graph
    narrow = G.edges[2, 3, 0]
    assert narrow[VESSEL_SIZE_CLASS] == "narrow" and narrow[VESSEL_SIZE_FROM] == "no_width"
    assert narrow["branch_order"] == "Unknown_Cap"
    assert narrow["diameter_source"] == "table"
    assert size_convention_of(G)["unsized"] == {"neighbours": 0, "no_width": 1}


def test_without_haemodynamics_the_classes_come_from_the_masks_own_widths(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        model, _ = _diameters(
            _settings(tmp_path, run_haemodynamics=False, use_edt_diameter_crosscheck=False),
            _network(tmp_path),
        )

    G = model.graph
    assert _trunk(G, VESSEL_SIZE_CLASS) == list(CLASSES)
    assert _trunk(G, "branch_order") == ["Unknown_Large", "Unknown_Small", "Unknown_Cap"]
    assert _trunk(G, "diameter_um") == [None, None, None], "nothing is stamped"
    assert all(w is not None for w in _trunk(G, "edt_diameter_um"))
    assert "from the mask's own widths (EDT) alone" in caplog.text


def test_with_the_box_off_branch_orders_come_first_and_an_old_convention_goes(tmp_path):
    G = tube_chain()
    tag_vessels_by_size(G, 6.0, 15.0)  # as an earlier run with the box left it
    for _u, _v, data in G.edges(data=True):
        data["diameter_um"] = 9.0

    model, _ = _diameters(_settings(tmp_path, classify_vessels_by_size=False), _network(tmp_path, G))

    G = model.graph
    assert size_convention_of(G) is None
    assert _trunk(G, VESSEL_SIZE_CLASS) == [None, None, None]
    assert _trunk(G, GLOBAL_MASK_CLASS) == [None, None, None]
    assert _trunk(G, "branch_order") == ["B01", "B02", "B03"]
    assert _trunk(G, "diameter_source") == ["edt_mask"] * 3


# --- post processing -------------------------------------------------------------


def _draw_in_the_wide_tube(G) -> tuple:
    """A vessel drawn by hand from node 1 back along the wide piece."""
    new = max(G.nodes) + 1
    points = [(10.0, 10.0, float(x)) for x in range(30, 9, -1)]
    G.add_node(new, pos=np.asarray(points[-1]))
    return add_vessel_between(G, 1, new, points)


def test_post_processing_classes_an_edited_vessel_by_the_convention_it_was_run_with(tmp_path, caplog):
    network = _network(tmp_path)
    settings = _settings(tmp_path)
    model, boundaries = _diameters(settings, network)
    drawn = _draw_in_the_wide_tube(model.graph)
    # The settings now state another convention: the network keeps its own.
    later = {**settings, "vessel_size_mid_min_diameter_um": 20.0, "vessel_size_wide_min_diameter_um": 30.0}

    with caplog.at_level(logging.INFO):
        out = post_process(later, model, boundaries, SCHEMA, network=network)

    G = out.graph
    assert G.edges[drawn]["diameter_source"] == "edt_mask"
    assert G.edges[drawn]["diameter_um"] >= 15.0
    assert G.edges[drawn][VESSEL_SIZE_CLASS] == "wide"
    assert G.edges[drawn]["branch_order"] == "Unknown_Large"
    assert _trunk(G, VESSEL_SIZE_CLASS) == list(CLASSES)
    record = size_convention_of(G)
    assert (record["mid_min_um"], record["wide_min_um"]) == (6.0, 15.0)
    assert record["counts"] == {"wide": 2, "mid": 1, "narrow": 1}
    assert out.results["post_process"]["vessel_classes"] == record
    assert "classed by size at mid 6 um and wide 15 um" in caplog.text
    assert "Run again from Diameters" in caplog.text


def test_post_processing_does_not_warn_when_the_settings_agree(tmp_path, caplog):
    network = _network(tmp_path)
    settings = _settings(tmp_path)
    model, boundaries = _diameters(settings, network)
    _draw_in_the_wide_tube(model.graph)

    with caplog.at_level(logging.WARNING):
        post_process(settings, model, boundaries, SCHEMA, network=network)

    assert "classed by size at" not in caplog.text


def test_post_processing_without_haemodynamics_classes_by_the_masks_widths(tmp_path):
    network = _network(tmp_path)
    settings = _settings(tmp_path, run_haemodynamics=False)
    model, boundaries = _diameters(settings, network)
    drawn = _draw_in_the_wide_tube(model.graph)

    out = post_process(settings, model, boundaries, SCHEMA, network=network)

    assert out.graph.edges[drawn][VESSEL_SIZE_CLASS] == "wide"
    assert out.graph.edges[drawn]["branch_order"] == "Unknown_Large"


# --- preflight -------------------------------------------------------------------


def _preflight(**values):
    settings = {
        "classify_vessels_by_size": True,
        "vessel_size_mid_min_diameter_um": 6.0,
        "vessel_size_wide_min_diameter_um": 15.0,
        "run_haemodynamics": True,
        "use_edt_diameter_crosscheck": True,
        "edt_diameter_prefer_over_table_on_fwhm_failure": True,
        "segmentation_source": "plasma_column",
        **values,
    }
    return check_vessel_size_classes(settings)


def test_preflight_says_what_the_convention_does_and_on_which_basis():
    report = _preflight()
    assert report.errors == [] and report.warnings == []
    (passed,) = [detail for label, detail in report.passed if label == "vessel classes by size"]
    assert "wide at least 15 µm; mid 6-15 µm; narrow under 6 µm" in passed
    assert "do not carry to the other" in passed
    assert "plasma_column" in passed


@pytest.mark.parametrize(
    "values, words",
    [
        ({"vessel_size_mid_min_diameter_um": None}, "vessel_size_mid_min_diameter_um is not set"),
        ({"vessel_size_wide_min_diameter_um": None}, "vessel_size_wide_min_diameter_um is not set"),
        ({"vessel_size_mid_min_diameter_um": 15.0}, "must be below"),
        ({"use_edt_diameter_crosscheck": False}, "no vessel's width is measured"),
        ({"edt_diameter_prefer_over_table_on_fwhm_failure": False}, "no vessel's width is measured"),
    ],
)
def test_preflight_refuses_a_convention_it_cannot_apply(values, words):
    report = _preflight(**values)
    assert len(report.errors) == 1 and words in report.errors[0]


def test_preflight_accepts_raw_image_widths_alone(tmp_path):
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"")
    report = _preflight(
        use_edt_diameter_crosscheck=False,
        measure_diameters_from_raw_image=True,
        raw_image_path=raw,
    )
    assert report.errors == []


def test_preflight_warns_that_without_haemodynamics_the_mask_decides():
    report = _preflight(run_haemodynamics=False, use_edt_diameter_crosscheck=False)
    assert report.errors == []
    assert any("mask's own widths" in warning for warning in report.warnings)


def test_preflight_warns_of_branch_order_settings_left_unread():
    report = _preflight(strict_branch_order_assignment=True)
    assert any("strict_branch_order_assignment" in warning for warning in report.warnings)


def test_preflight_is_silent_with_the_box_off_or_under_masks():
    for values in ({"classify_vessels_by_size": False}, {"automated_vessel_assignment": True}):
        report = _preflight(vessel_size_mid_min_diameter_um=None, **values)
        assert report.errors == [] and report.warnings == [] and report.passed == []


def test_the_schema_warns_of_a_box_left_ticked_under_the_masks():
    settings = {**SCHEMA.defaults(), "automated_vessel_assignment": True, "classify_vessels_by_size": True}
    with pytest.warns(IneffectiveSettingWarning, match="classify_vessels_by_size"):
        SCHEMA.validate(settings)


# --- statistics ------------------------------------------------------------------


def _classed(scheme_graph):
    for _u, _v, data in scheme_graph.edges(data=True):
        data.setdefault("length", 10.0)
    return scheme_graph


def test_the_statistics_head_each_class_by_its_widths():
    G = _classed(_star({(0, 1): 20.0, (1, 2): 9.0, (2, 3): 4.0}))
    tag_vessels_by_size(G, 6.0, 15.0)
    assign_unknown_vessel_classes(G)

    rows = compute_vessel_class_statistics(G)["Vessel Classes"]

    assert rows["Vessel Classes From"] == (
        "a size convention: each vessel by its measured width (wide at least 15 µm, "
        "mid 6-15 µm, narrow under 6 µm); no arteriole or venule side"
    )
    assert [key for key, value in rows.items() if isinstance(value, dict)] == [
        "Vessels at least 15 µm wide", "Vessels 6-15 µm wide", "Vessels under 6 µm wide",
    ]
    assert rows["Vessels 6-15 µm wide"]["Vessel Count"] == 1


def test_a_size_convention_reports_the_rows_global_masks_would_for_the_same_tags():
    by_size = _classed(_star({(0, 1): 20.0, (1, 2): 9.0, (2, 3): 4.0, (3, 4): 9.0}))
    tag_vessels_by_size(by_size, 6.0, 15.0)
    by_masks = by_size.copy()
    by_masks.graph.pop(VESSEL_CLASSES_SOURCE)
    for G in (by_size, by_masks):
        assign_unknown_vessel_classes(G)

    sized = compute_vessel_class_statistics(by_size)["Vessel Classes"]
    masked = compute_vessel_class_statistics(by_masks)["Vessel Classes"]

    titles = {
        "Large Vessels": "Vessels at least 15 µm wide",
        "Small Vessels": "Vessels 6-15 µm wide",
        "Capillaries": "Vessels under 6 µm wide",
    }
    for mask_title, size_title in titles.items():
        assert sized[size_title] == masked[mask_title]


# --- outputs ---------------------------------------------------------------------


def test_the_connectivity_table_names_each_vessels_class():
    G = _classed(_star({(0, 1): 20.0, (1, 2): 4.0}))
    assert {row["Size class"] for row in graph.connectivity_rows(G, [0], [2])} == {""}
    tag_vessels_by_size(G, 6.0, 15.0)
    rows = graph.connectivity_rows(G, [0], [2])
    assert [row["Size class"] for row in rows] == ["wide", "narrow"]


def test_the_citations_state_the_thresholds_and_cite_nothing():
    from haemolynx.pipeline.citations import render_citations

    text = render_citations({
        "classify_vessels_by_size": True,
        "vessel_size_mid_min_diameter_um": 6.0,
        "vessel_size_wide_min_diameter_um": 15.0,
    })
    assert "Vessel classes" in text
    assert "mid at least 6.0 um and wide at least 15.0 um" in text
    assert "No published rule" in text
    assert "Vessel classes" not in render_citations({"classify_vessels_by_size": False})


def test_a_template_refuses_a_run_classed_by_size():
    from haemolynx.synthetic.template import TemplateError, check_template_run

    G = _classed(_star({(0, 1): 20.0, (1, 2): 4.0}))
    tag_vessels_by_size(G, 6.0, 15.0)
    assign_unknown_vessel_classes(G)

    with pytest.raises(TemplateError, match="by a size convention") as refused:
        check_template_run(None, G, {"inlet": [0], "outlet": [2]})
    assert refused.value.reason == "global_classes"


def test_a_batch_refuses_the_convention_and_says_why():
    from haemolynx.pipeline.batch import check_batch

    settings = {**SCHEMA.defaults(), "classify_vessels_by_size": True}
    errors = check_batch(settings, SCHEMA).errors
    assert any("classify_vessels_by_size classes the vessels without masks" in e for e in errors)


def test_a_saved_run_keeps_the_classes_and_the_record(tmp_path):
    import pickle

    G = _star({(0, 1): 20.0, (1, 2): 4.0})
    tag_vessels_by_size(G, 6.0, 15.0)
    path = tmp_path / "graph.pkl"
    path.write_bytes(pickle.dumps(G))

    loaded = pickle.loads(path.read_bytes())

    assert size_convention_of(loaded) == size_convention_of(G)
    assert _attr(loaded, VESSEL_SIZE_CLASS) == _attr(G, VESSEL_SIZE_CLASS)


# --- a whole run -----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_a_whole_run_classes_each_vessel_by_its_own_width(tmp_path, caplog):
    """The tubes of known width through every stage, skeleton to export: each
    vessel's class is the class of its width, all three classes are there,
    no vessel has a branch order of its own, and the exports name the
    classes -- the VTK array and thresholds, and the statistics' headings."""
    import csv

    import pyvista as pv
    import tifffile

    from haemolynx.pipeline import run_pipeline_stages

    image = tmp_path / "tubes.tif"
    tifffile.imwrite(image, tube_chain_mask(branches=True))
    values = {
        **SCHEMA.defaults(),
        "input_path": image,
        "vtk_output_prefix": tmp_path / "run",
        "statistics": True,
        "show_plots_in_ide": False,
        "interactive_plots": False,
        "visualize_results": False,
        "voxel_size_policy": "override",
        "voxel_size_override_xyz": (1.0, 1.0, 1.0),
        "classify_vessels_by_size": True,
        "vessel_size_mid_min_diameter_um": 6.0,
        "vessel_size_wide_min_diameter_um": 15.0,
        "use_edt_diameter_crosscheck": True,
        "measure_diameters_from_raw_image": False,
    }
    settings = resolve_settings(values, schema=SCHEMA, config_path=None, overrides={"plot_dir": tmp_path / "plots"})

    with caplog.at_level(logging.INFO):
        G = run_pipeline_stages(settings, SCHEMA)

    vessels = [data for _u, _v, data in G.edges(data=True) if not data.get(IS_ZERO_RESISTANCE)]
    by_width = [
        data for data in vessels if data.get(VESSEL_SIZE_FROM) == "width"
    ]
    assert by_width, "no vessel was classed by a width of its own"
    for data in by_width:
        assert data[VESSEL_SIZE_CLASS] == vessel_size_class(data["diameter_um"], 6.0, 15.0)
    assert {data[VESSEL_SIZE_CLASS] for data in vessels} == set(CLASSES)
    expected_label = {"wide": "Unknown_Large", "mid": "Unknown_Small", "narrow": "Unknown_Cap"}
    assert all(data["branch_order"] == expected_label[data[VESSEL_SIZE_CLASS]] for data in vessels)
    assert size_convention_of(G)["counts"]["wide"] >= 1
    assert "Size convention: vessels classed by their measured width" in caplog.text

    mesh = pv.read(str(tmp_path / "run_vessels.vtp"))
    assert set(mesh.cell_data["vessel_size_class"]) - {""} == set(CLASSES)
    assert float(mesh.field_data["vessel_size_wide_min_um"][0]) == 15.0
    statistics = tmp_path / "tubes_statistics.csv"
    assert statistics.exists(), sorted(p.name for p in tmp_path.glob("*.csv"))
    with statistics.open(encoding="utf-8") as handle:
        sections = {row["Section"] for row in csv.DictReader(handle)}
    assert "Vessel Classes > Vessels at least 15 µm wide" in sections
    assert "Vessel Classes > Vessels under 6 µm wide" in sections
