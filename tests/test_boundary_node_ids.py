"""Boundary nodes named by ID: the ``node_ids`` selection method.

Every other method chooses among terminals by where they are. This one takes
exactly the nodes listed -- which is what clicking a node on the Boundaries
tab writes -- terminals or not, since an arteriole/venule boundary is where a
vessel hands over to the capillaries, and that is usually a junction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx import graph
from haemolynx.graph.boundaries import BOUNDARY_ROLE_SETTINGS
from haemolynx.pipeline import default_schema, stages
from haemolynx.pipeline.checks import check_large_vessel_branch_order_mode_prerequisites

IMAGE_SHAPE = (48, 48, 48)


def _defaults(**overrides) -> dict:
    settings = default_schema().defaults()
    settings.update(overrides)
    return settings


def _branching_network() -> nx.MultiGraph:
    """A Y: terminal 0 at the stem, junction 1, then two arms to terminals 4, 5."""
    positions = {
        0: (24.0, 8.0, 24.0),
        1: (24.0, 20.0, 24.0),
        2: (24.0, 32.0, 12.0),
        3: (24.0, 32.0, 36.0),
        4: (24.0, 41.0, 8.0),
        5: (24.0, 41.0, 40.0),
    }
    G = nx.MultiGraph()
    for node_id, position in positions.items():
        G.add_node(node_id, pos=np.asarray(position, dtype=float))
    for u, v in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5)):
        G.add_edge(u, v, length=1.0)
    return G


def _network_for_stages(G: nx.MultiGraph, output_dir: Path) -> stages.VesselNetwork:
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    volume = stages.SkeletonisedVolume(
        image=image,
        skeleton=image.astype(bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=output_dir,
    )
    return stages.VesselNetwork(graph=G, volume=volume)


# --- the schema offers it to every role --------------------------------------


@pytest.mark.parametrize("role", sorted(BOUNDARY_ROLE_SETTINGS))
def test_every_roles_dropdown_offers_node_ids_and_has_a_list_for_it(role):
    names = BOUNDARY_ROLE_SETTINGS[role]
    schema = default_schema()

    assert "node_ids" in schema[names["method"]].choices
    assert schema[names["node_ids"]].default == []


# --- the selector ------------------------------------------------------------


@pytest.mark.parametrize("role", sorted(BOUNDARY_ROLE_SETTINGS))
def test_node_ids_takes_exactly_the_nodes_listed(role):
    names = BOUNDARY_ROLE_SETTINGS[role]
    settings = _defaults(**{names["method"]: "node_ids", names["node_ids"]: [5, 0]})

    chosen = graph.select_boundary_nodes_for_role(
        _branching_network(), IMAGE_SHAPE, settings, role
    )

    assert chosen == [0, 5]


def test_a_junction_can_be_a_vessel_boundary():
    """No other method can name node 1: it is not a terminal."""
    settings = _defaults(
        arteriole_boundary_selection_method="node_ids",
        arteriole_boundary_node_ids=[1],
    )

    chosen = graph.select_boundary_nodes_for_role(
        _branching_network(), IMAGE_SHAPE, settings, "arteriole_boundary"
    )

    assert chosen == [1]


def test_an_id_the_graph_lacks_stops_the_run_naming_the_setting_and_the_id():
    settings = _defaults(inlet_node_selection_method="node_ids", inlet_node_ids=[0, 99])

    with pytest.raises(ValueError) as failure:
        graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "inlet"
        )

    message = str(failure.value)
    assert "inlet_node_ids" in message
    assert "99" in message
    assert "renumber" in message, "the usual cause -- a rebuilt graph -- is named"


def test_an_empty_list_names_the_setting_to_fill():
    settings = _defaults(outlet_node_selection_method="node_ids", outlet_node_ids=[])

    with pytest.raises(ValueError, match="outlet_node_ids"):
        graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "outlet"
        )


@pytest.mark.parametrize("written", [[4], ["4"], [4.0], [np.int64(4)], 4])
def test_an_id_however_it_was_written_names_the_same_node(written):
    """A quoted ID in a hand-edited config, a numpy int, a bare ID without
    brackets: all of them mean node 4, and the result is the node itself."""
    settings = _defaults(outlet_node_selection_method="node_ids", outlet_node_ids=written)

    chosen = graph.select_boundary_nodes_for_role(
        _branching_network(), IMAGE_SHAPE, settings, "outlet"
    )

    assert chosen == [4]
    assert type(chosen[0]) is int


def test_a_fractional_id_is_not_rounded_onto_a_node():
    settings = _defaults(outlet_node_selection_method="node_ids", outlet_node_ids=[4.5])

    with pytest.raises(ValueError, match="4.5"):
        graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "outlet"
        )


def test_a_node_an_earlier_role_took_is_left_out_and_logged(caplog):
    settings = _defaults(outlet_node_selection_method="node_ids", outlet_node_ids=[0, 5])

    with caplog.at_level(logging.WARNING, logger="haemolynx.graph.boundaries"):
        chosen = graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "outlet", exclude_nodes=[0]
        )

    assert chosen == [5]
    assert "earlier boundary role" in caplog.text
    assert "[0]" in caplog.text


def test_a_pressure_boundary_on_a_junction_is_warned_about(caplog):
    """An inlet clicked on a junction is most often a click that missed the
    terminal beside it -- worth saying, not worth refusing."""
    settings = _defaults(inlet_node_selection_method="node_ids", inlet_node_ids=[1])

    with caplog.at_level(logging.WARNING, logger="haemolynx.graph.boundaries"):
        chosen = graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "inlet"
        )

    assert chosen == [1]
    assert "not terminals" in caplog.text


def test_a_vessel_boundary_on_a_junction_is_not_warned_about(caplog):
    settings = _defaults(
        venule_boundary_selection_method="node_ids", venule_boundary_node_ids=[1]
    )

    with caplog.at_level(logging.WARNING, logger="haemolynx.graph.boundaries"):
        graph.select_boundary_nodes_for_role(
            _branching_network(), IMAGE_SHAPE, settings, "venule_boundary"
        )

    assert "not terminals" not in caplog.text


def test_a_graph_with_no_terminals_still_has_the_listed_nodes():
    """The other methods return nothing without a terminal; this one does
    not look for terminals at all."""
    G = nx.MultiGraph()
    for node_id in range(3):
        G.add_node(node_id, pos=np.asarray((float(node_id), 0.0, 0.0)))
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    settings = _defaults(inlet_node_selection_method="node_ids", inlet_node_ids=[2])

    assert graph.select_boundary_nodes_for_role(G, IMAGE_SHAPE, settings, "inlet") == [2]


@pytest.mark.parametrize("role", sorted(BOUNDARY_ROLE_SETTINGS))
def test_a_roles_node_ids_are_ignored_unless_its_method_is_node_ids(role):
    """The same guarantee the coordinates and volume boxes have: a stale list
    left in a config must not steer a role that selects some other way."""
    names = BOUNDARY_ROLE_SETTINGS[role]
    base = {
        names["method"]: "all_degree_1",
        names["coordinates"]: [],
        names["volume_boxes"]: [],
        names["node_ids"]: [],
        "inlet_nodes": [],
    }
    probe = {**base, names["node_ids"]: [1, 99]}

    G = _branching_network()
    assert graph.select_boundary_nodes_for_role(G, IMAGE_SHAPE, probe, role) == (
        graph.select_boundary_nodes_for_role(G, IMAGE_SHAPE, base, role)
    )


def test_select_nodes_by_id_is_public():
    assert graph.select_nodes_by_id(_branching_network(), ["3", 2]) == [3, 2]


# --- the stage ---------------------------------------------------------------


def test_the_stage_uses_listed_inlets_and_outlets(tmp_path):
    settings = _defaults(
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[0],
        outlet_node_selection_method="node_ids",
        outlet_node_ids=[4, 5],
        plot_dir=tmp_path,
    )

    boundaries = stages.assign_boundaries(
        settings, _network_for_stages(_branching_network(), tmp_path)
    )

    assert boundaries.inlet_nodes == [0]
    assert boundaries.outlet_nodes == [4, 5]
    assert boundaries.resistance_node_pair == (0, 4)


def test_the_stage_measures_degree_1_from_inlet_outlets_from_the_inlets_it_chose(tmp_path):
    """Regression: the stage wrote the inlets into the run's list only after
    choosing the outlets, so an outlet chosen by its distance from the inlets
    always found none and stopped the run, telling the user to choose the
    inlets first -- which they had."""
    settings = _defaults(
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[0],
        outlet_node_selection_method="degree_1_from_inlet",
        boundary_distance_from_inlet_node=30.0,
        boundary_open_end_max_distance_um=None,
        plot_dir=tmp_path,
    )

    boundaries = stages.assign_boundaries(
        settings, _network_for_stages(_branching_network(), tmp_path)
    )

    # 4 and 5 are 37.7 um from inlet 0; nothing else is a terminal past 30 um.
    assert boundaries.inlet_nodes == [0]
    assert boundaries.outlet_nodes == [4, 5]


@pytest.mark.parametrize(
    "role, says",
    [
        ("inlet", "which this role is the one to choose"),
        ("large_vessel_inlet", "which this role is the one to choose"),
        ("outlet", "inlet_node_selection_method chose none"),
        ("large_vessel_outlet", "large_vessel_inlet_node_selection_method chose none"),
    ],
)
def test_degree_1_from_inlet_with_no_inlets_names_what_would_choose_them(role, says):
    """Regression: every role was told to change inlet_node_selection_method,
    which does not choose the large-vessel network's inlet, and cannot help
    an inlet role measuring from itself."""
    names = BOUNDARY_ROLE_SETTINGS[role]
    settings = _defaults(**{names["method"]: "degree_1_from_inlet"}, inlet_nodes=[])

    with pytest.raises(ValueError, match=says):
        graph.select_boundary_nodes_for_role(_branching_network(), IMAGE_SHAPE, settings, role)


def test_an_edge_percent_outlet_band_keeps_terminals_the_unused_inlet_band_holds(tmp_path):
    """Regression: the outlet band left out every terminal of the first band,
    though the inlets came from node IDs and that band (its row hidden) chose
    nothing: with the outlet band over every terminal, the stem's end, in the
    first 10%, was no outlet -- and with only it besides the inlet, the run
    found none."""
    settings = _defaults(
        inlet_node_selection_method="node_ids",
        inlet_node_ids=[4],
        outlet_node_selection_method="edge_percent",
        boundary_last_percent=100.0,
        boundary_open_end_max_distance_um=None,
        plot_dir=tmp_path,
    )

    boundaries = stages.assign_boundaries(
        settings, _network_for_stages(_branching_network(), tmp_path)
    )

    assert boundaries.inlet_nodes == [4]
    assert boundaries.outlet_nodes == [0, 5]


def test_vessel_boundaries_named_only_by_id_are_selected(tmp_path):
    """The stage used to skip the arteriole/venule roles unless coordinates or
    volume boxes were set, so node IDs alone would have been ignored."""
    settings = _defaults(
        arteriole_boundary_selection_method="node_ids",
        arteriole_boundary_node_ids=[1],
        venule_boundary_selection_method="node_ids",
        venule_boundary_node_ids=[3],
        plot_dir=tmp_path,
    )

    boundaries = stages.assign_boundaries(
        settings, _network_for_stages(_branching_network(), tmp_path)
    )

    assert boundaries.arteriole_boundary_nodes == [1]
    assert boundaries.venule_boundary_nodes == [3]


def test_vessel_boundary_ids_ask_for_hierarchical_branch_orders():
    assert stages._vessel_boundary_configured(
        {
            "arteriole_boundary_selection_method": "node_ids",
            "arteriole_boundary_node_ids": [1],
        },
        "arteriole",
    )
    assert not stages._vessel_boundary_configured(
        {
            "arteriole_boundary_selection_method": "node_ids",
            "arteriole_boundary_node_ids": [],
            "venule_boundary_node_ids": [3],
        },
        "arteriole",
    )


def test_leftover_node_ids_do_not_configure_another_method(tmp_path):
    """Switching off node IDs leaves the list set. That must not run the role.

    Coordinates with an empty coordinate list used to raise. edge_percent
    used to pick terminals instead, because the leftover IDs counted as
    configuration whatever the method was.
    """
    leftover = {
        "arteriole_boundary_node_ids": [1],
        "venule_boundary_node_ids": [3],
    }
    assert not stages._vessel_boundary_configured(
        {**leftover, "arteriole_boundary_selection_method": "coordinates"}, "arteriole"
    )
    assert not stages._vessel_boundary_configured(
        {**leftover, "arteriole_boundary_selection_method": "edge_percent"}, "arteriole"
    )

    for method in ("coordinates", "edge_percent"):
        settings = _defaults(
            arteriole_boundary_selection_method=method,
            arteriole_boundary_node_ids=[1],
            venule_boundary_selection_method=method,
            venule_boundary_node_ids=[3],
            plot_dir=tmp_path,
        )
        boundaries = stages.assign_boundaries(
            settings, _network_for_stages(_branching_network(), tmp_path)
        )
        assert boundaries.arteriole_boundary_nodes == []
        assert boundaries.venule_boundary_nodes == []


def test_preflight_counts_node_ids_as_a_source_of_vessel_boundaries():
    settings = {
        "assign_large_vessel_branch_orders": True,
        "use_large_vessel_masks": True,
        "automated_vessel_assignment": True,
        "cut_network_at_large_vessel_volumes": False,
        "use_small_vessel_masks_for_boundary_assignment": False,
    }

    without = check_large_vessel_branch_order_mode_prerequisites(settings)
    with_ids = check_large_vessel_branch_order_mode_prerequisites(
        {
            **settings,
            "arteriole_boundary_selection_method": "node_ids",
            "arteriole_boundary_node_ids": [7],
        }
    )
    leftover = check_large_vessel_branch_order_mode_prerequisites(
        {
            **settings,
            "arteriole_boundary_selection_method": "coordinates",
            "arteriole_boundary_node_ids": [7],
        }
    )

    assert without.warnings
    assert not with_ids.warnings
    assert leftover.warnings


# --- preflight, and the hop fallback ------------------------------------------


@pytest.mark.parametrize(
    "role, method, listed",
    [
        ("inlet", "coordinates", "inlet_node_coordinates"),
        ("outlet", "volume", "outlet_node_volumes"),
        ("inlet", "node_ids", "inlet_node_ids"),
    ],
)
def test_preflight_refuses_an_inlet_or_outlet_method_with_an_empty_list(role, method, listed):
    """The Boundaries stage refused it only once every stage before had run;
    a list the masks or an unread method leave unread is no error."""
    from haemolynx.pipeline.checks import check_boundary_selection

    settings = _defaults(**{BOUNDARY_ROLE_SETTINGS[role]["method"]: method})

    (error,) = check_boundary_selection(settings).errors
    assert listed in error
    filled = {**settings, listed: [[1.0, 2.0, 3.0]] if method == "coordinates" else [[[0, 0, 0], [1, 1, 1]]] if method == "volume" else [0]}
    assert not check_boundary_selection(filled).errors
    masks_choose = {**settings, "automated_vessel_assignment": True, "use_large_vessel_masks": True,
                    "large_venule_mask_path": "v.tif", "inlets_outlets_from_vessel_masks": True}
    assert not check_boundary_selection(masks_choose).errors


def test_preflight_warns_of_edge_percent_along_z_on_a_one_slice_image(tmp_path):
    import tifffile

    from haemolynx.pipeline.checks import check_boundary_selection

    image = tmp_path / "flat.tif"
    tifffile.imwrite(image, np.zeros((1, 20, 30), dtype=np.uint8))
    settings = _defaults(input_path=image, boundary_axis=0)

    (warning,) = check_boundary_selection(settings).warnings
    assert "one slice" in warning
    assert not check_boundary_selection({**settings, "boundary_axis": 1}).warnings
    tifffile.imwrite(tmp_path / "stack.tif", np.zeros((4, 20, 30), dtype=np.uint8))
    assert not check_boundary_selection({**settings, "input_path": tmp_path / "stack.tif"}).warnings


def test_the_hop_fallback_takes_no_node_the_other_sides_masks_chose():
    """Regression: an uncovered inlet's hop neighbours could include a venule
    boundary node the masks found, which then ended both tiers at once."""
    G = _branching_network()
    settings = _defaults(
        inlet_nodes=[0], outlet_nodes=[4, 5],
        arteriole_boundary_nodes=[], venule_boundary_nodes=[1],
        small_vessel_boundary_fallback_to_hop_distance=True,
        small_vessel_boundary_fallback_hop_distance=1,
    )
    empty = np.zeros(IMAGE_SHAPE, dtype=bool)

    fallback = stages._small_vessel_boundary_fallback(
        settings, G, small_masks=(empty, empty), large_masks=None, voxel_size_zyx=(1.0, 1.0, 1.0)
    )

    uncovered, hop_nodes = fallback["arteriole_boundary_nodes"]
    assert uncovered == [0]
    assert 1 not in hop_nodes
