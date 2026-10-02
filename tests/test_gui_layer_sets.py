"""Swapping the network the viewer shows: the baseline, or one perturbation.

Pure decisions, no napari: which layers make up a network, what the menu
lists, and what each layer's visibility becomes on a swap.
"""
from __future__ import annotations

from haemolynx.gui.layer_sets import (
    BASELINE,
    BASELINE_LABEL,
    carried_roles,
    layer_set,
    layer_set_choices,
    role_visibility,
    set_layer_names,
    visibility_for,
)
from haemolynx.gui.results import (
    FLOW_DIRECTION,
    NODES,
    PERICYTES,
    VESSEL_TUBES,
    VESSELS,
    perturbation_flow_direction_layer_name,
    perturbation_layer_names,
    perturbation_pericyte_layer_name,
)
from haemolynx.gui.vessel_tubes import vessel_tubes_layer_name

A_VESSELS, A_NODES = perturbation_layer_names("dilate")
A_FLOW = perturbation_flow_direction_layer_name("dilate")
A_PERICYTES = perturbation_pericyte_layer_name("dilate")
A_TUBES = vessel_tubes_layer_name(A_VESSELS)
B_VESSELS, B_NODES = perturbation_layer_names("block")
B_PERICYTES = perturbation_pericyte_layer_name("block")


def test_the_baseline_is_the_run_s_own_vessels_nodes_arrows_and_pericytes():
    assert layer_set(BASELINE) == {
        "vessels": VESSELS, "nodes": NODES, "flow direction": FLOW_DIRECTION,
        "pericytes": PERICYTES,
    }
    assert set_layer_names(BASELINE)[-1] == VESSEL_TUBES


def test_a_perturbation_s_set_is_the_layers_named_after_it():
    assert layer_set("dilate") == {
        "vessels": A_VESSELS, "nodes": A_NODES, "flow direction": A_FLOW,
        "pericytes": A_PERICYTES,
    }
    assert A_TUBES in set_layer_names("dilate")


def test_the_menu_lists_the_baseline_first_then_each_perturbation_once():
    assert layer_set_choices(["dilate", "block", "dilate"]) == [
        (BASELINE_LABEL, BASELINE),
        ("dilate", "dilate"),
        ("block", "block"),
    ]


def test_a_perturbation_called_baseline_is_still_its_own_entry():
    choices = layer_set_choices(["Baseline"])
    assert choices == [(BASELINE_LABEL, None), ("Baseline", "Baseline")]
    assert layer_set("Baseline")["vessels"] != VESSELS


def test_vessels_count_as_on_when_drawn_as_tubes():
    visible = {VESSELS: False, VESSEL_TUBES: True, NODES: True}
    assert role_visibility(BASELINE, visible) == {"vessels": True, "nodes": True}


def test_only_the_kinds_of_layer_a_network_has_are_reported():
    assert role_visibility("dilate", {A_VESSELS: False}) == {"vessels": False}
    assert role_visibility("dilate", {}) == {}


def test_showing_a_perturbation_hides_every_other_network_tubes_and_all():
    present = [VESSELS, VESSEL_TUBES, NODES, FLOW_DIRECTION,
               A_VESSELS, A_NODES, A_FLOW, B_VESSELS, B_NODES]

    result = visibility_for("dilate", [BASELINE, "dilate", "block"],
                            {"nodes": True, "flow direction": False}, present)

    assert result == {
        VESSELS: False, VESSEL_TUBES: False, NODES: False, FLOW_DIRECTION: False,
        B_VESSELS: False, B_NODES: False,
        A_VESSELS: True, A_NODES: True, A_FLOW: False,
    }


def test_the_shown_network_s_tubes_are_left_to_the_drawing_mode():
    """Tubes or lines is the Tubes/Lines control's call, not the swap's."""
    result = visibility_for("dilate", ["dilate"], {}, [A_VESSELS, A_TUBES])
    assert A_TUBES not in result
    assert result[A_VESSELS] is True


def test_a_kind_with_no_carried_answer_keeps_its_own_visibility():
    result = visibility_for(BASELINE, ["dilate"], {}, [VESSELS, NODES, A_NODES])
    assert result == {VESSELS: True, A_NODES: False}


def test_the_baseline_is_hidden_even_when_not_listed():
    result = visibility_for("dilate", ["dilate"], {}, [VESSELS, A_VESSELS])
    assert result == {VESSELS: False, A_VESSELS: True}


def test_layers_not_in_the_viewer_are_not_mentioned():
    assert visibility_for("dilate", ["dilate", "block"], {"nodes": True}, []) == {}


def test_showing_a_perturbation_shows_its_own_pericytes_and_no_one_else_s():
    present = [VESSELS, PERICYTES, A_VESSELS, A_PERICYTES, B_VESSELS, B_PERICYTES]

    result = visibility_for("dilate", [BASELINE, "dilate", "block"],
                            {"pericytes": True}, present)

    assert result == {
        VESSELS: False, PERICYTES: False, B_VESSELS: False, B_PERICYTES: False,
        A_VESSELS: True, A_PERICYTES: True,
    }


def test_the_network_being_left_says_which_kinds_are_on():
    roles = carried_roles(
        incoming={"nodes": False, "pericytes": False},
        outgoing={"nodes": True, "pericytes": False},
        remembered={"nodes": False, "pericytes": True},
    )
    assert roles == {"nodes": True, "pericytes": False}


def test_a_kind_the_network_left_has_none_of_is_remembered_from_before():
    """Pericytes on in one perturbation stay on in the next, past a baseline
    that has no pericytes layer to carry the answer."""
    roles = carried_roles(
        incoming={"vessels": False, "pericytes": False},
        outgoing={"vessels": True},
        remembered={"pericytes": True},
    )
    assert roles == {"vessels": True, "pericytes": True}


def test_the_incoming_network_answers_only_for_a_kind_nothing_else_mentions():
    assert carried_roles({"pericytes": True}, {}, {}) == {"pericytes": True}
