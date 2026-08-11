"""The selectable viscosity laws, and how far apart they actually are.

Issue #85 asked for a test that the capillary power law and Pries agree below
7 um, and said in advance what to do if they did not: "If they do not agree,
that is the finding -- it would mean the current mu_ref = 3.0 mPa.s at 5 um
calibration is inconsistent with Pries."

They do not agree. Pries in vivo is about five times the power law across the
whole capillary range, in viscosity, in resistance and in the flow a network
solves to. That is measured here rather than asserted away, because it is the
number someone has to decide about: either the 5 um calibration is too low by
five times, or the in vivo law's endothelial surface layer does not belong in
this model.

Where the two *do* agree is at the top end -- Pries and the 3.5 mPa.s
large-vessel constant are within ~13% from 100 um up -- which is what makes
the constant a defensible placeholder for big vessels and a poor one for
arterioles.
"""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

from haemolynx import haemodynamics
from haemolynx.haemodynamics import PoiseuilleModel
from haemolynx.haemodynamics.viscosity import (
    CAPILLARY_REGIME_MAX_DIAMETER_UM,
    DEFAULT_HAEMATOCRIT,
    LARGE_VESSEL_VISCOSITY_PA_S,
    PLASMA_VISCOSITY_PA_S,
    PRIES_MAX_DIAMETER_UM,
    PRIES_MIN_DIAMETER_UM,
    PlaceholderViscosityWarning,
    pries_in_vivo_viscosity,
    validity_range_um,
    viscosity_for,
)

SEGMENT_LENGTH_UM = 500.0

#: The capillary range both laws claim, from Pries' fitted floor to the power
#: law's ceiling.
CAPILLARY_DIAMETERS_UM = (3.3, 4.0, 5.0, 6.0, 7.0)

#: Where the large-vessel constant is meant to be a good answer.
LARGE_DIAMETERS_UM = (100.0, 200.0, 500.0, 1000.0)


def _model(law: str, haematocrit: float = DEFAULT_HAEMATOCRIT) -> PoiseuilleModel:
    return PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
        viscosity_law=law,
        haematocrit=haematocrit,
    )


def _flow_through_a_chain(law: str, diameter_um: float) -> float:
    """Flow (m^3/s) through three vessels in series, solved end to end.

    A resistance test can be satisfied by a law that is wrong in a way the
    solve cancels out, so the comparison is repeated on what a run actually
    reports.
    """
    G = nx.MultiGraph()
    for node in range(4):
        G.add_node(node, pos=np.array([0.0, 0.0, node * SEGMENT_LENGTH_UM]))
    for node in range(3):
        G.add_edge(
            node,
            node + 1,
            branch_order="B01",
            length=SEGMENT_LENGTH_UM,
            voxels=[
                (0.0, 0.0, node * SEGMENT_LENGTH_UM),
                (0.0, 0.0, (node + 1) * SEGMENT_LENGTH_UM),
            ],
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderViscosityWarning)
        _model(law).set_poiseuille_resistances(G, {"B01": diameter_um})

    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    solved = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        input_p_bc=1000.0,
        output_p_bc=500.0,
        starting_nodes=[0],
        output_nodes=[3],
    )
    haemodynamics.set_edge_flows(G, node_list, solved["pressure"])
    flows = [abs(data["flow_signed"]) for _u, _v, data in G.edges(data=True)]
    # Series: every vessel carries the same flow, which is also the check that
    # the solve is doing what this test thinks it is.
    assert flows == pytest.approx([flows[0]] * len(flows), rel=1e-9)
    return flows[0]


# ---------------------------------------------------------------------------
# The finding: below 7 um the two laws do not agree
# ---------------------------------------------------------------------------

#: Measured, not chosen. Pries / power law over 3.3-7 um.
CAPILLARY_DISAGREEMENT = (4.8, 6.0)


@pytest.mark.parametrize("diameter", CAPILLARY_DIAMETERS_UM)
def test_pries_is_about_five_times_the_power_law_in_capillaries(diameter):
    """The agreement test #85 asked for, reporting what is actually there.

    Not a tolerance anyone chose: 4.8-6.0x is what the two laws do. The power
    law is pinned to 3.0 mPa.s at 5 um; Pries in vivo says 15.1 mPa.s at the
    same diameter, because it includes the ~1.1 um endothelial surface layer
    that blood does not flow through, and a 5 um vessel has little else.
    """
    power = viscosity_for(diameter, law="capillary_power_law")
    pries = viscosity_for(diameter, law="pries_in_vivo")
    low, high = CAPILLARY_DISAGREEMENT
    assert low < pries / power < high, (
        f"at {diameter} um the laws differ by {pries / power:.2f}x "
        f"({power * 1e3:.2f} vs {pries * 1e3:.2f} mPa.s); if this has moved, "
        "one of the laws changed and every resistance moved with it"
    )


@pytest.mark.parametrize("diameter", CAPILLARY_DIAMETERS_UM)
def test_the_disagreement_carries_straight_into_resistance(diameter):
    """Resistance is linear in viscosity, so the same factor, not a softened one."""
    power = _model("capillary_power_law").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    pries = _model("pries_in_vivo").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    low, high = CAPILLARY_DISAGREEMENT
    assert low < pries / power < high


def test_and_into_the_flow_a_network_solves_to():
    """Five times the resistance is a fifth of the flow, end to end."""
    power = _flow_through_a_chain("capillary_power_law", 5.0)
    pries = _flow_through_a_chain("pries_in_vivo", 5.0)
    assert power / pries == pytest.approx(
        viscosity_for(5.0, law="pries_in_vivo")
        / viscosity_for(5.0, law="capillary_power_law"),
        rel=1e-6,
    )
    assert 4.8 < power / pries < 6.0


# ---------------------------------------------------------------------------
# Where they do agree: large vessels against the constant
# ---------------------------------------------------------------------------

#: Pries against the 3.5 mPa.s constant from 100 um up. The constant was
#: chosen as "whole blood once Fahraeus-Lindqvist has died out", and this is
#: how close that is to the law that models the whole curve.
LARGE_VESSEL_TOLERANCE = 0.15


@pytest.mark.parametrize("diameter", LARGE_DIAMETERS_UM)
def test_pries_agrees_with_the_large_vessel_constant(diameter):
    """The placeholder is a good answer for big vessels, which is its defence."""
    pries = viscosity_for(diameter, law="pries_in_vivo")
    assert pries == pytest.approx(
        LARGE_VESSEL_VISCOSITY_PA_S, rel=LARGE_VESSEL_TOLERANCE
    ), f"at {diameter} um Pries says {pries * 1e3:.3f} mPa.s"


@pytest.mark.parametrize("diameter", LARGE_DIAMETERS_UM)
def test_a_large_vessel_resistance_is_the_same_either_way(diameter):
    """Same comparison through Poiseuille, where the run actually uses it.

    `capillary_power_law` returns the constant above 7 um, so this is the
    constant case against the new law on the same vessel.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderViscosityWarning)
        constant = _model("capillary_power_law").resistance_of_uniform_segment(
            SEGMENT_LENGTH_UM, diameter
        )
    pries = _model("pries_in_vivo").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    assert pries == pytest.approx(constant, rel=LARGE_VESSEL_TOLERANCE)


def test_a_large_vessel_network_solves_to_the_same_flow_either_way():
    """And end to end, which is what a user would compare."""
    constant = _flow_through_a_chain("capillary_power_law", 200.0)
    pries = _flow_through_a_chain("pries_in_vivo", 200.0)
    assert pries == pytest.approx(constant, rel=LARGE_VESSEL_TOLERANCE)


# ---------------------------------------------------------------------------
# The intermediate region, which is what #90 was about
# ---------------------------------------------------------------------------


def test_pries_has_the_fahraeus_lindqvist_minimum_in_the_right_place():
    """Apparent viscosity falls with diameter, bottoms out, then rises.

    In vivo the minimum sits somewhere in the tens of microns -- much wider
    than the ~7 um of an in vitro tube, because the surface layer takes a
    larger share of a narrow vessel. A law without this shape is not modelling
    the transition, whatever it does at the ends.
    """
    diameters = np.linspace(PRIES_MIN_DIAMETER_UM, 400.0, 4000)
    viscosities = np.array([pries_in_vivo_viscosity(d) for d in diameters])
    at = diameters[int(np.argmin(viscosities))]
    assert 20.0 < at < 80.0, f"minimum at {at:.1f} um"


def test_pries_is_monotonic_on_each_side_of_that_minimum():
    diameters = np.linspace(PRIES_MIN_DIAMETER_UM, 400.0, 4000)
    viscosities = np.array([pries_in_vivo_viscosity(d) for d in diameters])
    turn = int(np.argmin(viscosities))
    assert np.all(np.diff(viscosities[:turn]) < 0), "should fall to the minimum"
    assert np.all(np.diff(viscosities[turn:]) > 0), "should rise after it"


def test_pries_never_predicts_blood_thinner_than_plasma():
    """The failure that bounds the power law at 7 um, across Pries' whole range."""
    diameters = np.geomspace(PRIES_MIN_DIAMETER_UM, PRIES_MAX_DIAMETER_UM, 500)
    for diameter in diameters:
        assert pries_in_vivo_viscosity(diameter) > PLASMA_VISCOSITY_PA_S


def test_pries_is_continuous_where_the_power_law_steps():
    """The concrete defect #90 named: a 7.5 um vessel more resistive than a 7 um one.

    The power law jumps from 1.72 to 3.5 mPa.s at 7 um. Pries crosses the same
    diameter without a step, so resistance stays monotonic in diameter there.
    """
    edge = CAPILLARY_REGIME_MAX_DIAMETER_UM
    below = pries_in_vivo_viscosity(edge - 1e-6)
    above = pries_in_vivo_viscosity(edge + 1e-6)
    assert below == pytest.approx(above, rel=1e-5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderViscosityWarning)
        power_below = viscosity_for(edge, law="capillary_power_law")
        power_above = viscosity_for(edge + 0.5, law="capillary_power_law")
    assert power_above > 2 * power_below, (
        "the power law's step is what this is contrasted against; if it has "
        "gone, this test has lost its point"
    )


def test_a_wider_vessel_is_never_more_resistive_under_pries():
    """What the power law's step gets wrong, stated as the property that matters."""
    diameters = np.linspace(3.5, 300.0, 800)
    resistances = [
        _model("pries_in_vivo").resistance_of_uniform_segment(SEGMENT_LENGTH_UM, d)
        for d in diameters
    ]
    assert np.all(np.diff(resistances) < 0)


def test_the_power_law_does_get_that_wrong_in_the_placeholder_band():
    """Named so the contrast above is not asserting something vacuous."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderViscosityWarning)
        at_seven = _model("capillary_power_law").resistance_of_uniform_segment(
            SEGMENT_LENGTH_UM, 7.0
        )
        at_eight = _model("capillary_power_law").resistance_of_uniform_segment(
            SEGMENT_LENGTH_UM, 8.0
        )
    assert at_eight > at_seven, (
        "the 7 um step used to make an 8 um vessel more resistive than a 7 um "
        "one; if that has been fixed, this test should go"
    )


# ---------------------------------------------------------------------------
# Haematocrit, the parameter the new law introduces
# ---------------------------------------------------------------------------


def test_at_the_reference_haematocrit_the_correction_is_exactly_one():
    """The law is written around H = 0.45; f(H) is 1 there by construction."""
    relative = pries_in_vivo_viscosity(20.0) / PLASMA_VISCOSITY_PA_S
    d = 20.0
    mu45 = 6 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    surface = (d / (d - 1.1)) ** 2
    assert relative == pytest.approx((1 + (mu45 - 1) * surface) * surface)


@pytest.mark.parametrize("diameter", [5.0, 20.0, 100.0])
def test_thicker_blood_is_more_viscous(diameter):
    values = [
        pries_in_vivo_viscosity(diameter, haematocrit=h) for h in (0.2, 0.45, 0.6)
    ]
    assert values[0] < values[1] < values[2]


def test_haematocrit_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="fraction"):
        pries_in_vivo_viscosity(10.0, haematocrit=1.5)


def test_a_vessel_narrower_than_the_surface_layer_is_refused():
    """The term is singular at 1.1 um; a message beats a division by zero."""
    with pytest.raises(ValueError, match="singular"):
        pries_in_vivo_viscosity(1.0)


# ---------------------------------------------------------------------------
# Choosing a law
# ---------------------------------------------------------------------------


def test_the_default_law_is_the_one_that_was_there_before():
    """Adding a law must not silently move everybody's numbers by five times."""
    from haemolynx.pipeline import default_schema

    assert default_schema()["viscosity_law"].default == "capillary_power_law"
    assert PoiseuilleModel(40.0, 100.0).viscosity_law == "capillary_power_law"


def test_an_unknown_law_is_refused_by_name():
    with pytest.raises(ValueError, match="pries_in_vivo"):
        PoiseuilleModel(40.0, 100.0, viscosity_law="pries")
    with pytest.raises(ValueError, match="Unknown viscosity_law"):
        viscosity_for(5.0, law="nonsense")


def test_only_the_law_with_a_placeholder_warns():
    """`pries_in_vivo` covers 7-100 um, so it has nothing to apologise for."""
    with pytest.warns(PlaceholderViscosityWarning):
        viscosity_for(20.0, law="capillary_power_law")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlaceholderViscosityWarning)
        viscosity_for(20.0, law="pries_in_vivo")


def test_each_law_states_the_range_it_was_fitted_over():
    assert validity_range_um("pries_in_vivo") == (
        PRIES_MIN_DIAMETER_UM,
        PRIES_MAX_DIAMETER_UM,
    )
    assert validity_range_um("capillary_power_law")[1] == CAPILLARY_REGIME_MAX_DIAMETER_UM


def test_the_run_records_which_law_produced_its_numbers():
    """Resistances are not comparable across laws, so the graph carries it."""
    description = _model("pries_in_vivo", haematocrit=0.4).describe_viscosity_law()
    assert "pries_in_vivo" in description
    assert "0.4" in description
    assert "3.3" in description and "1978" in description


def test_the_law_survives_on_the_graph_a_run_pickles():
    """A graph read back next month has to say what produced its resistances."""
    import pickle

    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, SEGMENT_LENGTH_UM]))
    G.add_edge(0, 1, branch_order="B01", length=SEGMENT_LENGTH_UM,
               voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, SEGMENT_LENGTH_UM)])

    config = haemodynamics.HaemodynamicsApplyConfig(
        diameters={
            "diameter_by_branch_order": {"B01": 5.0},
            "viscosity_law": "pries_in_vivo",
            "haematocrit": 0.42,
        },
        fwhm={},
        resistance_node_pair=(0, 1),
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    G, summary = haemodynamics.apply_poiseuille_haemodynamics(G, config=config)

    restored = pickle.loads(pickle.dumps(G))
    assert restored.graph["viscosity_law"] == "pries_in_vivo"
    assert restored.graph["haematocrit"] == pytest.approx(0.42)
    assert "pries_in_vivo" in summary["viscosity"]


def test_choosing_the_law_in_a_config_reaches_the_resistances(tmp_path):
    """The setting is the whole point of #85: selectable, not edited in code."""
    from haemolynx.parsers import load_config
    from haemolynx.pipeline import default_schema, write_default_config

    schema = default_schema()
    path = tmp_path / "config.yaml"
    write_default_config(
        path,
        schema=schema,
        values={**{s.name: s.default for s in schema},
                "viscosity_law": "pries_in_vivo", "haematocrit": 0.5},
    )
    loaded = load_config(path, schema)
    assert loaded["viscosity_law"] == "pries_in_vivo"
    assert loaded["haematocrit"] == pytest.approx(0.5)

    model = PoiseuilleModel(
        40.0, 100.0,
        viscosity_law=loaded["viscosity_law"],
        haematocrit=loaded["haematocrit"],
    )
    assert model.calculate_viscosity(5.0) == pytest.approx(
        pries_in_vivo_viscosity(5.0, haematocrit=0.5)
    )
