"""The viscosity laws, the diameter each expects, and how far apart they are.

Issue #85 asked for a test that the capillary power law and Pries agree below
7 um, and said what to do if they did not: "If they do not agree, that is the
finding." The first answer here was that they differ by five times -- but that
was the wrong comparison, and the reason is the point of this file.

Pries published two forms. The *in vitro* one is blood in a glass tube of a
known bore, so its diameter is the channel the fluid occupies. The *in vivo*
one is the same blood in a living network, where the measured diameter runs
wall to wall and includes the endothelial surface layer -- so its
``(D / (D - 1.1))^2`` factors, which appear squared and therefore carry
``(D / (D - 1.1))^4``, are the Poiseuille correction for quoting a resistance
against a diameter wider than the channel. That is a diameter correction, not
a property of blood.

HaemoLynx segments plasma-stained images, so its diameters are already the
channel. Against the right form -- in vitro, `diameter_basis="plasma_column"`
-- the two laws agree to 2% at 3 um and diverge from there, which is what a
one-point calibration looks like rather than two rival models. Against the
wrong form they differ by five times, and a test pins that too, because it is
the mistake this setting exists to prevent.
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
    PRIES_SINGULARITY_UM,
    PlaceholderViscosityWarning,
    pries_in_vitro_viscosity,
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


def _model(
    law: str,
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
) -> PoiseuilleModel:
    return PoiseuilleModel(
        constriction_length=40.0,
        constriction_spacing=100.0,
        viscosity_law=law,
        haematocrit=haematocrit,
        diameter_basis=diameter_basis,
    )


def _flow_through_a_chain(
    law: str, diameter_um: float, diameter_basis: str = "plasma_column"
) -> float:
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
        _model(law, diameter_basis=diameter_basis).set_poiseuille_resistances(
            G, {"B01": diameter_um}
        )

    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    solved = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        inlet_p_bc=1000.0,
        outlet_p_bc=500.0,
        inlet_nodes=[0],
        outlet_nodes=[3],
    )
    haemodynamics.set_edge_flows(G, node_list, solved["pressure"])
    flows = [abs(data["flow_signed"]) for _u, _v, data in G.edges(data=True)]
    # Series: every vessel carries the same flow, which is also the check that
    # the solve is doing what this test thinks it is.
    assert flows == pytest.approx([flows[0]] * len(flows), rel=1e-9)
    return flows[0]


# ---------------------------------------------------------------------------
# Below 7 um, against the form that matches how the diameter was measured
# ---------------------------------------------------------------------------

#: In vitro against the power law over 3.3-7 um. They coincide at the bottom
#: and separate steadily: 1.13x at 3.3 um, 3.16x at 7 um. The power law is
#: anchored at one point and its d^-1.647 slope is too steep to hold either
#: side of it -- it crosses below plasma at 8.7 um, which is what its 7 um
#: guard exists to hide.
CAPILLARY_AGREEMENT = {3.3: 1.13, 4.0: 1.49, 5.0: 2.03, 6.0: 2.59, 7.0: 3.16}


def test_the_two_laws_agree_where_the_power_law_was_calibrated():
    """At the bottom of the capillary range they are the same law.

    This is the agreement #85 asked for, and it is real but narrow: 2% at
    3 um. It is also the evidence that the disagreement further up is the
    power law's slope rather than a units error or a wrong constant -- a
    factor-of-five bug does not vanish at one diameter.
    """
    assert pries_in_vitro_viscosity(3.0) == pytest.approx(
        viscosity_for(3.0, law="capillary_power_law"), rel=0.02
    )


@pytest.mark.parametrize("diameter,expected", sorted(CAPILLARY_AGREEMENT.items()))
def test_how_far_apart_they_get_across_the_capillary_range(diameter, expected):
    """Measured, not chosen. If these move, one of the laws changed."""
    ratio = pries_in_vitro_viscosity(diameter) / viscosity_for(
        diameter, law="capillary_power_law"
    )
    assert ratio == pytest.approx(expected, rel=0.02), (
        f"at {diameter} um the laws now differ by {ratio:.2f}x, not {expected}x"
    )


@pytest.mark.parametrize("diameter", CAPILLARY_DIAMETERS_UM)
def test_the_difference_carries_straight_into_resistance(diameter):
    """Resistance is linear in viscosity, so the same factor, not a softened one."""
    power = _model("capillary_power_law").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    pries = _model("pries").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    assert pries / power == pytest.approx(
        pries_in_vitro_viscosity(diameter)
        / viscosity_for(diameter, law="capillary_power_law"),
        rel=1e-9,
    )


def test_and_into_the_flow_a_network_solves_to():
    """Twice the viscosity at 5 um is half the flow, end to end."""
    power = _flow_through_a_chain("capillary_power_law", 5.0)
    pries = _flow_through_a_chain("pries", 5.0)
    assert power / pries == pytest.approx(CAPILLARY_AGREEMENT[5.0], rel=0.02)


# ---------------------------------------------------------------------------
# The mistake the diameter_basis setting exists to prevent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diameter", CAPILLARY_DIAMETERS_UM)
def test_reading_a_plasma_column_as_anatomical_costs_a_factor_of_several(diameter):
    """Subtracting the surface layer from a diameter that never included it.

    A plasma stain images the channel the fluid occupies, so the ~1.1 um of
    glycocalyx is already out of the measurement. Calling that diameter
    anatomical takes it out again, and in a capillary there is little else to
    take.
    """
    as_measured = viscosity_for(diameter, law="pries")
    double_counted = viscosity_for(
        diameter, law="pries", diameter_basis="anatomical"
    )
    assert 1.8 < double_counted / as_measured < 6.0


def test_the_two_bases_are_the_same_physics_read_two_ways():
    """The in vivo form is the in vitro form quoted against a wider diameter.

    A vessel whose anatomical diameter is D carries blood through D - 1.1. Its
    resistance can be written either as the in vivo viscosity over D^4, or as
    the in vitro viscosity over (D - 1.1)^4, and the two must agree -- that is
    what says the surface-layer factors are a diameter correction and not a
    property of blood. They agree to 15%; the rest is that the two forms were
    fitted to different datasets.
    """
    for anatomical in (4.4, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0):
        channel = anatomical - PRIES_SINGULARITY_UM
        quoted_against_anatomical = (
            pries_in_vivo_viscosity(anatomical) / anatomical**4
        )
        quoted_against_channel = pries_in_vitro_viscosity(channel) / channel**4
        assert quoted_against_anatomical == pytest.approx(
            quoted_against_channel, rel=0.15
        ), f"at {anatomical} um the two conventions disagree"


def test_the_default_basis_is_the_one_this_projects_images_produce():
    """HaemoLynx segments plasma-stained volumes: the diameter is the channel."""
    from haemolynx.pipeline import default_schema

    assert default_schema()["diameter_basis"].default == "plasma_column"
    assert PoiseuilleModel(40.0, 100.0).diameter_basis == "plasma_column"
    assert viscosity_for(5.0, law="pries") == pytest.approx(
        pries_in_vitro_viscosity(5.0)
    )


def test_an_unknown_basis_is_refused_by_name():
    with pytest.raises(ValueError, match="plasma_column"):
        viscosity_for(5.0, law="pries", diameter_basis="lumen")
    with pytest.raises(ValueError, match="diameter_basis"):
        PoiseuilleModel(40.0, 100.0, diameter_basis="lumen")


# ---------------------------------------------------------------------------
# Where the power law's placeholder does hold up: large vessels
# ---------------------------------------------------------------------------

#: Pries against the 3.5 mPa.s constant from 100 um up.
LARGE_VESSEL_TOLERANCE = 0.18


@pytest.mark.parametrize("diameter", LARGE_DIAMETERS_UM)
def test_pries_agrees_with_the_large_vessel_constant(diameter):
    """The placeholder is a good answer for big vessels, which is its defence."""
    pries = viscosity_for(diameter, law="pries")
    assert pries == pytest.approx(
        LARGE_VESSEL_VISCOSITY_PA_S, rel=LARGE_VESSEL_TOLERANCE
    ), f"at {diameter} um Pries says {pries * 1e3:.3f} mPa.s"


@pytest.mark.parametrize("diameter", LARGE_DIAMETERS_UM)
def test_a_large_vessel_resistance_is_the_same_either_way(diameter):
    """Same comparison through Poiseuille, where a run actually uses it.

    `capillary_power_law` returns the constant above 7 um, so this is the
    constant case against the new law on the same vessel.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderViscosityWarning)
        constant = _model("capillary_power_law").resistance_of_uniform_segment(
            SEGMENT_LENGTH_UM, diameter
        )
    pries = _model("pries").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, diameter
    )
    assert pries == pytest.approx(constant, rel=LARGE_VESSEL_TOLERANCE)


def test_a_large_vessel_network_solves_to_the_same_flow_either_way():
    """And end to end, which is what a user would compare."""
    constant = _flow_through_a_chain("capillary_power_law", 200.0)
    pries = _flow_through_a_chain("pries", 200.0)
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
        _model("pries").resistance_of_uniform_segment(SEGMENT_LENGTH_UM, d)
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


def test_the_default_law_is_pries_at_the_measured_diameter():
    """The default changed, deliberately, and the old law is still selectable.

    The power law was anchored at one point and unphysical above 8.7 um; the
    default is now the law that covers the tree, read at the diameter this
    project's imaging actually produces. Runs from before the change are not
    comparable, which is why the graph records the law.
    """
    from haemolynx.pipeline import default_schema

    assert default_schema()["viscosity_law"].default == "pries"
    assert PoiseuilleModel(40.0, 100.0).viscosity_law == "pries"
    assert "capillary_power_law" in default_schema()["viscosity_law"].choices


def test_an_unknown_law_is_refused_by_name():
    with pytest.raises(ValueError, match="pries"):
        PoiseuilleModel(40.0, 100.0, viscosity_law="pries_in_vivo")
    with pytest.raises(ValueError, match="Unknown viscosity_law"):
        viscosity_for(5.0, law="nonsense")


def test_only_the_law_with_a_placeholder_warns():
    """`pries_in_vivo` covers 7-100 um, so it has nothing to apologise for."""
    with pytest.warns(PlaceholderViscosityWarning):
        viscosity_for(20.0, law="capillary_power_law")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlaceholderViscosityWarning)
        viscosity_for(20.0, law="pries")


def test_each_law_states_the_range_it_was_fitted_over():
    assert validity_range_um("pries") == (
        PRIES_MIN_DIAMETER_UM,
        PRIES_MAX_DIAMETER_UM,
    )
    assert validity_range_um("capillary_power_law")[1] == CAPILLARY_REGIME_MAX_DIAMETER_UM


def test_the_run_records_which_law_and_which_basis_produced_its_numbers():
    """Comparable across neither, so both travel with the numbers."""
    description = _model("pries", haematocrit=0.4).describe_viscosity_law()
    assert "pries" in description and "in vitro" in description
    assert "plasma_column" in description
    assert "0.4" in description
    assert "3.3" in description and "1978" in description

    anatomical = _model("pries", diameter_basis="anatomical").describe_viscosity_law()
    assert "in vivo" in anatomical and "anatomical" in anatomical


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
            "viscosity_law": "pries",
            "diameter_basis": "anatomical",
            "haematocrit": 0.42,
        },
        fwhm={},
        resistance_node_pair=(0, 1),
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    G, summary = haemodynamics.apply_poiseuille_haemodynamics(G, config=config)

    restored = pickle.loads(pickle.dumps(G))
    assert restored.graph["viscosity_law"] == "pries"
    assert restored.graph["diameter_basis"] == "anatomical"
    assert restored.graph["haematocrit"] == pytest.approx(0.42)
    assert "in vivo" in summary["viscosity"]


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
                "viscosity_law": "pries", "haematocrit": 0.5},
    )
    loaded = load_config(path, schema)
    assert loaded["viscosity_law"] == "pries"
    assert loaded["haematocrit"] == pytest.approx(0.5)

    model = PoiseuilleModel(
        40.0, 100.0,
        viscosity_law=loaded["viscosity_law"],
        haematocrit=loaded["haematocrit"],
    )
    assert model.calculate_viscosity(5.0) == pytest.approx(
        pries_in_vitro_viscosity(5.0, haematocrit=0.5)
    )
