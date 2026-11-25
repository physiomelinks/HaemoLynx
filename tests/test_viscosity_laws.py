"""The viscosity laws, the diameter each expects, and how far apart they are.

Issue #85 asked for a test that the capillary power law and Pries agree below
7 um, and said what to do if they did not: "If they do not agree, that is the
finding." They do not: the power law matches neither of Pries's forms.

Pries published two forms. The *in vitro* one (Pries, Neuhaus and Gaehtgens
1992) is blood in a glass tube of a known bore, so its diameter is the channel
the fluid occupies. The *in vivo* one (Pries et al. 1994) was fitted to flow in
living networks against diameters measured wall to wall, which include the
endothelial surface layer; and Pries and Secomb's 2005 model accounts for the
difference as that layer, its thickness depending on diameter and
haematocrit. Each form is pinned here to its equations as published (Pries and
Secomb 2005, eqs 5-11; Secomb and Pries 2013, eqs 3-7) and to what those papers
say the laws do.

HaemoLynx segments plasma-stained images, so its diameters are the channel and
the default reads them through the in vitro form. Reading them through the in
vivo form instead makes a capillary six to nine times as viscous, and a test
pins that too, because it is the mistake the diameter basis exists to prevent.

Until 2026-10-13 the in vitro form carried the in vivo law's ``mu45``; the
tests here pinned that hybrid's ratios, and a regression test now pins it gone.
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
    PRIES_FORM_CHOICES,
    PRIES_FORMS,
    PRIES_IN_VITRO_1992,
    PRIES_IN_VIVO_1994,
    PRIES_IN_VIVO_2005,
    PRIES_MAX_DIAMETER_UM,
    PRIES_MIN_DIAMETER_UM,
    PlaceholderViscosityWarning,
    pries_in_vitro_viscosity,
    pries_in_vivo_viscosity,
    pries_secomb_2005_layer_um,
    pries_secomb_2005_viscosity,
    run_viscosity_law,
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

#: In vitro over the power law across 3.3-7 um. The power law is the more
#: viscous throughout -- 1.2x at 3.3 um, 1.7x at 5 um, 1.15x at 7 um -- having
#: been pinned at 3.0 mPa.s at 5 um, where the in vitro law says 1.77. Its
#: d^-1.647 slope then crosses below plasma at 8.7 um, which is what its 7 um
#: guard exists to hide.
CAPILLARY_AGREEMENT = {3.3: 0.822, 4.0: 0.639, 5.0: 0.588, 6.0: 0.688, 7.0: 0.869}


def test_the_power_law_matches_neither_pries_form():
    """What #85 asked: they do not agree, and that is the finding.

    Over 3.3-7 um the power law runs 1.1-1.7x above the in vitro law and at
    about a fifth of the in vivo one, so it was calibrated to neither.
    """
    for diameter in CAPILLARY_DIAMETERS_UM:
        power = viscosity_for(diameter, law="capillary_power_law")
        assert 1.1 < power / pries_in_vitro_viscosity(diameter) < 1.75
        assert 0.15 < power / pries_in_vivo_viscosity(diameter) < 0.22


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
    """The viscosity ratio at 5 um is the flow ratio, end to end."""
    power = _flow_through_a_chain("capillary_power_law", 5.0)
    pries = _flow_through_a_chain("pries", 5.0)
    assert power / pries == pytest.approx(CAPILLARY_AGREEMENT[5.0], rel=0.02)


# ---------------------------------------------------------------------------
# The mistake the diameter basis exists to prevent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diameter", CAPILLARY_DIAMETERS_UM)
def test_reading_a_plasma_column_as_endothelial_costs_a_factor_of_several(diameter):
    """Subtracting the surface layer from a diameter that never included it.

    A plasma stain images the channel the fluid occupies, so the ~1.1 um of
    glycocalyx is already out of the measurement. Calling that diameter
    endothelial (wall to wall) takes it out again, and in a capillary there is
    little else to take.
    """
    as_measured = viscosity_for(diameter, law="pries")
    double_counted = viscosity_for(
        diameter, law="pries", diameter_basis="endothelial"
    )
    assert 5.5 < double_counted / as_measured < 9.0


def test_anatomical_is_the_old_name_for_endothelial():
    """Graphs and edges saved before the rename still read as they did."""
    from haemolynx.haemodynamics.viscosity import normalise_diameter_basis

    assert normalise_diameter_basis("anatomical") == "endothelial"
    assert viscosity_for(5.0, law="pries", diameter_basis="anatomical") == viscosity_for(
        5.0, law="pries", diameter_basis="endothelial"
    )
    assert PoiseuilleModel(40.0, 100.0, diameter_basis="anatomical").diameter_basis == "endothelial"


# ---------------------------------------------------------------------------
# The laws as published
# ---------------------------------------------------------------------------
#
# Written out again from the papers rather than imported, so a slip in the
# module cannot be copied into its own check: Pries and Secomb 2005 (Am J
# Physiol Heart Circ Physiol 289:H2657), eqs 5-11, and Secomb and Pries 2013
# (C R Physique 14:470), eqs 3-7. Relative viscosity, times plasma's.


def _published_c(d):
    g = 1 + 1e-11 * d**12
    return (0.8 + np.exp(-0.075 * d)) * (-1 + 1 / g) + 1 / g


def _published_f(d, h):
    c = _published_c(d)
    return ((1 - h) ** c - 1) / ((1 - 0.45) ** c - 1)


def _published_in_vitro(d, h):
    eta45 = 220 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    return 1 + (eta45 - 1) * _published_f(d, h)


def _published_in_vivo_1994(d, h):
    eta45 = 6 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    s = (d / (d - 1.1)) ** 2
    return (1 + (eta45 - 1) * _published_f(d, h) * s) * s


def _published_layers_2005(d, h):
    d_off, d_crit, d_50, w_max = 2.4, 10.5, 100.0, 2.6
    e_amp, e_width, e_peak, e_hd = 1.1, 0.03, 0.6, 1.18
    if d <= d_off:
        return 0.0, 0.0
    w_as = (d - d_off) / (d + d_50 - 2 * d_off) * w_max
    if d <= d_crit:
        w_peak = e_amp * (d - d_off) / (d_crit - d_off)
    else:
        w_peak = e_amp * np.exp(-e_width * (d - d_crit))
    return w_as + w_peak * e_peak, w_as + w_peak * (1 + h * e_hd)


def _published_in_vivo_2005(d, h):
    w_ph, w_eff = _published_layers_2005(d, h)
    return _published_in_vitro(d - 2 * w_ph, h) * (d / (d - 2 * w_eff)) ** 4


PUBLISHED_FORMS = (
    (pries_in_vitro_viscosity, _published_in_vitro),
    (pries_in_vivo_viscosity, _published_in_vivo_1994),
    (pries_secomb_2005_viscosity, _published_in_vivo_2005),
)


@pytest.mark.parametrize("haematocrit", [0.0, 0.2, 0.45, 0.6])
@pytest.mark.parametrize(
    "diameter", [2.0, 3.3, 5.0, 7.0, 10.0, 10.5, 12.0, 30.0, 100.0, 500.0, 1978.0]
)
def test_each_form_is_its_published_equation(diameter, haematocrit):
    for law, published in PUBLISHED_FORMS:
        assert law(diameter, haematocrit) / PLASMA_VISCOSITY_PA_S == pytest.approx(
            published(diameter, haematocrit), rel=1e-12
        ), law.__name__


def test_the_in_vitro_form_is_not_the_in_vivo_base():
    """Regression: the in vitro form used the 1994 law's mu45, 6 exp(-0.085 D),
    where Pries 1992 has 220 exp(-1.3 D): 5.06 times plasma at 5 um and
    H = 0.45, against the published 1.47."""
    relative = pries_in_vitro_viscosity(5.0) / PLASMA_VISCOSITY_PA_S
    assert relative == pytest.approx(1.471, abs=0.001)
    hybrid = 6 * np.exp(-0.085 * 5.0) + 3.2 - 2.44 * np.exp(-0.06 * 5.0**0.645)
    assert relative < hybrid / 3


def test_the_in_vitro_minimum_is_where_the_papers_put_it():
    """'A distinctive minimum at 6-7 um' (Pries and Secomb 2005, abstract),
    about 1.25 times plasma there and about 3.2 in tubes above 1 mm (their
    introduction, from Pries 1992)."""
    diameters = np.linspace(PRIES_MIN_DIAMETER_UM, 100.0, 4000)
    relative = np.array([pries_in_vitro_viscosity(d) for d in diameters])
    relative /= PLASMA_VISCOSITY_PA_S
    lowest = int(np.argmin(relative))
    assert 6.0 <= diameters[lowest] <= 7.0
    assert relative[lowest] == pytest.approx(1.25, abs=0.03)
    assert pries_in_vitro_viscosity(1000.0) / PLASMA_VISCOSITY_PA_S == pytest.approx(
        3.2, rel=0.01
    )


def test_the_2005_layer_is_about_a_micron_from_10_to_40_um():
    """'A layer thickness of ~0.8-1 um for 10- to 40-um-diameter vessels and
    declined strongly for smaller diameters' (their abstract): the physical
    layer, W_ph."""
    for diameter in np.linspace(10.0, 40.0, 61):
        physical, _effective = pries_secomb_2005_layer_um(diameter)
        assert 0.78 <= physical <= 1.0, diameter
    assert pries_secomb_2005_layer_um(5.0)[0] < 0.4


def test_the_2005_layer_vanishes_below_d_off_and_tends_to_w_max():
    assert pries_secomb_2005_layer_um(2.4) == (0.0, 0.0)
    assert pries_secomb_2005_layer_um(2.0, haematocrit=0.6) == (0.0, 0.0)
    physical, effective = pries_secomb_2005_layer_um(1e6)
    assert physical == pytest.approx(2.6, rel=1e-3)
    assert effective == pytest.approx(2.6, rel=1e-3)


def test_the_2005_effective_layer_peaks_at_10_5_um_and_rises_with_haematocrit():
    """'A marked peak at a vessel diameter just above 10 um (10.5 um), the
    height of which increases with increasing HD' (their Fig. 2)."""
    diameters = np.linspace(3.0, 40.0, 3701)
    for haematocrit in (0.2, 0.45, 0.6):
        effective = [pries_secomb_2005_layer_um(d, haematocrit)[1] for d in diameters]
        assert diameters[int(np.argmax(effective))] == pytest.approx(10.5, abs=0.011)
    peaks = [pries_secomb_2005_layer_um(10.5, h)[1] for h in (0.2, 0.45, 0.6)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_the_2005_viscosity_has_the_shape_the_paper_describes():
    """Effective viscosity falls as the diameter falls below ~10 um (their
    Discussion) and its minimum moves out to ~30-50 um (their Fig. 4).
    Resistance still rises as diameter falls: see
    test_a_wider_vessel_is_never_more_resistive_under_pries."""
    near = np.linspace(6.0, 30.0, 2401)
    values = [pries_secomb_2005_viscosity(d) for d in near]
    assert near[int(np.argmax(values))] == pytest.approx(10.5, abs=0.011)
    wide = np.linspace(15.0, 400.0, 3851)
    values = [pries_secomb_2005_viscosity(d) for d in wide]
    assert 30.0 <= wide[int(np.argmin(values))] <= 50.0


def test_the_default_basis_is_the_one_this_projects_images_produce():
    """HaemoLynx segments plasma-stained volumes: the diameter is the channel.

    The run's basis is what the segmented vessels are -- `segmentation_source`
    on the Input tab, beside the segmented image, a plasma label unless said
    otherwise -- and a vessel measured from the raw image carries the basis
    that measurement read. `diameter_basis` is no longer a setting of its own:
    a config still naming it reads into the segmentation source."""
    from haemolynx.pipeline import default_schema

    schema = default_schema()
    assert "diameter_basis" not in schema
    source = schema["segmentation_source"]
    assert source.section == "Input and segmentation"
    assert source.default == "plasma_column"
    assert tuple(source.choices) == ("plasma_column", "endothelial")
    assert schema.upgrade({"diameter_basis": "anatomical"}) == {"segmentation_source": "endothelial"}
    assert schema.upgrade({"diameter_basis": "plasma_column"}) == {"segmentation_source": "plasma_column"}
    assert schema.upgrade({"segmentation_source": "anatomical"}) == {"segmentation_source": "endothelial"}
    assert PoiseuilleModel(40.0, 100.0).diameter_basis == "plasma_column"
    assert viscosity_for(5.0, law="pries") == pytest.approx(
        pries_in_vitro_viscosity(5.0)
    )


def test_the_runs_basis_is_what_the_segmented_vessels_are():
    """Every reader of the run's basis -- the stage's apply config, a
    perturbation's blood model, the sweeps' strategy arguments -- takes it
    from the segmentation source, through one helper, so a mask segmented
    from an endothelial stain is modelled wall to wall everywhere."""
    from haemolynx.haemodynamics import run_diameter_basis
    from haemolynx.haemodynamics.constriction_strategy import constriction_strategy_kwargs
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import _poiseuille_model_for

    assert run_diameter_basis({}) == "plasma_column"
    assert run_diameter_basis({"segmentation_source": "endothelial"}) == "endothelial"
    assert run_diameter_basis({"segmentation_source": "anatomical"}) == "endothelial"
    # An apply config's diameters section, or a run written before the choice
    # moved to the Input tab, still names it diameter_basis.
    assert run_diameter_basis({"diameter_basis": "anatomical"}) == "endothelial"
    assert run_diameter_basis({"segmentation_source": "plasma_column", "diameter_basis": "endothelial"}) == "plasma_column"

    settings = {setting.name: setting.default for setting in default_schema()}
    settings["segmentation_source"] = "endothelial"
    assert _poiseuille_model_for(settings).diameter_basis == "endothelial"
    kwargs = constriction_strategy_kwargs(settings, diameter_by_branch_order={"B01": 6.0})
    assert kwargs["diameter_basis"] == "endothelial"
    settings["segmentation_source"] = "plasma_column"
    assert _poiseuille_model_for(settings).diameter_basis == "plasma_column"


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


@pytest.mark.parametrize("law", [form for form, _published in PUBLISHED_FORMS])
def test_pries_never_predicts_blood_thinner_than_plasma(law):
    """The failure that bounds the power law at 7 um, across Pries' whole range."""
    diameters = np.geomspace(PRIES_MIN_DIAMETER_UM, PRIES_MAX_DIAMETER_UM, 500)
    for diameter in diameters:
        assert law(diameter) > PLASMA_VISCOSITY_PA_S


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


@pytest.mark.parametrize("law", ["pries", *PRIES_FORMS])
@pytest.mark.parametrize("haematocrit", [0.2, 0.45, 0.6])
def test_a_wider_vessel_is_never_more_resistive_under_pries(law, haematocrit):
    """What the power law's step gets wrong, stated as the property that
    matters -- and, for the 2005 model, whose viscosity falls below 10 um,
    what its authors say of resistance: it 'continues to increase with
    decreasing diameter in this range'."""
    diameters = np.linspace(3.5, 300.0, 800)
    resistances = [
        _model(law, haematocrit=haematocrit).resistance_of_uniform_segment(SEGMENT_LENGTH_UM, d)
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


@pytest.mark.parametrize("law", [form for form, _published in PUBLISHED_FORMS])
@pytest.mark.parametrize("diameter", [5.0, 20.0, 100.0])
def test_thicker_blood_is_more_viscous(diameter, law):
    values = [law(diameter, haematocrit=h) for h in (0.2, 0.45, 0.6)]
    assert values[0] < values[1] < values[2]


@pytest.mark.parametrize("law", [form for form, _published in PUBLISHED_FORMS])
def test_haematocrit_outside_zero_to_one_is_refused(law):
    with pytest.raises(ValueError, match="fraction"):
        law(10.0, haematocrit=1.5)


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
    for law in ("pries", *PRIES_FORMS):
        assert validity_range_um(law) == (
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

    endothelial = _model("pries", diameter_basis="endothelial").describe_viscosity_law()
    assert "in vivo" in endothelial and "endothelial" in endothelial
    assert "Pries et al. 1994" in endothelial

    # A held form says which, and that it holds whatever the diameters are.
    for law, paper in (
        (PRIES_IN_VITRO_1992, "Pries et al. 1992"),
        (PRIES_IN_VIVO_1994, "Pries et al. 1994"),
        (PRIES_IN_VIVO_2005, "Pries and Secomb 2005"),
    ):
        description = _model(law).describe_viscosity_law()
        assert description.startswith(law) and paper in description
        assert "every vessel" in description and "plasma_column" in description
    assert "83 um" in _model(PRIES_IN_VIVO_2005).describe_viscosity_law()


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
            "diameter_basis": "endothelial",
            "haematocrit": 0.42,
        },
        fwhm={},
        resistance_node_pair=(0, 1),
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    G, summary = haemodynamics.apply_poiseuille_haemodynamics(G, config=config)

    restored = pickle.loads(pickle.dumps(G))
    assert restored.graph["viscosity_law"] == "pries"
    assert restored.graph["diameter_basis"] == "endothelial"
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


# --- per-edge discharge haematocrit override --------------------------------
#
# haemodynamics.haematocrit_distribution writes a per-edge
# "discharge_haematocrit" attribute; PoiseuilleModel reads it in preference
# to the uniform self.haematocrit wherever present. These pin that the
# override actually takes effect, and -- critically -- that its absence
# reproduces today's behaviour exactly, since that is what every existing
# run (the feature defaults off) must keep doing.


def _bare_chain_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, SEGMENT_LENGTH_UM]))
    G.add_edge(
        0, 1, branch_order="B01", length=SEGMENT_LENGTH_UM, diameter_um=8.0,
    )
    return G


def test_calculate_viscosity_override_wins_over_the_uniform_value():
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    assert model.calculate_viscosity(8.0, haematocrit=0.3) == pytest.approx(
        pries_in_vitro_viscosity(8.0, haematocrit=0.3)
    )
    assert model.calculate_viscosity(8.0, haematocrit=0.3) != pytest.approx(
        model.calculate_viscosity(8.0)
    )


def test_calculate_viscosity_with_no_override_keeps_the_uniform_value():
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    assert model.calculate_viscosity(8.0, haematocrit=None) == pytest.approx(
        model.calculate_viscosity(8.0)
    )


def test_set_poiseuille_resistances_reads_a_per_edge_discharge_haematocrit():
    """Regression: the resistance loop used to read only self.haematocrit,
    with no way for a per-edge value to reach it at all."""
    G = _bare_chain_graph()
    G[0][1][0]["discharge_haematocrit"] = 0.3
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)
    _, results = model.set_poiseuille_resistances(G, diameter_by_branch_order={})
    assert results["edges_set"] == 1
    expected = model.resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, 8.0, haematocrit=0.3
    )
    assert G[0][1][0]["resistance"] == pytest.approx(expected)
    # And it must differ from what the uniform 0.45 would have given --
    # otherwise the override was silently ignored.
    uniform = model.resistance_of_uniform_segment(SEGMENT_LENGTH_UM, 8.0)
    assert G[0][1][0]["resistance"] != pytest.approx(uniform)


def test_set_poiseuille_resistances_with_no_override_matches_the_pre_feature_baseline():
    """Critical backward-compatibility pin: an edge with no
    discharge_haematocrit attribute at all -- every run before this feature
    existed, and every run today with haematocrit_model left at its "fixed"
    default -- must get byte-identical resistances to the plain uniform path."""
    model = PoiseuilleModel(40.0, 100.0, viscosity_law="pries", haematocrit=0.45)

    with_no_attr = _bare_chain_graph()
    _, _ = model.set_poiseuille_resistances(with_no_attr, diameter_by_branch_order={})

    reference = _bare_chain_graph()
    expected = model.resistance_of_uniform_segment(SEGMENT_LENGTH_UM, 8.0)
    from haemolynx.haemodynamics.poiseuille import set_edge_resistance

    set_edge_resistance(reference[0][1][0], expected)

    assert with_no_attr[0][1][0]["resistance"] == pytest.approx(
        reference[0][1][0]["resistance"]
    )
    assert with_no_attr[0][1][0]["conductance"] == pytest.approx(
        reference[0][1][0]["conductance"]
    )


# ---------------------------------------------------------------------------
# Holding every vessel to one form: pries_viscosity_form
# ---------------------------------------------------------------------------


def test_auto_leaves_pries_choosing_and_a_form_holds_every_vessel():
    assert run_viscosity_law({}) == "pries"
    assert run_viscosity_law({"viscosity_law": "pries", "pries_viscosity_form": "auto"}) == "pries"
    held = {
        form: run_viscosity_law({"viscosity_law": "pries", "pries_viscosity_form": form})
        for form in PRIES_FORM_CHOICES
        if form != "auto"
    }
    assert held == {
        "in_vivo_1994": PRIES_IN_VIVO_1994,
        "in_vivo_2005": PRIES_IN_VIVO_2005,
        "in_vitro_1992": PRIES_IN_VITRO_1992,
    }
    # Only pries has forms: another law ignores the override.
    assert run_viscosity_law(
        {"viscosity_law": "constant", "pries_viscosity_form": "in_vivo_2005"}
    ) == "constant"
    with pytest.raises(ValueError, match="pries_viscosity_form"):
        run_viscosity_law({"viscosity_law": "pries", "pries_viscosity_form": "in_vivo_2010"})


@pytest.mark.parametrize(
    "law,published",
    [
        (PRIES_IN_VITRO_1992, _published_in_vitro),
        (PRIES_IN_VIVO_1994, _published_in_vivo_1994),
        (PRIES_IN_VIVO_2005, _published_in_vivo_2005),
    ],
)
def test_a_held_form_is_the_same_whatever_the_diameter_measures(law, published):
    for basis in ("plasma_column", "endothelial"):
        viscosity = viscosity_for(8.0, law=law, haematocrit=0.3, diameter_basis=basis)
        assert viscosity / PLASMA_VISCOSITY_PA_S == pytest.approx(published(8.0, 0.3))
        model = _model(law, haematocrit=0.3, diameter_basis=basis)
        assert model.calculate_viscosity(8.0) == pytest.approx(viscosity)


def test_a_held_form_overrides_a_vessels_own_basis_too():
    """A vessel measured wall to wall carries its own basis, which auto reads;
    held to in vitro, it is in vitro all the same."""
    from haemolynx.haemodynamics.poiseuille import EDGE_DIAMETER_BASIS

    def resistance(law: str) -> float:
        G = _bare_chain_graph()
        G[0][1][0][EDGE_DIAMETER_BASIS] = "endothelial"
        _model(law).set_poiseuille_resistances(G, diameter_by_branch_order={})
        return G[0][1][0]["resistance"]

    in_vitro = _model("pries").resistance_of_uniform_segment(SEGMENT_LENGTH_UM, 8.0)
    in_vivo = _model("pries", diameter_basis="endothelial").resistance_of_uniform_segment(
        SEGMENT_LENGTH_UM, 8.0
    )
    assert resistance("pries") == pytest.approx(in_vivo)
    assert resistance(PRIES_IN_VITRO_1992) == pytest.approx(in_vitro)
    assert in_vivo > 5 * in_vitro


def test_every_reader_of_the_settings_takes_the_held_form():
    """The stage's apply config, a perturbation's blood model and the sweeps'
    strategy arguments all read the law through run_viscosity_law, so the arms
    of a run agree on it."""
    from haemolynx.haemodynamics.constriction_strategy import constriction_strategy_kwargs
    from haemolynx.pipeline import default_schema
    from haemolynx.pipeline.stages import _haemodynamics_apply_config, _poiseuille_model_for

    schema = default_schema()
    settings = {setting.name: setting.default for setting in schema}
    settings["diameter_by_branch_order"] = {"B01": 6.0}
    assert _poiseuille_model_for(settings).viscosity_law == "pries"
    settings["pries_viscosity_form"] = "in_vivo_2005"
    assert _poiseuille_model_for(settings).viscosity_law == PRIES_IN_VIVO_2005
    kwargs = constriction_strategy_kwargs(settings, diameter_by_branch_order={"B01": 6.0})
    assert kwargs["viscosity_law"] == PRIES_IN_VIVO_2005
    config = _haemodynamics_apply_config(settings, schema, voxel_size_zyx=(1.0, 1.0, 1.0))
    assert config.viscosity_law == PRIES_IN_VIVO_2005
    settings["viscosity_law"] = "constant"
    assert _poiseuille_model_for(settings).viscosity_law == "constant"


def test_the_held_form_reaches_the_resistances_and_the_graph():
    """End to end through the apply layer: the resistance is the published
    2005 viscosity through Poiseuille, and the graph says which law made it."""
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.0, 0.0, SEGMENT_LENGTH_UM]))
    G.add_edge(0, 1, branch_order="B01", length=SEGMENT_LENGTH_UM,
               voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, SEGMENT_LENGTH_UM)])
    config = haemodynamics.HaemodynamicsApplyConfig(
        diameters={
            "diameter_by_branch_order": {"B01": 8.0},
            "viscosity_law": "pries",
            "pries_viscosity_form": "in_vivo_2005",
            "haematocrit": 0.45,
        },
        fwhm={},
        resistance_node_pair=(0, 1),
        voxel_size_zyx=(1.0, 1.0, 1.0),
    )
    G, summary = haemodynamics.apply_poiseuille_haemodynamics(G, config=config)

    assert G.graph["viscosity_law"] == PRIES_IN_VIVO_2005
    assert "Pries and Secomb 2005" in summary["viscosity"]
    viscosity = _published_in_vivo_2005(8.0, 0.45) * PLASMA_VISCOSITY_PA_S
    length_m, diameter_m = SEGMENT_LENGTH_UM * 1e-6, 8.0e-6
    poiseuille = 128.0 * viscosity * length_m / (np.pi * diameter_m**4)
    assert G[0][1][0]["resistance"] == pytest.approx(poiseuille, rel=1e-9)


def test_the_override_is_an_advanced_choice_under_the_pries_law():
    from haemolynx.haemodynamics.perturbations import INCOMPARABLE_OVERRIDES
    from haemolynx.pipeline import default_schema

    setting = default_schema()["pries_viscosity_form"]
    assert setting.default == "auto"
    assert tuple(setting.choices) == PRIES_FORM_CHOICES
    assert setting.advanced
    assert "viscosity_law=pries" in setting.requires
    assert setting.section == default_schema()["viscosity_law"].section
    # Like the law itself, a perturbation may not change it.
    assert "pries_viscosity_form" in INCOMPARABLE_OVERRIDES
