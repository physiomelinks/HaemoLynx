"""Physical-unit checks for the Poiseuille resistance model.

Resistance is in Pa.s/m^3 and conductance in m^3/(Pa.s), so a pressure drop in
Pa divided by a resistance gives a flow in m^3/s. These tests pin the absolute
scale against measured capillary physiology, and pin the scaling exponents so an
algebra error cannot hide inside a plausible-looking absolute value.
"""
import math

import networkx as nx
import pytest

from ImageLynx.graph.validate import assert_no_forbidden_edge_attributes
from ImageLynx.haemodynamics.poiseuille import (
    MAX_VALID_DIAMETER_UM,
    PLASMA_VISCOSITY_PA_S,
    REFERENCE_DIAMETER_UM,
    REFERENCE_VISCOSITY_PA_S,
    VISCOSITY_DIAMETER_EXPONENT,
    PoiseuilleModel,
    set_edge_resistance,
)

MODEL = PoiseuilleModel(constriction_length=40.0, constriction_spacing=100.0)

# A representative brain capillary.
CAPILLARY_DIAMETER_UM = 5.0
CAPILLARY_LENGTH_UM = 500.0
# ~5 mmHg across a single capillary.
CAPILLARY_PRESSURE_DROP_PA = 667.0

M3_PER_S_TO_NL_PER_MIN = 6.0e13


def _capillary_resistance() -> float:
    return MODEL.resistance_of_uniform_segment(CAPILLARY_LENGTH_UM, CAPILLARY_DIAMETER_UM)


# --- absolute correctness against measured physiology ----------------------


def test_single_capillary_resistance_is_physiological():
    """A 5 um x 500 um capillary should sit in the literature 1e16-1e17 Pa.s/m^3 band."""
    resistance = _capillary_resistance()
    assert 1e16 <= resistance <= 1e17, f"{resistance:.3e} Pa.s/m^3 outside physiological band"


def test_single_capillary_flow_is_physiological():
    """Flow at ~5 mmHg should be 0.1-1 nL/min, the measured range for a capillary."""
    flow_m3_s = CAPILLARY_PRESSURE_DROP_PA / _capillary_resistance()
    flow_nl_min = flow_m3_s * M3_PER_S_TO_NL_PER_MIN
    assert 0.1 <= flow_nl_min <= 1.0, f"{flow_nl_min:.3f} nL/min outside physiological range"


def test_single_capillary_velocity_is_physiological():
    """Mean red-cell velocity should be 0.1-1 mm/s."""
    flow_m3_s = CAPILLARY_PRESSURE_DROP_PA / _capillary_resistance()
    radius_m = (CAPILLARY_DIAMETER_UM / 2.0) * 1e-6
    velocity_mm_s = (flow_m3_s / (math.pi * radius_m**2)) * 1e3
    assert 0.1 <= velocity_mm_s <= 1.0, f"{velocity_mm_s:.3f} mm/s outside physiological range"


def test_conductance_is_exactly_the_reciprocal_of_resistance():
    edge: dict = {}
    set_edge_resistance(edge, _capillary_resistance())
    assert edge["conductance"] == pytest.approx(1.0 / edge["resistance"], rel=1e-15)
    assert set(edge) == {"resistance", "conductance"}


def test_set_edge_resistance_rejects_non_physical_values():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            set_edge_resistance({}, bad)


# --- scaling laws ----------------------------------------------------------


def test_resistance_is_linear_in_length():
    single = MODEL.resistance_of_uniform_segment(CAPILLARY_LENGTH_UM, CAPILLARY_DIAMETER_UM)
    double = MODEL.resistance_of_uniform_segment(2 * CAPILLARY_LENGTH_UM, CAPILLARY_DIAMETER_UM)
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_resistance_scales_as_diameter_to_the_minus_5_647():
    """d^-4 from Poiseuille composed with d^-1.647 from the viscosity law."""
    wide = MODEL.resistance_of_uniform_segment(CAPILLARY_LENGTH_UM, 6.0)
    narrow = MODEL.resistance_of_uniform_segment(CAPILLARY_LENGTH_UM, 3.0)
    expected_ratio = 2.0 ** (4.0 + VISCOSITY_DIAMETER_EXPONENT)
    assert narrow / wide == pytest.approx(expected_ratio, rel=1e-12)
    assert expected_ratio == pytest.approx(50.13, rel=1e-3)


def test_viscosity_matches_reference_point_exactly():
    assert MODEL.calculate_viscosity(REFERENCE_DIAMETER_UM) == pytest.approx(
        REFERENCE_VISCOSITY_PA_S, rel=1e-15
    )


def test_viscosity_rises_as_vessels_narrow():
    """Fahraeus-Lindqvist reversal: apparent viscosity increases below ~7 um."""
    viscosities = [MODEL.calculate_viscosity(d) for d in (3.0, 4.0, 5.0, 6.0, 7.0)]
    assert viscosities == sorted(viscosities, reverse=True)


# --- validity range --------------------------------------------------------


def test_diameter_above_seven_microns_is_rejected():
    with pytest.raises(ValueError, match="above-7um viscosity regime"):
        MODEL.calculate_viscosity(MAX_VALID_DIAMETER_UM + 0.001)


def test_diameter_at_the_limit_is_accepted():
    assert MODEL.calculate_viscosity(MAX_VALID_DIAMETER_UM) > 0


def test_validity_limit_sits_inside_the_physically_meaningful_region():
    """At the 7 um cutoff the model must still predict blood thicker than plasma."""
    assert MODEL.calculate_viscosity(MAX_VALID_DIAMETER_UM) > PLASMA_VISCOSITY_PA_S


def test_non_positive_diameter_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        MODEL.calculate_viscosity(0.0)


# --- regression: haemodynamics must not disturb geometry -------------------


def test_series_capillaries_add_resistance_and_preserve_length():
    """Two segments in series sum, and running haemodynamics leaves `length` untouched."""
    G = nx.MultiGraph()
    for u, v in ((0, 1), (1, 2)):
        G.add_edge(
            u,
            v,
            length=CAPILLARY_LENGTH_UM / 2.0,
            branch_order="B01",
            voxels=[[0.0, 0.0, 0.0], [0.0, 0.0, CAPILLARY_LENGTH_UM / 2.0]],
        )
    lengths_before = [d["length"] for _, _, d in G.edges(data=True)]

    G, results = MODEL.set_poiseuille_weights(G, {"B01": CAPILLARY_DIAMETER_UM})

    assert results["weights_set"] == 2
    lengths_after = [d["length"] for _, _, d in G.edges(data=True)]
    assert lengths_after == lengths_before

    total_series = sum(d["resistance"] for _, _, d in G.edges(data=True))
    whole = MODEL.resistance_of_uniform_segment(CAPILLARY_LENGTH_UM, CAPILLARY_DIAMETER_UM)
    assert total_series == pytest.approx(whole, rel=1e-12)


def test_haemodynamics_does_not_write_the_removed_weight_attribute():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=CAPILLARY_LENGTH_UM, branch_order="B01")
    G, _ = MODEL.set_poiseuille_weights(G, {"B01": CAPILLARY_DIAMETER_UM})
    assert_no_forbidden_edge_attributes(G, context="poiseuille output")


def test_forbidden_weight_attribute_raises_with_actionable_message():
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=10.0, weight=10.0)
    with pytest.raises(ValueError, match="removed 'weight' attribute"):
        assert_no_forbidden_edge_attributes(G, context="unit test")
