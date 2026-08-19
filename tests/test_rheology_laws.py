"""The two Pries-Secomb viscosity laws, kept distinct.

`calculate_pries_secomb_viscosity` combined the in vitro base relation with the in vivo
wall-layer correction, which is neither law. At D = 8 um and H = 0.45 the two bases differ by
about 3.4x in apparent viscosity, and resistance is linear in viscosity, so the choice is not
cosmetic.

In vitro is Pries et al. (1992) for glass tubes. In vivo is Pries et al. (1994) and adds the
endothelial surface layer, which is why it is larger and why it carries the wall correction.
"""
import numpy as np
import pytest

from ImageLynx.haemodynamics.rheology import calculate_pries_secomb_viscosity

MU_PLASMA = 1.2


def test_the_in_vivo_law_is_the_default():
    """These are living microvessels; the glass-tube relation is the special case."""
    default = calculate_pries_secomb_viscosity(8.0, 0.45)
    vivo = calculate_pries_secomb_viscosity(8.0, 0.45, law="in_vivo")
    assert default == pytest.approx(vivo)


def test_in_vivo_exceeds_in_vitro_across_the_microvascular_range():
    """The endothelial surface layer adds resistance the glass tube does not have."""
    for d in (4.0, 6.0, 8.0, 12.0, 20.0, 40.0):
        vivo = calculate_pries_secomb_viscosity(d, 0.45, law="in_vivo")
        vitro = calculate_pries_secomb_viscosity(d, 0.45, law="in_vitro")
        assert vivo > vitro, f"at D = {d} um"


def test_the_wall_correction_belongs_to_the_in_vivo_law_only():
    """A glass tube has no cell-depleted endothelial layer to correct for.

    With the correction removed the in vivo law must fall to something close to the in vitro
    one at large diameters, where the two base relations converge.
    """
    big = 100.0
    vivo = calculate_pries_secomb_viscosity(big, 0.45, law="in_vivo")
    vitro = calculate_pries_secomb_viscosity(big, 0.45, law="in_vitro")
    assert vivo / vitro < 1.15, "the laws should converge once the wall layer is negligible"


def test_the_two_laws_diverge_most_in_the_capillary_range():
    """Where the choice matters, and where every vessel in this study sits."""
    small = (calculate_pries_secomb_viscosity(5.0, 0.45, law="in_vivo")
             / calculate_pries_secomb_viscosity(5.0, 0.45, law="in_vitro"))
    large = (calculate_pries_secomb_viscosity(50.0, 0.45, law="in_vivo")
             / calculate_pries_secomb_viscosity(50.0, 0.45, law="in_vitro"))
    assert small > large


def test_plasma_viscosity_is_returned_when_there_are_no_cells():
    for law in ("in_vivo", "in_vitro"):
        assert calculate_pries_secomb_viscosity(8.0, 0.0, law=law) == pytest.approx(MU_PLASMA)
        assert calculate_pries_secomb_viscosity(0.0, 0.45, law=law) == pytest.approx(MU_PLASMA)


def test_viscosity_rises_with_haematocrit_under_both_laws():
    for law in ("in_vivo", "in_vitro"):
        vals = [calculate_pries_secomb_viscosity(8.0, h, law=law)
                for h in (0.1, 0.2, 0.3, 0.45, 0.6)]
        assert np.all(np.diff(vals) > 0), law


def test_only_the_in_vitro_law_shows_the_classic_fahraeus_lindqvist_minimum():
    """The two laws have genuinely different shapes, and that is the point of separating them.

    In a glass tube apparent viscosity falls as the tube narrows, reaching a minimum around
    6 to 8 um before rising steeply near red cell dimensions. In vivo the endothelial surface
    layer occupies a larger share of a narrow lumen, so apparent viscosity rises monotonically
    as diameter falls and the minimum in the capillary range disappears.
    """
    d = np.linspace(3.5, 60.0, 200)
    vitro = np.array([calculate_pries_secomb_viscosity(x, 0.45, law="in_vitro") for x in d])
    vivo = np.array([calculate_pries_secomb_viscosity(x, 0.45, law="in_vivo") for x in d])

    d_min_vitro = d[int(np.argmin(vitro))]
    assert 4.0 < d_min_vitro < 12.0, f"in vitro minimum at {d_min_vitro:.1f} um"

    capillary = d <= 20.0
    assert np.all(np.diff(vivo[capillary]) < 0), (
        "in vivo viscosity should fall monotonically with rising diameter up to 20 um")


def test_the_in_vivo_wall_factor_is_applied_twice():
    """The published relation applies (D/(D-1.1))^2 inside the bracket and again outside.

    Applying it once understates apparent viscosity by 1.26x at 8 um and 2.2x at 3 um, which
    is precisely the range every vessel in this study occupies.
    """
    d, h, mu_p = 8.0, 0.45, 1.2
    wall = (d / (d - 1.1)) ** 2
    mu45 = 6.0 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d ** 0.645)
    tail = 1.0 / (1.0 + 1e-11 * d ** 12)
    c = (0.8 + np.exp(-0.075 * d)) * (-1.0 + tail) + tail
    h_term = ((1.0 - h) ** c - 1.0) / ((1.0 - 0.45) ** c - 1.0)

    expected = (1.0 + (mu45 - 1.0) * h_term * wall) * wall * mu_p
    once = (1.0 + (mu45 - 1.0) * h_term) * wall * mu_p

    assert calculate_pries_secomb_viscosity(d, h, law="in_vivo") == pytest.approx(expected)
    assert not calculate_pries_secomb_viscosity(d, h, law="in_vivo") == pytest.approx(once)


def test_an_unknown_law_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="law"):
        calculate_pries_secomb_viscosity(8.0, 0.45, law="in_silico")


def test_the_small_diameter_cap_still_applies_under_both_laws():
    """Below about 3 um an RBC cannot pass, and the relation runs away."""
    for law in ("in_vivo", "in_vitro"):
        assert (calculate_pries_secomb_viscosity(1.0, 0.45, law=law)
                == pytest.approx(calculate_pries_secomb_viscosity(3.0, 0.45, law=law)))
