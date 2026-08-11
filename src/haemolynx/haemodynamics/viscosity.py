"""Apparent blood viscosity: the available laws, and what each is good for.

Blood is not a Newtonian fluid at these scales. Its apparent viscosity depends
on the vessel it is flowing through -- the Fahraeus-Lindqvist effect -- so a
resistance model needs a diameter-dependent viscosity, and which law supplies
it changes every resistance a run produces. The laws therefore live together
here, each declaring the range it is valid over, and the run records which one
it used.

The three:

``capillary_power_law``
    A power law pinned to 3.0 mPa.s at 5 um, with a constant 3.5 mPa.s above
    7 um. Calibrated for capillaries, and the constant branch is an admitted
    placeholder: it steps up discontinuously at 7 um, so a 7.5 um vessel is
    predicted *more* resistive per unit length than a 7 um one.

``pries_in_vivo``
    Pries et al., valid 3.3-1978 um, so one expression covers the whole tree
    with no discontinuity. Includes the endothelial surface layer, which is
    why it is several times the in-vitro value in the smallest vessels.

``constant``
    Plasma viscosity everywhere. For isolating whether a difference comes from
    the geometry or from the viscosity model.

They do not agree, and the disagreement is not small -- see
``tests/test_viscosity_laws.py``, which measures it rather than asserting it
away. Resistances are not comparable across laws.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Every law that can be selected.
VISCOSITY_LAWS: tuple[str, ...] = (
    "capillary_power_law",
    "pries_in_vivo",
    "constant",
)

# --- the capillary power law ------------------------------------------------
#
#     mu(d) = REFERENCE_VISCOSITY_PA_S * (REFERENCE_DIAMETER_UM / d) ** EXPONENT
#
# Only the absolute scale depends on the reference point; every resistance
# *ratio* within the law is fixed by the exponent alone.
REFERENCE_VISCOSITY_PA_S = 3.0e-3
REFERENCE_DIAMETER_UM = 5.0
VISCOSITY_DIAMETER_EXPONENT = 1.647

#: Plasma viscosity. The power law crosses this at ~8.7 um, above which it
#: would predict blood thinner than its own plasma, so it is only used below.
PLASMA_VISCOSITY_PA_S = 1.2e-3
CAPILLARY_REGIME_MAX_DIAMETER_UM = 7.0

#: Whole blood at physiological haematocrit in vessels wide enough
#: (d >~ 300 um) that the Fahraeus-Lindqvist effect has died out.
LARGE_VESSEL_VISCOSITY_PA_S = 3.5e-3

#: Upper end of the placeholder's error: by ~100 um the constant is close to
#: the true macroscale value, and below it the placeholder is poor.
PLACEHOLDER_REGIME_MAX_DIAMETER_UM = 100.0

# --- Pries et al. in vivo ---------------------------------------------------
#
#   mu_apparent(D, H) = PLASMA * mu_rel
#   mu_rel = [1 + (mu45 - 1) * f(H) * s] * s,      s = (D / (D - 1.1))^2
#   mu45   = 6*exp(-0.085 D) + 3.2 - 2.44*exp(-0.06 * D^0.645)
#   f(H)   = ((1 - H)^C - 1) / ((1 - 0.45)^C - 1)
#   C      = (0.8 + exp(-0.075 D)) * (-1 + 1/g) + 1/g,   g = 1 + 1e-11 * D^12
#
# `s` is the endothelial surface layer: the ~1.1 um of glycocalyx that flowing
# blood does not use, which is what makes the in vivo law so much higher than
# an in vitro tube measurement in the smallest vessels.
PRIES_MIN_DIAMETER_UM = 3.3
PRIES_MAX_DIAMETER_UM = 1978.0

#: Discharge haematocrit the law is written around; f(H) is 1 here by
#: construction, so this is the value at which mu45 *is* the relative viscosity.
DEFAULT_HAEMATOCRIT = 0.45

#: Diameter below which the surface-layer term (D / (D - 1.1))^2 runs away --
#: it is singular at 1.1 um. Well below the law's stated floor, and only here
#: so the failure is a message rather than a division by zero.
PRIES_SINGULARITY_UM = 1.1


class PlaceholderViscosityWarning(UserWarning):
    """A vessel fell in the 7-100 um regime where viscosity is a placeholder.

    Only ``capillary_power_law`` raises this: it is the law with a placeholder
    branch. ``pries_in_vivo`` models the transition, which is the point of
    having it, so it warns about nothing.

    Silence it deliberately, once you have accepted the approximation, with::

        warnings.filterwarnings("ignore", category=PlaceholderViscosityWarning)
    """


def capillary_power_law_viscosity(diameter: float) -> float:
    """The calibrated capillary law, with a constant placeholder above 7 um."""
    if diameter > CAPILLARY_REGIME_MAX_DIAMETER_UM:
        if diameter <= PLACEHOLDER_REGIME_MAX_DIAMETER_UM:
            # Deliberately free of the actual diameter so repeated calls
            # collapse to one message under the default warning filter.
            warnings.warn(
                "Vessel diameters between "
                f"{CAPILLARY_REGIME_MAX_DIAMETER_UM} and "
                f"{PLACEHOLDER_REGIME_MAX_DIAMETER_UM} um use a placeholder "
                "viscosity: it is held constant at "
                f"{LARGE_VESSEL_VISCOSITY_PA_S * 1e3} mPa.s instead of "
                "recovering gradually from the capillary law toward the "
                "macroscale value. The error is largest just above "
                f"{CAPILLARY_REGIME_MAX_DIAMETER_UM} um, where viscosity "
                "also steps up discontinuously. Treat resistances for these "
                "vessels as order-of-magnitude only — set viscosity_law to "
                "'pries_in_vivo' for a law that covers this range.",
                PlaceholderViscosityWarning,
                stacklevel=3,
            )
        return LARGE_VESSEL_VISCOSITY_PA_S
    return REFERENCE_VISCOSITY_PA_S * (
        (REFERENCE_DIAMETER_UM / diameter) ** VISCOSITY_DIAMETER_EXPONENT
    )


def pries_in_vivo_viscosity(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> float:
    """Pries et al. in vivo apparent viscosity, in Pa.s.

    Valid over 3.3-1978 um. Outside that the expression still evaluates, and
    is still monotonic and above plasma, so it is not clamped -- but it is
    extrapolation, and `describe_law` says where the ends are.
    """
    if diameter <= PRIES_SINGULARITY_UM:
        raise ValueError(
            f"Diameter must exceed {PRIES_SINGULARITY_UM} um for the Pries in "
            f"vivo law -- the endothelial surface layer term is singular "
            f"there -- got {diameter} um. Fix: this vessel is narrower than "
            "any the law was fitted to; use 'capillary_power_law', or check "
            "the diameter."
        )
    if not 0.0 <= haematocrit < 1.0:
        raise ValueError(
            f"haematocrit is a fraction in [0, 1), got {haematocrit}."
        )

    d = float(diameter)
    # Relative apparent viscosity of blood at H = 0.45.
    mu45 = 6.0 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    # The haematocrit exponent, which tends to 1 in large vessels.
    g = 1.0 + 1e-11 * d**12
    shape = (0.8 + np.exp(-0.075 * d)) * (-1.0 + 1.0 / g) + 1.0 / g
    # How the law scales away from H = 0.45; exactly 1 at 0.45.
    numerator = (1.0 - haematocrit) ** shape - 1.0
    denominator = (1.0 - DEFAULT_HAEMATOCRIT) ** shape - 1.0
    haematocrit_factor = numerator / denominator
    # The endothelial surface layer: blood flows through D - 1.1, not D.
    surface_layer = (d / (d - PRIES_SINGULARITY_UM)) ** 2

    relative = (
        1.0 + (mu45 - 1.0) * haematocrit_factor * surface_layer
    ) * surface_layer
    return PLASMA_VISCOSITY_PA_S * float(relative)


def constant_viscosity(diameter: float) -> float:
    """Plasma viscosity, whatever the diameter."""
    del diameter
    return PLASMA_VISCOSITY_PA_S


def viscosity_for(
    diameter: float,
    *,
    law: str = "capillary_power_law",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
) -> float:
    """Apparent blood viscosity in Pa.s, by the named law."""
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    if law == "capillary_power_law":
        return capillary_power_law_viscosity(diameter)
    if law == "pries_in_vivo":
        return pries_in_vivo_viscosity(diameter, haematocrit)
    if law == "constant":
        return constant_viscosity(diameter)
    raise ValueError(
        f"Unknown viscosity_law {law!r}. Fix: choose one of "
        f"{', '.join(VISCOSITY_LAWS)}."
    )


def validity_range_um(law: str) -> tuple[float, float]:
    """The diameters *law* is fitted over, as (smallest, largest) in um."""
    if law == "capillary_power_law":
        # Above 7 um it is the placeholder, which warns for itself.
        return (0.0, CAPILLARY_REGIME_MAX_DIAMETER_UM)
    if law == "pries_in_vivo":
        return (PRIES_MIN_DIAMETER_UM, PRIES_MAX_DIAMETER_UM)
    if law == "constant":
        return (0.0, float("inf"))
    raise ValueError(f"Unknown viscosity_law {law!r}.")


def describe_law(law: str, haematocrit: float = DEFAULT_HAEMATOCRIT) -> str:
    """One line naming the law and its range, for a run's metadata.

    Resistances are not comparable across laws, so which one produced a set of
    numbers has to travel with them.
    """
    low, high = validity_range_um(law)
    span = f"{low:g}-{high:g} um" if np.isfinite(high) else "any diameter"
    if law == "pries_in_vivo":
        return f"{law} (fitted {span}, discharge haematocrit {haematocrit:g})"
    return f"{law} (fitted {span})"
