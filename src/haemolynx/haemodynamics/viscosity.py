"""Apparent blood viscosity: the available laws, and which diameter they expect.

Blood is not Newtonian at these scales -- its apparent viscosity depends on the
vessel it is flowing through, the Fahraeus-Lindqvist effect -- so a resistance
model needs a diameter-dependent viscosity, and the choice changes every
resistance a run produces.

The question that decides the answer is not "which law" but **which diameter
did you measure**, and getting it wrong is a silent factor of five.

``pries`` with ``diameter_basis="plasma_column"`` (the default)
    Pries et al.'s *in vitro* tube law, at the diameter as measured. Correct
    when the segmented diameter is the channel the fluid occupies -- a plasma
    stain, a fluorescent-dextran column, anything that images the lumen the
    blood is in. This is what HaemoLynx's own imaging produces.

``pries`` with ``diameter_basis="anatomical"``
    Pries et al.'s *in vivo* law, at the diameter as measured. Correct when the
    diameter runs wall to wall -- brightfield intravital microscopy, a cast, an
    endothelial stain -- and therefore includes the endothelial surface layer,
    the ~1.1 um of glycocalyx that flowing blood does not use.

    The in vivo law's ``(D / (D - 1.1))^2`` factors are exactly that correction:
    they appear squared, so the leading term carries ``(D / (D - 1.1))^4``,
    which is the Poiseuille factor for referencing a resistance to a diameter
    wider than the channel it flowed through. Applying them to a plasma-column
    diameter subtracts the glycocalyx twice, and costs a factor of about five
    in a capillary. `tests/test_viscosity_laws.py` checks the two conventions
    against each other and finds them consistent to within 15%.

``capillary_power_law``
    The law this project used before: pinned to 3.0 mPa.s at 5 um, with a
    constant 3.5 mPa.s above 7 um. Kept for comparison with earlier results.
    It agrees with Pries in vitro to 2% at 3 um and diverges from there, which
    is the signature of a one-point calibration: its ``d^-1.647`` slope is too
    steep, and it crosses below plasma viscosity at 8.7 um, which is what the
    7 um guard and its placeholder branch exist to hide.

``constant``
    Plasma viscosity everywhere, for separating a geometry effect from a
    viscosity one.

Resistances are not comparable across laws or across bases, so a run records
both on the graph.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Every law that can be selected.
VISCOSITY_LAWS: tuple[str, ...] = ("pries", "capillary_power_law", "constant")

#: What a measured diameter means. `plasma_column` is the channel the fluid
#: occupies; `anatomical` runs wall to wall and so includes the surface layer.
DIAMETER_BASES: tuple[str, ...] = ("plasma_column", "anatomical")

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
    branch. ``pries`` models the transition, which is the point of having it,
    so it warns about nothing.

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
                "'pries' for a law that covers this range.",
                PlaceholderViscosityWarning,
                stacklevel=3,
            )
        return LARGE_VESSEL_VISCOSITY_PA_S
    return REFERENCE_VISCOSITY_PA_S * (
        (REFERENCE_DIAMETER_UM / diameter) ** VISCOSITY_DIAMETER_EXPONENT
    )


def _pries_relative_terms(
    diameter: float, haematocrit: float
) -> tuple[float, float]:
    """``(mu45, f(H))`` -- the two terms both Pries forms are built from."""
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
    return float(mu45), float(numerator / denominator)


def pries_in_vitro_viscosity(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> float:
    """Pries et al. in vitro (tube) apparent viscosity, in Pa.s.

    Blood flowing through a channel of exactly this diameter, as measured in
    glass tubes where there is nothing between the fluid and the wall. Use it
    when the diameter you have is the channel -- a plasma-stained lumen -- and
    the surface layer is therefore already excluded by the measurement.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    mu45, haematocrit_factor = _pries_relative_terms(diameter, haematocrit)
    return PLASMA_VISCOSITY_PA_S * (1.0 + (mu45 - 1.0) * haematocrit_factor)


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
            "the surface layer itself, so an anatomical diameter cannot be "
            "right; check the diameter, or use diameter_basis='plasma_column' "
            "if it is already the channel."
        )
    d = float(diameter)
    mu45, haematocrit_factor = _pries_relative_terms(d, haematocrit)
    # The endothelial surface layer: blood flows through D - 1.1, not D. It
    # appears twice, so the leading term carries (D / (D - 1.1))^4 -- the
    # Poiseuille factor for quoting a resistance against a diameter wider than
    # the channel the blood went through.
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
    law: str = "pries",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
) -> float:
    """Apparent blood viscosity in Pa.s, by the named law.

    ``diameter_basis`` says what *diameter* is, and only ``pries`` reads it:
    ``plasma_column`` for the channel the fluid occupies, ``anatomical`` for a
    wall-to-wall measurement that includes the endothelial surface layer. The
    two differ by about five times in a capillary, so this is not a detail.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    if law == "pries":
        if diameter_basis == "plasma_column":
            return pries_in_vitro_viscosity(diameter, haematocrit)
        if diameter_basis == "anatomical":
            return pries_in_vivo_viscosity(diameter, haematocrit)
        raise ValueError(
            f"Unknown diameter_basis {diameter_basis!r}. Fix: choose one of "
            f"{', '.join(DIAMETER_BASES)}."
        )
    if law == "capillary_power_law":
        return capillary_power_law_viscosity(diameter)
    if law == "constant":
        return constant_viscosity(diameter)
    raise ValueError(
        f"Unknown viscosity_law {law!r}. Fix: choose one of "
        f"{', '.join(VISCOSITY_LAWS)}."
    )


def validity_range_um(law: str) -> tuple[float, float]:
    """The diameters *law* is fitted over, as (smallest, largest) in um."""
    if law == "pries":
        return (PRIES_MIN_DIAMETER_UM, PRIES_MAX_DIAMETER_UM)
    if law == "capillary_power_law":
        # Above 7 um it is the placeholder, which warns for itself.
        return (0.0, CAPILLARY_REGIME_MAX_DIAMETER_UM)
    if law == "constant":
        return (0.0, float("inf"))
    raise ValueError(f"Unknown viscosity_law {law!r}.")


def describe_law(
    law: str,
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
) -> str:
    """One line naming the law, its basis and its range, for a run's metadata.

    Resistances are comparable across neither, so which produced a set of
    numbers has to travel with them.
    """
    low, high = validity_range_um(law)
    span = f"{low:g}-{high:g} um" if np.isfinite(high) else "any diameter"
    if law == "pries":
        form = "in vitro" if diameter_basis == "plasma_column" else "in vivo"
        return (
            f"{law} {form} ({diameter_basis} diameters, fitted {span}, "
            f"discharge haematocrit {haematocrit:g})"
        )
    return f"{law} (fitted {span})"
