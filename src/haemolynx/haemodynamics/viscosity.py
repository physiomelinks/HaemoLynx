"""Apparent blood viscosity: the available laws, and which diameter they expect.

Blood is not Newtonian at these scales -- its apparent viscosity depends on the
vessel it is flowing through, the Fahraeus-Lindqvist effect -- so a resistance
model needs a diameter-dependent viscosity, and the choice changes every
resistance a run produces.

The question that decides the answer is not "which law" but **which diameter
did you measure**, and getting it wrong is a silent factor of four to nine in
a capillary.

``pries`` with ``diameter_basis="plasma_column"`` (the default)
    Pries, Neuhaus and Gaehtgens' (1992) *in vitro* tube law, at the diameter
    as measured: blood in a glass tube of that bore. Correct when the
    segmented diameter is the channel the fluid occupies -- a plasma stain, a
    fluorescent-dextran column, anything that images the lumen the blood is
    in. This is what HaemoLynx's own imaging produces.

``pries`` with ``diameter_basis="endothelial"`` (once ``"anatomical"``)
    Pries et al.'s (1994) *in vivo* law, at the diameter as measured: fitted
    to flow in rat mesenteric networks against diameters measured wall to wall
    by intravital microscopy. Correct when the diameter runs wall to wall --
    brightfield intravital microscopy, a cast, an endothelial stain -- and
    therefore includes the endothelial surface layer, the ~1.1 um of
    glycocalyx that flowing blood does not use. Its ``(D / (D - 1.1))^2``
    factors stand for that layer, and it is far above the in vitro law in the
    smallest vessels (8.6x at 5 um, 4.4x at 10 um), so reading a plasma-column
    diameter through it counts a layer the measurement already left out.

``pries_in_vitro_1992``, ``pries_in_vivo_1994``, ``pries_in_vivo_2005``
    One form of Pries's law for every vessel, whatever its diameter measures:
    the Haemodynamics tab's ``pries_viscosity_form`` (see
    :func:`run_viscosity_law`). ``pries_in_vivo_2005`` is Pries and Secomb's
    (2005) in vivo model, the newer of the two: the in vitro law at the
    diameter an endothelial surface layer leaves free, the layer's thickness
    depending on diameter and haematocrit. It fits network flows better than
    the 1994 law (Secomb and Pries 2013) and, like it, reads a diameter as
    wall to wall.

``capillary_power_law``
    The law this project used before: pinned to 3.0 mPa.s at 5 um, with a
    constant 3.5 mPa.s above 7 um, from no recorded source. Kept for
    comparison with earlier results. It matches neither Pries form -- 1.1-1.7x
    the in vitro law over 3-7 um, a fifth of the in vivo one -- and its
    ``d^-1.647`` slope crosses below plasma viscosity at 8.7 um, which is what
    the 7 um guard and its placeholder branch exist to hide.

``constant``
    Plasma viscosity everywhere, for separating a geometry effect from a
    viscosity one.

Resistances are not comparable across laws or across bases, so a run records
both on the graph.

Until 2026-10-13 the in vitro form used the in vivo law's ``mu45``
(``6 exp(-0.085 D)``) where Pries 1992 has ``220 exp(-1.3 D)``: 3.4x too
viscous at 5 um, 2.9x at 10 um, converging above 60 um. Plasma-column runs
from before then overstate capillary resistance by that much.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

#: Pries's laws held to one form for every vessel, whatever its diameter
#: measures, where ``pries`` chooses per vessel by its diameter basis.
PRIES_IN_VITRO_1992 = "pries_in_vitro_1992"
PRIES_IN_VIVO_1994 = "pries_in_vivo_1994"
PRIES_IN_VIVO_2005 = "pries_in_vivo_2005"
PRIES_FORMS: tuple[str, ...] = (PRIES_IN_VITRO_1992, PRIES_IN_VIVO_1994, PRIES_IN_VIVO_2005)

#: Every law a model can apply: ``viscosity_law``'s three, and the Pries forms
#: its override (``pries_viscosity_form``) holds a run to.
VISCOSITY_LAWS: tuple[str, ...] = ("pries", *PRIES_FORMS, "capillary_power_law", "constant")

#: The Haemodynamics tab's override of the form ``pries`` chooses.
PRIES_FORM_SETTING = "pries_viscosity_form"
#: Its values: ``auto`` leaves ``pries`` choosing per vessel (in vitro for a
#: plasma column, in vivo 1994 wall to wall); each other value names the form
#: (``pries_<value>``) every vessel is held to.
PRIES_FORM_CHOICES: tuple[str, ...] = ("auto", "in_vivo_1994", "in_vivo_2005", "in_vitro_1992")

#: What a measured diameter means. `plasma_column` is the channel the fluid
#: occupies, as a luminal (plasma) stain images it; `endothelial` runs wall to
#: wall, as an endothelial stain images it, and so includes the surface layer.
DIAMETER_BASES: tuple[str, ...] = ("plasma_column", "endothelial")
#: Former names of a basis: graphs and runs saved before still read.
DIAMETER_BASIS_ALIASES: dict[str, str] = {"anatomical": "endothelial"}


def normalise_diameter_basis(basis: str) -> str:
    """*basis* as :data:`DIAMETER_BASES` names it (``anatomical`` was the
    wall-to-wall basis's former name); anything else is returned as given."""
    return DIAMETER_BASIS_ALIASES.get(basis, basis) if isinstance(basis, str) else basis


#: The Input tab's setting that says what the segmented vessels are, and so
#: what a diameter taken from the segmentation measures: the run's basis.
SEGMENTATION_SOURCE_SETTING = "segmentation_source"


def run_diameter_basis(settings: Mapping[str, Any]) -> str:
    """The run's diameter basis from *settings*: what a diameter the
    segmentation gave measures, which every diameter not read from the raw
    image is on. ``segmentation_source`` (the Input tab's choice) when it is
    there, else ``diameter_basis`` (an apply config's diameters section, or a
    run written before the choice moved to the Input tab), else plasma_column
    -- the mask of a plasma label. A former spelling reads as the current."""
    for key in (SEGMENTATION_SOURCE_SETTING, "diameter_basis"):
        value = settings.get(key)
        if value is not None:
            return normalise_diameter_basis(value)
    return "plasma_column"


def run_viscosity_law(settings: Mapping[str, Any]) -> str:
    """The law a run's resistances are computed by, from *settings*:
    ``viscosity_law``, or -- when that is ``pries`` and ``pries_viscosity_form``
    names a form rather than ``auto`` -- that form (:data:`PRIES_FORMS`),
    which every vessel is then held to whatever its diameter measures. Read
    wherever settings become a blood model, so the baseline, the
    perturbations and the sweeps agree."""
    law = settings.get("viscosity_law") or "pries"
    form = settings.get(PRIES_FORM_SETTING) or "auto"
    if law != "pries" or form == "auto":
        return law
    held = f"pries_{form}"
    if held not in PRIES_FORMS:
        raise ValueError(
            f"Unknown {PRIES_FORM_SETTING} {form!r}. Fix: choose one of "
            f"{', '.join(PRIES_FORM_CHOICES)}."
        )
    return held

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

# --- Pries et al. 1992, in vitro --------------------------------------------
#
#   mu_apparent(D, H) = PLASMA * mu_rel
#   mu_rel = 1 + (mu45 - 1) * f(H)
#   mu45   = 220*exp(-1.3 D) + 3.2 - 2.44*exp(-0.06 * D^0.645)
#   f(H)   = ((1 - H)^C - 1) / ((1 - 0.45)^C - 1)
#   C      = (0.8 + exp(-0.075 D)) * (-1 + 1/g) + 1/g,   g = 1 + 1e-11 * D^12
#
# As restated in Pries & Secomb 2005 (eqs 5-7) and Secomb & Pries 2013
# (eqs 3-5). Its minimum, ~1.25 at H = 0.45, is at 6-7 um.
#
# --- Pries et al. 1994, in vivo ---------------------------------------------
#
#   mu_rel = [1 + (mu45* - 1) * f(H) * s] * s,     s = (D / (D - 1.1))^2
#   mu45*  = 6*exp(-0.085 D) + 3.2 - 2.44*exp(-0.06 * D^0.645)
#
# with f(H) and C as above (Secomb & Pries 2013 eqs 6-7; Secomb's NetFlow
# viscor.cpp, less its scaling of D by red-cell volume). `s` is the
# endothelial surface layer: the ~1.1 um of glycocalyx that flowing blood does
# not use, which with mu45* is what makes the in vivo law so much higher than
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

# --- Pries & Secomb 2005, in vivo with an endothelial surface layer ---------
#
#   mu_vivo = mu_vitro(D_ph, H) * (D / D_eff)^4             (their eqs 5-11)
#   D_ph  = D - 2 W_ph,    W_ph  = W_as + W_peak * E_peak
#   D_eff = D - 2 W_eff,   W_eff = W_as + W_peak * (1 + H * E_HD)
#   W_as   = W_max * (D - D_off) / (D + D_50 - 2 D_off)                D > D_off
#   W_peak = E_amp * (D - D_off) / (D_crit - D_off)        D_off < D <= D_crit
#          = E_amp * exp(-E_width * (D - D_crit))                   D > D_crit
#
# Both thicknesses are zero at or below D_off. D is the anatomical (wall to
# wall) diameter. The values are their best fit with the haematocrit term
# (E_HD > 0), on three rat mesenteric networks of 3.3-83 um vessels; W_max is
# the layer van Haaren et al. saw in 150 um arteries, which the fit
# extrapolates to.
PRIES_SECOMB_2005_D_OFF_UM = 2.4
PRIES_SECOMB_2005_D_CRIT_UM = 10.5
PRIES_SECOMB_2005_D_50_UM = 100.0
PRIES_SECOMB_2005_E_AMP_UM = 1.1
PRIES_SECOMB_2005_E_WIDTH_PER_UM = 0.03
PRIES_SECOMB_2005_E_PEAK = 0.6
PRIES_SECOMB_2005_E_HD = 1.18
PRIES_SECOMB_2005_W_MAX_UM = 2.6
#: The vessels the 2005 layer was fitted on, for its description.
PRIES_SECOMB_2005_FITTED_MAX_UM = 83.0


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


def _check_haematocrit(haematocrit: float) -> None:
    if not 0.0 <= haematocrit < 1.0:
        raise ValueError(
            f"haematocrit is a fraction in [0, 1), got {haematocrit}."
        )


def _haematocrit_factor(diameter: float, haematocrit: float) -> float:
    """f(H): how every Pries form scales away from H = 0.45, exactly 1 there."""
    _check_haematocrit(haematocrit)
    d = float(diameter)
    # The haematocrit exponent, which tends to 1 in large vessels.
    g = 1.0 + 1e-11 * d**12
    shape = (0.8 + np.exp(-0.075 * d)) * (-1.0 + 1.0 / g) + 1.0 / g
    numerator = (1.0 - haematocrit) ** shape - 1.0
    denominator = (1.0 - DEFAULT_HAEMATOCRIT) ** shape - 1.0
    return float(numerator / denominator)


def _in_vitro_mu45(diameter: float) -> float:
    """Relative apparent viscosity at H = 0.45 in a glass tube (Pries 1992)."""
    d = float(diameter)
    return float(220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645))


def _in_vivo_mu45(diameter: float) -> float:
    """The 1994 in vivo law's ``mu45*``, before its surface-layer factors."""
    d = float(diameter)
    return float(6.0 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645))


def pries_in_vitro_viscosity(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> float:
    """Pries et al. (1992) in vitro (tube) apparent viscosity, in Pa.s.

    Blood flowing through a channel of exactly this diameter, as measured in
    glass tubes where there is nothing between the fluid and the wall. Use it
    when the diameter you have is the channel -- a plasma-stained lumen -- and
    the surface layer is therefore already excluded by the measurement.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    haematocrit_factor = _haematocrit_factor(diameter, haematocrit)
    return PLASMA_VISCOSITY_PA_S * (
        1.0 + (_in_vitro_mu45(diameter) - 1.0) * haematocrit_factor
    )


def pries_in_vivo_viscosity(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> float:
    """Pries et al. (1994) in vivo apparent viscosity, in Pa.s.

    Valid over 3.3-1978 um. Outside that the expression still evaluates, and
    is still monotonic and above plasma, so it is not clamped -- but it is
    extrapolation, and `describe_law` says where the ends are.
    """
    if diameter <= PRIES_SINGULARITY_UM:
        raise ValueError(
            f"Diameter must exceed {PRIES_SINGULARITY_UM} um for the Pries in "
            f"vivo law -- the endothelial surface layer term is singular "
            f"there -- got {diameter} um. Fix: this vessel is narrower than "
            "the surface layer itself, so a wall-to-wall (endothelial) diameter "
            "cannot be right; check the diameter, or the raw image's stain "
            "(raw_image_stain) if it is a plasma label."
        )
    d = float(diameter)
    haematocrit_factor = _haematocrit_factor(d, haematocrit)
    # The endothelial surface layer: blood flows through D - 1.1, not D. It
    # appears twice, so the leading term carries (D / (D - 1.1))^4 -- the
    # Poiseuille factor for quoting a resistance against a diameter wider than
    # the channel the blood went through.
    surface_layer = (d / (d - PRIES_SINGULARITY_UM)) ** 2

    relative = (
        1.0 + (_in_vivo_mu45(d) - 1.0) * haematocrit_factor * surface_layer
    ) * surface_layer
    return PLASMA_VISCOSITY_PA_S * float(relative)


def pries_secomb_2005_layer_um(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> tuple[float, float]:
    """``(W_ph, W_eff)``: Pries and Secomb's (2005) endothelial surface layer,
    in um, in a vessel *diameter* um across wall to wall.

    ``W_ph`` is the layer's physical thickness, what a stain kept out of it
    would show (their eq 10); ``W_eff`` is how thick it acts on flowing blood,
    thicker round 10 um and with more red cells, which crowd against it there
    (eq 11). Both are zero at or below ``D_off``.
    """
    _check_haematocrit(haematocrit)
    d = float(diameter)
    if d <= PRIES_SECOMB_2005_D_OFF_UM:
        return 0.0, 0.0
    off = PRIES_SECOMB_2005_D_OFF_UM
    asymptotic = PRIES_SECOMB_2005_W_MAX_UM * (d - off) / (
        d + PRIES_SECOMB_2005_D_50_UM - 2.0 * off
    )
    if d <= PRIES_SECOMB_2005_D_CRIT_UM:
        peak = PRIES_SECOMB_2005_E_AMP_UM * (d - off) / (PRIES_SECOMB_2005_D_CRIT_UM - off)
    else:
        peak = PRIES_SECOMB_2005_E_AMP_UM * float(
            np.exp(-PRIES_SECOMB_2005_E_WIDTH_PER_UM * (d - PRIES_SECOMB_2005_D_CRIT_UM))
        )
    physical = asymptotic + peak * PRIES_SECOMB_2005_E_PEAK
    effective = asymptotic + peak * (1.0 + haematocrit * PRIES_SECOMB_2005_E_HD)
    return physical, effective


def pries_secomb_2005_viscosity(
    diameter: float, haematocrit: float = DEFAULT_HAEMATOCRIT
) -> float:
    """Pries and Secomb (2005) in vivo apparent viscosity, in Pa.s, at a
    wall-to-wall diameter.

    The in vitro law at the diameter the surface layer leaves free
    (``D - 2 W_ph``), times ``(D / D_eff)^4`` with ``D_eff = D - 2 W_eff``:
    the resistance of blood confined to ``D_eff``, quoted against ``D`` (see
    :func:`pries_secomb_2005_layer_um`). ``D_eff`` stays positive for any
    haematocrit below 1, so nothing here is singular.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    physical, effective = pries_secomb_2005_layer_um(diameter, haematocrit)
    d = float(diameter)
    free = pries_in_vitro_viscosity(d - 2.0 * physical, haematocrit)
    return free * (d / (d - 2.0 * effective)) ** 4


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
    ``plasma_column`` for the channel the fluid occupies (the in vitro form),
    ``endothelial`` (once ``anatomical``) for a wall-to-wall measurement that
    includes the endothelial surface layer (the 1994 in vivo form). The two
    differ by four to nine times in a capillary, so this is not a detail. A
    form named outright (:data:`PRIES_FORMS`) ignores it.
    """
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter} um.")
    diameter_basis = normalise_diameter_basis(diameter_basis)
    if law == "pries":
        if diameter_basis == "plasma_column":
            return pries_in_vitro_viscosity(diameter, haematocrit)
        if diameter_basis == "endothelial":
            return pries_in_vivo_viscosity(diameter, haematocrit)
        raise ValueError(
            f"Unknown diameter_basis {diameter_basis!r}. Fix: choose one of "
            f"{', '.join(DIAMETER_BASES)}."
        )
    if law == PRIES_IN_VITRO_1992:
        return pries_in_vitro_viscosity(diameter, haematocrit)
    if law == PRIES_IN_VIVO_1994:
        return pries_in_vivo_viscosity(diameter, haematocrit)
    if law == PRIES_IN_VIVO_2005:
        return pries_secomb_2005_viscosity(diameter, haematocrit)
    if law == "capillary_power_law":
        return capillary_power_law_viscosity(diameter)
    if law == "constant":
        return constant_viscosity(diameter)
    raise ValueError(
        f"Unknown viscosity_law {law!r}. Fix: choose one of "
        f"{', '.join(VISCOSITY_LAWS)}."
    )


def validity_range_um(law: str) -> tuple[float, float]:
    """The diameters *law* is fitted over, as (smallest, largest) in um.

    Every Pries form gives the in vitro law's span; the 2005 surface layer's
    own fit stops at 83 um, which :func:`describe_law` says."""
    if law == "pries" or law in PRIES_FORMS:
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
    diameter_basis = normalise_diameter_basis(diameter_basis)
    if law == "pries":
        form = _PRIES_FORM_NAMES[
            PRIES_IN_VITRO_1992 if diameter_basis == "plasma_column" else PRIES_IN_VIVO_1994
        ]
        return (
            f"{law} {form} ({diameter_basis} diameters, fitted {span}, "
            f"discharge haematocrit {haematocrit:g})"
        )
    if law in PRIES_FORMS:
        fitted = f"fitted {span}"
        if law == PRIES_IN_VIVO_2005:
            fitted += f", its surface layer on {low:g}-{PRIES_SECOMB_2005_FITTED_MAX_UM:g} um vessels"
        return (
            f"{law}: {_PRIES_FORM_NAMES[law]} for every vessel, whatever it "
            f"measures (the run's {diameter_basis} diameters included; {fitted}, "
            f"discharge haematocrit {haematocrit:g})"
        )
    return f"{law} (fitted {span})"


_PRIES_FORM_NAMES: dict[str, str] = {
    PRIES_IN_VITRO_1992: "in vitro (Pries et al. 1992)",
    PRIES_IN_VIVO_1994: "in vivo (Pries et al. 1994)",
    PRIES_IN_VIVO_2005: "in vivo (Pries and Secomb 2005)",
}
