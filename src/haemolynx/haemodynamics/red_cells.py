"""Red cells in a solved vessel: tube haematocrit, red-cell velocity, flux and spacing.

The solve leaves each vessel a volumetric flow ``Q`` (``flow_abs``, m^3/s) and
a discharge haematocrit ``H_D``: the run's one value (``G.graph["haematocrit"]``)
or, under ``haematocrit_model="distributed_iterative"``, its own
``discharge_haematocrit``. The red cells in it travel faster than the blood's
mean velocity -- they keep to the faster-moving middle of the lumen -- so the
haematocrit inside the vessel at any instant, the *tube* haematocrit ``H_T``,
is below the haematocrit of the blood flowing through it: the Fahraeus effect.
Pries, Secomb, Gaehtgens and Gross (1990, Circ Res 67:826-834) fitted it over
glass-tube measurements as

    H_T / H_D = H_D + (1 - H_D) (1 + 1.7 exp(-0.415 D) - 0.6 exp(-0.011 D)),  D in um

(:func:`fahraeus_ratio`), and red-cell conservation gives the rest from ``Q``,
``H_D``, the diameter ``D`` and the mean cell volume ``V``:

* red-cell flux       ``F = Q H_D / V``                  cells per second
* red-cell velocity   ``v_rbc = v_mean H_D / H_T``       ``v_mean = Q / A``, ``A = pi (D/2)^2``
* linear density      ``n = H_T A / V = F / v_rbc``      cells per unit length

These are predictions read off the solve, never measurements: every error in
the flow (the assumed pressure drop, the diameters, the viscosity law, what the
segmentation missed) is in them, and the flux adds the haematocrit model's.
Nothing here changes a resistance or a flow. The statistics write them per
vessel (:mod:`haemolynx.statistics.flux_heterogeneity`) under the attribute
names below.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

__all__ = [
    "DEFAULT_RED_CELL_VOLUME_FL",
    "FAHRAEUS_FIT_MIN_DIAMETER_UM",
    "RED_CELL_EDGE_ATTRIBUTES",
    "RED_CELL_FLUX_ATTR",
    "RED_CELL_LINEAR_DENSITY_ATTR",
    "RED_CELL_VELOCITY_ATTR",
    "TUBE_HAEMATOCRIT_ATTR",
    "fahraeus_ratio",
    "red_cell_flux_per_s",
    "red_cell_linear_density_per_mm",
    "red_cell_velocity_um_s",
    "tube_haematocrit",
    "vessel_discharge_haematocrit",
]

#: Mean red-cell volume (MCV) the flux and spacing are counted with, fL: a
#: mouse's (45-50 fL). Rat ~55-60, human ~90. The ``red_cell_volume_fl``
#: setting's default.
DEFAULT_RED_CELL_VOLUME_FL = 50.0

#: Below this diameter the 1990 fit's tube-to-discharge ratio runs above 1 --
#: cells denser inside the vessel than in the blood flowing through it, which
#: nothing was measured to show: the fit covers 3 um up. The ratio is held at
#: 1 there (cells in single file move with the plasma around them).
FAHRAEUS_FIT_MIN_DIAMETER_UM = 2.58

#: Per-vessel attributes the statistics write, NaN where a vessel carries no
#: flow or has no haematocrit (and, for the three needing a diameter, none).
TUBE_HAEMATOCRIT_ATTR = "tube_haematocrit"
RED_CELL_VELOCITY_ATTR = "red_cell_velocity_um_s"
RED_CELL_FLUX_ATTR = "red_cell_flux_per_s"
RED_CELL_LINEAR_DENSITY_ATTR = "red_cell_linear_density_per_mm"
RED_CELL_EDGE_ATTRIBUTES: tuple[str, ...] = (
    TUBE_HAEMATOCRIT_ATTR,
    RED_CELL_VELOCITY_ATTR,
    RED_CELL_FLUX_ATTR,
    RED_CELL_LINEAR_DENSITY_ATTR,
)

_FEMTOLITRE_M3 = 1e-18
_UM_PER_MM = 1000.0


def _check_haematocrit(discharge_haematocrit: float) -> float:
    h = float(discharge_haematocrit)
    if not 0.0 <= h < 1.0:
        raise ValueError(f"discharge_haematocrit is a fraction in [0, 1), got {h}")
    return h


def _check_diameter(diameter_um: float) -> float:
    d = float(diameter_um)
    if not d > 0.0:
        raise ValueError(f"diameter_um must be positive, got {d}")
    return d


def _check_volume(red_cell_volume_fl: float) -> float:
    v = float(red_cell_volume_fl)
    if not v > 0.0:
        raise ValueError(f"red_cell_volume_fl must be positive, got {v}")
    return v


def fahraeus_ratio(discharge_haematocrit: float, diameter_um: float) -> float:
    """``H_T / H_D`` in a vessel *diameter_um* wide carrying blood of
    *discharge_haematocrit*: Pries et al. 1990's fit, held at 1 below
    :data:`FAHRAEUS_FIT_MIN_DIAMETER_UM` (see there)."""
    h = _check_haematocrit(discharge_haematocrit)
    d = _check_diameter(diameter_um)
    shape = 1.0 + 1.7 * math.exp(-0.415 * d) - 0.6 * math.exp(-0.011 * d)
    return min(h + (1.0 - h) * shape, 1.0)


def tube_haematocrit(discharge_haematocrit: float, diameter_um: float) -> float:
    """The haematocrit inside the vessel at any instant, ``H_T``."""
    return float(discharge_haematocrit) * fahraeus_ratio(discharge_haematocrit, diameter_um)


def red_cell_velocity_um_s(
    mean_velocity_um_s: float, discharge_haematocrit: float, diameter_um: float
) -> float:
    """The cells' velocity, ``v_mean H_D / H_T``: red-cell conservation
    (``v_rbc H_T A = Q H_D``). Defined by the ratio, so a vessel carrying no
    cells still has the speed one would travel at."""
    return float(mean_velocity_um_s) / fahraeus_ratio(discharge_haematocrit, diameter_um)


def red_cell_flux_per_s(
    flow_m3_s: float, discharge_haematocrit: float, red_cell_volume_fl: float
) -> float:
    """Cells per second through the vessel: ``Q H_D / V``."""
    h = _check_haematocrit(discharge_haematocrit)
    v = _check_volume(red_cell_volume_fl)
    return abs(float(flow_m3_s)) * h / (v * _FEMTOLITRE_M3)


def red_cell_linear_density_per_mm(
    diameter_um: float, discharge_haematocrit: float, red_cell_volume_fl: float
) -> float:
    """Cells per millimetre of vessel: ``H_T A / V`` (a femtolitre is a cubic
    micron, so the cross-section in um^2 times a millimetre in um is in fL)."""
    d = _check_diameter(diameter_um)
    v = _check_volume(red_cell_volume_fl)
    area_um2 = math.pi * (0.5 * d) ** 2
    return tube_haematocrit(discharge_haematocrit, d) * area_um2 * _UM_PER_MM / v


def _finite_haematocrit(value: Any) -> Optional[float]:
    try:
        h = float(value)
    except (TypeError, ValueError):
        return None
    return h if math.isfinite(h) and 0.0 <= h < 1.0 else None


def vessel_discharge_haematocrit(
    data: Mapping[str, Any], run_haematocrit: Any
) -> tuple[Optional[float], bool]:
    """The discharge haematocrit a vessel's red cells are counted at, and
    whether it is the vessel's own: its ``discharge_haematocrit`` (written
    under ``haematocrit_model="distributed_iterative"``) when that is a
    fraction in [0, 1), else *run_haematocrit* (``G.graph["haematocrit"]``,
    the one value a fixed-haematocrit run solved with) when that is, else
    ``(None, False)``."""
    own = _finite_haematocrit(data.get("discharge_haematocrit"))
    if own is not None:
        return own, True
    return _finite_haematocrit(run_haematocrit), False
