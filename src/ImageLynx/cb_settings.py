"""The frozen analysis settings for the carotid body study, in one place.

Every constant here was previously written out separately in each driver under
``examples/``. Four of the open items in ``cb_modelling_reference.md`` were the
same defect in different clothes: a driver constant that had silently drifted from the
``carotid_image_to_model.py`` config default it was supposed to match.

    open item 1   two segmentation thresholds: config 0.65/0.75 against the frozen 0.90 (closed:
                  PreprocessingConfig now reads HYSTERESIS_LOW/HIGH from here)
    open item 2   two boundary rules: band on axis 0 in H1, face on axis 1 in H2
    open item 8   ``M_max`` 10x apart: PerfusionConfig 0.005 against the driver's 0.05 (closed:
                  PerfusionConfig now declares 0.05 too)
    open item 10  pressures: config 100/2 mmHg against the drivers' 60/20 mmHg (closed:
                  HaemodynamicsConfig now reads 60/20 from here)

A shared module does not by itself decide which value is right. What it does is make the
disagreement impossible to reintroduce silently: there is now exactly one place to change,
and ``tests/test_cb_settings.py`` asserts that the pipeline defaults agree with it.

**These are the values every published H1 and H2 number was produced with.** Changing one
invalidates the results in ``cb_modelling_reference.md`` sections 7 and 13. Each is
annotated with what fixed it, and with the measurement that justifies it where one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .specimens import PROCESSING_VOXEL_UM

# ---------------------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------------------

#: Analysed sub-volume, in voxels, identical for all six specimens.
#:
#: 160^3 at the processing voxel is 298.2 x 298.6 x 298.6 um, or 0.0266 mm^3. The imaged
#: blocks it is cut from span 0.227 mm^3 (WKY-C) to 0.653 mm^3 (WKY-A), so the ROI is 4-12%
#: of the block depending on specimen. Matching by size rather than by fraction is what
#: stops raw counts tracking block extent instead of biology: SHR volumes average 89 Mvoxel
#: against 63 for WKY, so a fixed *percentage* would sample a larger absolute box from SHR.
ROI_VOXELS: Tuple[int, int, int] = (160, 160, 160)

#: ROI volume in mm^3, derived rather than restated. Denominator for every density.
ROI_MM3: float = float(np.prod(ROI_VOXELS) * np.prod(PROCESSING_VOXEL_UM) / 1e9)

# ---------------------------------------------------------------------------------------
# Segmentation threshold
# ---------------------------------------------------------------------------------------

#: The probability grid swept by ``cb_h1_batch.py --stage threshold``.
THRESHOLD_GRID: Tuple[float, ...] = (
    0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99,
)

#: One threshold for all six specimens: the median of the six per-specimen selections,
#: snapped to the grid above.
#:
#: The selector runs per specimen, but no specimen runs at its own choice. Per-specimen
#: thresholds would absorb exactly the classifier-quality differences H1 is trying to
#: measure, turning a confound into an apparently clean result. The per-specimen choices
#: are still reported and passed to ``assess_cohort_split``, so a threshold that splits by
#: group stays visible.
#:
#: 0.95 since the 2026-09-28 re-selection, when all six chose 0.95. It was 0.90 (four chose
#: 0.90, two 0.85) until open items 27, 15, 17 and 13 changed what the selector measures: the
#: placed ROI's lateral centre over its own slices (13), calibre on the centreline rather than
#: over every foreground voxel (15, the change that moved it), and an inclusive cut (17).
FROZEN_THRESHOLD: float = 0.95

#: The pipeline receives ``FROZEN_THRESHOLD`` as ``--hysteresis-low`` only, and a low given
#: alone takes low + this offset as its seed, capped at ``HYSTERESIS_SEED_CAP``, so the seed
#: sits above the flood threshold. The operative band was 0.90 / 0.95 until 2026-09-28 and is
#: 0.95 / 0.999 since.
HYSTERESIS_HIGH_OFFSET: float = 0.05

#: Highest seed threshold. 0.95 + 0.05 would be 1.00; the cap keeps the seed below 1. On the
#: Ilastik export, quantised to hundredths, the inclusive seed p >= 0.999 is exactly the
#: saturated set p == 1.0.
HYSTERESIS_SEED_CAP: float = 0.999

#: The pair the H1 masks were built with. ``PreprocessingConfig`` reads both as its defaults
#: (it said 0.65 / 0.75 until open item 1), so a direct run builds the same mask.
HYSTERESIS_LOW: float = FROZEN_THRESHOLD
HYSTERESIS_HIGH: float = min(HYSTERESIS_SEED_CAP, FROZEN_THRESHOLD + HYSTERESIS_HIGH_OFFSET)

# ---------------------------------------------------------------------------------------
# Pressure boundaries
# ---------------------------------------------------------------------------------------

#: Axis whose two faces carry the pressure boundaries.
#:
#: Chosen by availability, not anatomy: it is the only axis solvable in all six specimens.
#: Axis 0 has no outlet terminal in SHR-A and axis 2 has no inlet terminal in SHR-C, and
#: the face rule refuses rather than falling back. That is a property of these six graphs
#: rather than a general rule, and it is stated as such in section 2.8.
BOUNDARY_AXIS: int = 1

#: How close to a face a degree-1 node has to be to count as crossing it.
#:
#: One voxel means "on the face". The other values in the sweep (2 and 4) exist only to
#: show the answer does not depend on this one: varying each rule's own free parameter
#: over its plausible range gives a shunt-ratio spread of 13.3% for the face rule against
#: 75.8% for the band rule - a 5.7-fold reduction, and section 13.4's reason for calling
#: boundary selection the largest single lever in the model.
BOUNDARY_FACE_TOLERANCE_VOXELS: float = 1.0

#: Arteriolar to venular, in mmHg. Every published H2 number used this pair.
#:
#: ``HaemodynamicsConfig`` declared 100/2 mmHg until open item 10, which assumes the entire
#: systemic arterial-to-venous drop falls across ~1 mm of tissue. It does not; most of it
#: falls across the arterial tree upstream and the venous tree downstream. The config now
#: takes its defaults from these two values.
#:
#: Neither choice rescues absolute perfusion. Measured across all six at 60/20, total inlet
#: flow runs 6,511-16,240 um^3/s at flow-weighted velocities of 4.1-9.7 um/s, against a
#: physiological 200-1,000 um/s. Reaching 500 um/s would need about 3,257 mmHg.
INLET_PRESSURE_MMHG: float = 60.0
OUTLET_PRESSURE_MMHG: float = 20.0

#: The coupled flow-haematocrit solve (``solve_coupled_flow_and_hematocrit``), shared by the
#: pipeline and every H2 driver (open item 37).
#:
#: Until item 37 the loop ran at most 15 passes with relaxation 0.5 and an absolute flow
#: tolerance of 1e-4, and it stopped on the cap in every specimen, still wandering. Two
#: switches drove it: rounding-level edges flipping direction, and junctions switching
#: between skimming and proportional mixing. With both removed, relaxation 0.5 still does
#: not converge, and 0.3 and 0.4 leave SHR-C on a periodic cycle. At 0.2 all six converge,
#: in 173 (SHR-B) to 457 (WKY-B) passes, and the pipeline (mPa) and the H2 drivers (mmHg)
#: land on the same haematocrit to 1e-10. The cap is 2.2x the slowest at 0.95, and 1.4x the
#: slowest sensitivity run (WKY-B at 0.93, 726 passes).
#:
#: The stop is scale-free: the largest per-pass flow change, as a fraction of the largest
#: flow, and the largest gap between skimmed and current haematocrit on a flowing edge.
RHEOLOGY_MAX_ITERATIONS: int = 1000
RHEOLOGY_RELAXATION: float = 0.2
RHEOLOGY_FLOW_RTOL: float = 1e-6
RHEOLOGY_HEMATOCRIT_ATOL: float = 1e-4


def rheology_solver_kwargs() -> dict:
    """The rheology settings as keyword arguments for ``solve_coupled_flow_and_hematocrit``."""
    return {
        "max_iterations": RHEOLOGY_MAX_ITERATIONS,
        "relaxation": RHEOLOGY_RELAXATION,
        "flow_rtol": RHEOLOGY_FLOW_RTOL,
        "hematocrit_atol": RHEOLOGY_HEMATOCRIT_ATOL,
    }

# ---------------------------------------------------------------------------------------
# TH / glomus channel
# ---------------------------------------------------------------------------------------

#: Probability above which a voxel of the TH channel is glomus tissue.
TH_THRESHOLD: float = 0.5

#: Fraction of an edge's centreline that has to lie inside the glomus mask for the edge to
#: count as penetrating rather than bypassing.
#:
#: Sampled along the whole polyline rather than at the endpoints: a capillary penetrating a
#: cluster usually begins and ends in stroma, so an endpoint test would classify exactly
#: the vessels H2 section 2.1 is about as extra-glomus.
PENETRATION_FRACTION: float = 0.5

# ---------------------------------------------------------------------------------------
# Tissue transport
# ---------------------------------------------------------------------------------------

#: Perfusion grid pitch, isotropic, in um. Chosen on convergence (open item 30).
#:
#: With vessels mapped over their cross-section (``vessel_mapping="cross_section"``), Tier 1
#: median PO2 and TH-weighted PO2 on WKY-C and SHR-C move at most 0.21 mmHg from 3 to 2 um,
#: but up to 0.78 mmHg from 4 to 3 um (reference section 6.8). 3 um is the coarsest grid
#: within 0.5 mmHg of the next refinement. It was 4 um until open item 30; with the centreline
#: mapping no grid converged, because a finer grid drew each vessel thinner.
#:
#: Measured tissue-to-vessel distance has a median of 5.28-7.92 um across the three WKY
#: specimens, so at 3 um the median tissue voxel sits 1.8-2.6 cells from a vessel.
GRID_UM: float = 3.0

#: Volume-weighted mean maximum metabolic rate, mmol/L/s.
#:
#: 0.05 mmol/L/s is 0.067 mL O2 per mL per minute, against roughly 0.040 for brain - the
#: right order for a metabolically active organ. Every published H2 section 2.3 result used
#: the value here. ``PerfusionConfig.M_max`` declared 0.005, ten times lower, until open item 8
#: set it to this value; ``tests/test_cb_settings.py`` keeps the two equal.
BASE_M_MAX: float = 0.05

#: Glomus-to-stroma metabolic contrasts swept by the hypoxic-fraction driver.
#:
#: A swept parameter, not a measurement. The volume-weighted mean is held at ``BASE_M_MAX``
#: across all three so the runs differ in distribution rather than in total consumption.
METABOLIC_CONTRASTS: Tuple[float, ...] = (1.0, 2.0, 4.0)

#: PO2 thresholds, mmHg, below which tissue is counted hypoxic.
HYPOXIC_THRESHOLDS_MMHG: Tuple[float, ...] = (5.0, 10.0, 20.0)


@dataclass
class PerfusionSettings:
    """The perfusion parameters the H2 drivers pass to the transport solver.

    Previously an identical ad-hoc ``PerfConfig`` class defined separately in
    ``cb_h2_hypoxic_fraction.py`` and ``cb_h2_vtk.py``. ``M_max`` is per-instance because
    the contrast sweep varies it; everything else is fixed.
    """

    #: Oxygen diffusivity in tissue, m^2/s. Converted to um^2/s at matrix assembly.
    M_max: float | np.ndarray = BASE_M_MAX
    sigma_diff: float = 1.5e-9
    #: Rate constant of the saturating metabolic sink, per mmol.
    k_reduce: float = 0.1
    #: Arterial PO2 of the blood entering the network, mmHg. It used to be written out as
    #: 100.0 inside the solver bodies (open item 3); the value is unchanged.
    po2_arterial_mmHg: float = 100.0
    #: Systemic haematocrit. Tier 1 no longer reads it: its washout is at each cell's own
    #: haematocrit, as its source is (open item 29). Until then the washout was evaluated at this
    #: value, first written out as 0.45 in the solver (open item 4). The value is unchanged.
    systemic_hematocrit: float = 0.45
    #: Newton step cap and stop tolerance for Tier 1 (the tolerance is the largest cell residual
    #: in mmHg over the largest PO2, ``_relative_residual``). They were hard-coded in the solver
    #: at 50 and 1e-5 (open item 6). The cap went 50 -> 200 with ``PerfusionConfig``'s (open
    #: item 32): Tier 1 stops after 9-11 steps on every H2 run, so no H2 number moved.
    picard_max_iterations: int = 200
    picard_tolerance: float = 1e-5
