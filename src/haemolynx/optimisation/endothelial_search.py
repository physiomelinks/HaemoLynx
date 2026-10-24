"""Sequential, image-informed search over the endothelial ring fit's settings
(the Diameters tab's "Endothelial diameter" block): the counterpart of
:mod:`.fwhm_search` for a raw image that stains the vessel wall (claudin-5,
CD31, isolectin...), where a run reads each vessel's internal diameter as the
lumen inside its wall ring (:mod:`haemolynx.haemodynamics.endothelial`)
instead of by FWHM.

:func:`optimise_endothelial_settings` is the same coordinate descent -- one
setting (or the blur's pair) at a time, everything else held, ties keeping
the value the user had -- over two groups (:data:`ENDOTHELIAL_GROUP_NAMES`):

``ring_gate``
    The weakest ring a reading is kept for (``endothelial_min_ring_contrast``),
    from half to twice its value and the ring fit's own default: a ratio to
    the fit's own scatter, so the same candidates suit any image. First,
    since the gate also decides which rings calibrate the blur and wall: set
    too strict, none do and nothing is read.
``blur_and_wall``
    The PSF and the wall's thickness every section is fitted with: estimated
    from the image's own wide vessels (unset, the default) or fixed. The
    candidates are the image's: its rings' estimate, fixed, and the ends of
    its 95% interval. Fixed values are what make the fit work at all when
    the rings will not calibrate on their own (they disagree on the blur).

The image's own blur and wall -- what the planted vessels are drawn with,
and the fixed candidates' source when the gate being tuned is too strict for
them -- are calibrated at :func:`image_calibration_gate`, never only at the
gate under test.

Nothing here is fitted to one stack: every candidate comes from the image or
from the setting's own scale.

**A trial is the run.** Every trial measures a fixed sample of the vessels a
run measures -- no zero-resistance bridge, and only network handling's keeps
when the run calibrates on those -- inside the whole vessel graph, through
the run's own call (:func:`haemolynx.haemodynamics.apply.measure_endothelial_diameters`)
with the run's haemodynamics config, holding the trial's values of
:data:`ENDOTHELIAL_SETTING_NAMES`. The vessels start as the run's ring fit
finds them: with the mask's widths it reads them by when the run measures
those (``use_edt_diameter_crosscheck``), and none of the run's results. A
trial calibrates the blur and wall as the run does, on the same vessels and
gate; the rings' fits, which do not depend on the gate, are made once per
blur (:func:`haemolynx.haemodynamics.apply.endothelial_calibration_fits`).

**Scored by accuracy, not coverage.** Given the run's segmentation, vessels
of known lumen are drawn into a copy of the image beside sampled ones, with
the image's own blur, wall, ring brightness and noise
(:mod:`haemolynx.haemodynamics.endothelial_planted`), and every trial is
scored on how far off it reads them, on how many of the sample's real
vessels it measures and how their readings agree along each, and on decoys
-- sampled vessels moved into vessel-free tissue (:mod:`~haemolynx.haemodynamics.fwhm_decoys`)
-- it finds a ring in. The planted vessels carry the image's calibrated blur
and wall, so they favour those two values by construction; between the
image's estimate and fixed values, the real vessels decide.

**Read, never swept:** ``edt_junction_proximity_exclusion_um``, the ring
fit's junction zones, is the mask-width setting's and moves those widths
too -- the fallback of every vessel the ring fit leaves -- and the planted
vessels have no junction to judge it by. The sampling along a vessel and
the length a section is averaged over are not settings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import networkx as nx
import numpy as np

from haemolynx.haemodynamics import apply, endothelial, endothelial_planted, fwhm_decoys, sections
from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig
from haemolynx.haemodynamics.fwhm_diameter import CANONICAL_AXIS_ORDER
from haemolynx.io.load import _to_binary_volume_for_skeletonization

from .fwhm_search import _fresh_template, _has_mask_widths, _representative_edges
from .progress import GROUP_FINISHED, ProgressCallback
from .scorecard import ScoreRow
from .search import OptimisationResult, _SweepBookkeeping
from .trial_cache import TrialCache

#: The groups this search runs, in order: the gate first, since it decides
#: whether the image's rings calibrate the blur and wall at all.
ENDOTHELIAL_GROUP_NAMES: tuple[str, ...] = ("ring_gate", "blur_and_wall")

#: A short label for each group, for a GUI checkbox list.
ENDOTHELIAL_GROUP_LABELS: dict[str, str] = {
    "ring_gate": "Ring contrast gate (judged on planted vessels and decoys)",
    "blur_and_wall": "Blur and wall: estimated from the image, or fixed",
}

#: Every setting this search can decide.
ENDOTHELIAL_SETTING_NAMES: tuple[str, ...] = (
    "endothelial_psf_sigma_xy_um",
    "endothelial_psf_sigma_z_um",
    "endothelial_wall_thickness_um",
    "endothelial_min_ring_contrast",
)

#: The name the blur's pair is swept and reported under, as one value
#: ``(sigma_xy, sigma_z)``.
PSF_SWEEP_NAME = "endothelial_psf_sigma_um"

#: How much the planted vessels' lumen error costs: twice its size, as for
#: FWHM (``.fwhm_metrics.PLANTED_ERROR_WEIGHT``) -- a setting reading them
#: 10% further off costs what losing a fifth of the sample does.
PLANTED_ERROR_WEIGHT = 2.0
#: How much a decoy given a width costs: as much as a real vessel without one.
DECOY_WEIGHT = 1.0

#: The gate's candidates, as multiples of its value.
_GATE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)

#: Candidate blur and wall values are rounded to this many places (um).
_ROUNDING = 3

#: "Auto" sample size: the trials' time budget (s), the trials a pass makes
#: at most (blur 5 + wall 5 + gate 6, less the baselines they share), and
#: the fewest vessels ever sampled. The budget is FWHM's three minutes
#: (``.fwhm_search.DEFAULT_AUTO_SAMPLE_TARGET_SECONDS``) times three and a
#: bit: a ring reading costs ~0.2 s a vessel (E14.5), and three minutes
#: sampled 13 of 1,342 vessels there, 4-6 of them read -- so few that one
#: vessel's spread moved the score more than any setting did.
DEFAULT_AUTO_SAMPLE_TARGET_SECONDS = 600.0
_TRIALS_PER_PASS = 14
_MIN_SAMPLE_EDGE_COUNT = 10
#: ...and the most: from :data:`~haemolynx.haemodynamics.sections.MIN_EDGES_FOR_WORKERS`
#: vessels a reading starts worker processes, copying the image to disk and
#: importing the package in each, on every trial -- a cost a probe of ten,
#: read in this process, does not see.
_MAX_AUTO_SAMPLE_EDGE_COUNT = sections.MIN_EDGES_FOR_WORKERS - 1
#: Calibrations a pass fits at most: one free blur and three fixed ones.
_CALIBRATIONS_PER_PASS = 4


@dataclass(frozen=True)
class RingMeasurementQuality:
    """How good one trial's internal diameters are.

    The sample's real vessels give :attr:`real_cost`: over every sampled
    vessel, 1 for one without a width and its readings' spread (CV) for one
    with -- a vessel without a width counts as all of it wrong, as an
    unmeasured planted vessel does. The planted vessels give their lumen
    error, the decoys the share given a width.
    """

    n_edges_total: int
    n_edges_measured: int
    #: Median over the measured vessels of their readings' CV.
    median_diameter_cv: float
    real_cost: float
    decoy_false_positive_rate: float = 0.0
    n_decoys: int = 0
    planted_error: float = 0.0
    n_planted: int = 0
    planted_measured_fraction: float = 0.0
    #: Whether the trial had a blur and wall to fit with (the settings', or
    #: the image's estimate); without them it measures nothing.
    calibrated: bool = True

    @property
    def measured_fraction(self) -> float:
        return self.n_edges_measured / float(self.n_edges_total) if self.n_edges_total else 0.0

    @property
    def score(self) -> float:
        """Lower is better: the real vessels' cost, plus the planted
        vessels' error (:data:`PLANTED_ERROR_WEIGHT`) and the decoys given a
        width (:data:`DECOY_WEIGHT`) where there are some."""
        return (
            self.real_cost
            + (PLANTED_ERROR_WEIGHT * self.planted_error if self.n_planted else 0.0)
            + (DECOY_WEIGHT * self.decoy_false_positive_rate if self.n_decoys else 0.0)
        )


def ring_measurement_quality(
    G: nx.MultiGraph,
    edges: Iterable[tuple],
    *,
    calibrated: bool = True,
    decoys: Optional[nx.MultiGraph] = None,
    planted_report: Optional[Mapping[str, Any]] = None,
) -> RingMeasurementQuality:
    """A :class:`RingMeasurementQuality` from the trial's graph *G*, read on
    its sampled *edges*, its measured *decoys* and its planted vessels'
    report (:func:`~haemolynx.haemodynamics.endothelial_planted.planted_lumen_report`)."""
    edges = list(edges)
    costs, spreads = [], []
    for edge in edges:
        data = G.edges[edge]
        readings = np.asarray(data.get("endothelial_diameter_samples_um") or [], dtype=float)
        if data.get("endothelial_status") != "measured" or readings.size == 0:
            costs.append(1.0)
            continue
        spread = float(np.std(readings) / np.mean(readings)) if np.mean(readings) > 0 else 1.0
        spreads.append(spread)
        costs.append(min(spread, 1.0))
    n_decoys = 0 if decoys is None else decoys.number_of_edges()
    decoys_read = 0 if decoys is None else sum(
        data.get("endothelial_status") == "measured" for _u, _v, data in decoys.edges(data=True)
    )
    planted = planted_report or {}
    return RingMeasurementQuality(
        n_edges_total=len(edges),
        n_edges_measured=len(spreads),
        median_diameter_cv=float(np.median(spreads)) if spreads else 0.0,
        real_cost=float(np.mean(costs)) if costs else 1.0,
        decoy_false_positive_rate=decoys_read / float(n_decoys) if n_decoys else 0.0,
        n_decoys=n_decoys,
        planted_error=float(planted.get("relative_error", 0.0)),
        n_planted=int(planted.get("planted", 0)),
        planted_measured_fraction=float(planted.get("measured_fraction", 0.0)),
        calibrated=calibrated,
    )


#: The before/after scorecard: each measure's name, the
#: :class:`RingMeasurementQuality` field it reads, whether higher is better,
#: and how far it may move the wrong way and still count as the same
#: (``None``: one vessel of the sample).
_SCORECARD_MEASURES: tuple[tuple[str, str, bool, Optional[float]], ...] = (
    ("sampled vessels measured", "measured_fraction", True, None),
    ("width spread along a vessel (CV)", "median_diameter_cv", False, 0.02),
    ("decoys given a width", "decoy_false_positive_rate", False, 0.0),
    ("planted vessels' lumen error", "planted_error", False, 0.02),
)


def _scorecard(
    before: RingMeasurementQuality, after: RingMeasurementQuality
) -> tuple[ScoreRow, ...]:
    one_vessel = 1.0 / max(before.n_edges_total, 1)
    return tuple(
        ScoreRow(
            name, float(getattr(before, field)), float(getattr(after, field)), higher_is_better,
            one_vessel if tolerance is None else tolerance,
        )
        for name, field, higher_is_better, tolerance in _SCORECARD_MEASURES
        if (field != "planted_error" or before.n_planted)
        and (field != "decoy_false_positive_rate" or before.n_decoys)
    )


# -- candidates ---------------------------------------------------------------------


def _rounded(value: float) -> float:
    return round(float(value), _ROUNDING)


def ring_gate_candidates(current: Optional[float]) -> list[float]:
    """``endothelial_min_ring_contrast`` candidates: half to twice its value
    (the ring fit's default when it is unset or 0), the value itself, and
    the ring fit's default -- a gate set so strict the image's rings no
    longer calibrate reads nothing at half or twice it either."""
    default = endothelial.ENDOTHELIAL_MIN_RING_CONTRAST
    base = float(current) if current is not None and float(current) > 0 else default
    values = {_rounded(base * factor) for factor in _GATE_FACTORS} | {float(default)}
    if current is not None:
        values.add(float(current))
    return sorted(values)


def image_calibration_gate(current: Optional[float]) -> float:
    """The gate the image's own blur and wall are calibrated at, for the
    planted vessels and the fixed candidates: the ring fit's default, or the
    user's where that is weaker, but never under half the default -- a gate
    set too strict must not leave the search nothing to judge it by."""
    default = endothelial.ENDOTHELIAL_MIN_RING_CONTRAST
    if current is None:
        return float(default)
    return float(min(max(float(current), 0.5 * default), default))


def _dedupe(values: Iterable[Any]) -> list[Any]:
    from .search import _same_value

    kept: list[Any] = []
    for value in values:
        if not any(_same_value(value, other) for other in kept):
            kept.append(value)
    return kept


def _calibrated_values(
    calibration: Optional[tuple[Any, Any, dict[str, Any]]],
) -> tuple[Optional[tuple[float, float, float]], Optional[float], dict[str, Any]]:
    """The blur and wall a calibration found -- also when it refused them
    for disagreeing (its details keep them) -- and its details."""
    if calibration is None:
        return None, None, {}
    psf, wall, details = calibration
    if psf is None and "sigma_xy_um" in details:
        psf = (details["sigma_z_um"], details["sigma_xy_um"], details["sigma_xy_um"])
    if wall is None:
        wall = details.get("wall_um")
    return (
        None if psf is None else tuple(float(v) for v in psf),
        None if wall is None else float(wall),
        dict(details),
    )


def psf_candidates(
    incumbent: tuple[Optional[float], Optional[float]], details: Mapping[str, Any]
) -> list[tuple[Optional[float], Optional[float]]]:
    """``(sigma_xy, sigma_z)`` candidates: the setting's own pair, unset (the
    image's estimate), and -- from a calibration with the blur free,
    *details* -- that estimate fixed and the ends of its 95% interval."""
    values: list[tuple[Optional[float], Optional[float]]] = [tuple(incumbent), (None, None)]
    if "sigma_xy_um" in details and "sigma_z_um" in details:
        values.append((_rounded(details["sigma_xy_um"]), _rounded(details["sigma_z_um"])))
        xy_interval = details.get("sigma_xy_interval_um")
        z_interval = details.get("sigma_z_interval_um")
        if xy_interval is not None and z_interval is not None:
            for end in (0, 1):
                values.append((_rounded(xy_interval[end]), _rounded(z_interval[end])))
    return _dedupe(values)


def wall_candidates(incumbent: Optional[float], details: Mapping[str, Any]) -> list[Optional[float]]:
    """``endothelial_wall_thickness_um`` candidates: its own value, unset
    (the image's estimate), and -- from a calibration at the blur being
    used, *details* -- that estimate fixed and the ends of its 95% interval."""
    values: list[Optional[float]] = [incumbent, None]
    if "wall_um" in details:
        values.append(_rounded(details["wall_um"]))
        interval = details.get("wall_interval_um")
        if interval is not None:
            values += [_rounded(interval[0]), _rounded(interval[1])]
    return _dedupe(values)


# -- the run's config ---------------------------------------------------------------

#: The branch-order diameter table a config made here carries: measuring
#: widths reads none of it, but a config must have one.
_UNREAD_DIAMETER_TABLE = {"diameter_by_branch_order": {"unread": 1.0}}


def _measurement_config(
    starting_values: Mapping[str, Any],
    *,
    voxel_size_zyx: tuple[float, float, float],
    axis_order: str = CANONICAL_AXIS_ORDER,
    raw_tiff_path: str | Path | None = None,
    raw_channel: int | None = None,
) -> HaemodynamicsApplyConfig:
    """The haemodynamics config a run reads internal diameters with, made
    from *starting_values*, for a caller without the run's own: the panel
    passes that one, as the Diameters stage builds it."""
    edt = {
        name: value
        for name, value in starting_values.items()
        if name.startswith("edt_") or name == "use_edt_diameter_crosscheck"
    }
    walls = {
        name: value for name, value in starting_values.items() if name.startswith("endothelial_")
    }
    walls.update(
        use_endothelial_diameters=True, endothelial_image_path=raw_tiff_path,
        endothelial_channel=raw_channel,
    )
    return HaemodynamicsApplyConfig(
        diameters=dict(_UNREAD_DIAMETER_TABLE),
        endothelial=walls,
        edt=edt,
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        axis_order=axis_order,
    )


def _with_ring_settings(
    config: HaemodynamicsApplyConfig, settings: Mapping[str, Any]
) -> HaemodynamicsApplyConfig:
    """*config* holding *settings*' values of :data:`ENDOTHELIAL_SETTING_NAMES`;
    every other setting is the run's."""
    return replace(
        config,
        endothelial={
            **config.endothelial,
            **{name: settings[name] for name in ENDOTHELIAL_SETTING_NAMES if name in settings},
        },
    )


# -- the search ---------------------------------------------------------------------


class _Calibrations:
    """The run's calibrations of the blur and wall on one image and set of
    vessels, made as they are asked for and kept: the rings' fits by the
    blur they were fitted with, each calibration made from them by ``(blur,
    gate)`` -- a calibration depends on nothing else a trial changes."""

    def __init__(
        self,
        template: nx.MultiGraph,
        config: HaemodynamicsApplyConfig,
        raw_volume: np.ndarray,
        calibration_edges: Optional[list[tuple]],
    ) -> None:
        self.template = template
        self.config = config
        self.raw_volume = raw_volume
        self.calibration_edges = calibration_edges
        self._fits: dict[Any, endothelial.RingCalibrationFits] = {}
        self._calibrations: dict[tuple, Any] = {}
        #: Seconds the rings' fits took, by blur.
        self.seconds: dict[Any, float] = {}

    def __call__(self, settings: Mapping[str, Any]):
        """The calibration for *settings* -- ``None`` when they set both the
        blur and the wall -- as the run's ring fit would make it."""
        config = _with_ring_settings(self.config, settings)
        psf, wall = apply.endothelial_psf_and_wall(config)
        if psf is not None and wall is not None:
            return None
        gate = settings.get("endothelial_min_ring_contrast")
        key = (psf, None if gate is None else float(gate))
        if key not in self._calibrations:
            if psf not in self._fits:
                start = time.perf_counter()
                self._fits[psf] = apply.endothelial_calibration_fits(
                    self.template, config, raw_volume=self.raw_volume,
                    calibration_edges=self.calibration_edges,
                )
                self.seconds[psf] = time.perf_counter() - start
            self._calibrations[key] = apply.calibrate_endothelial_rings(
                self.template, config, raw_volume=self.raw_volume,
                calibration_edges=self.calibration_edges, fits=self._fits[psf],
            )
        return self._calibrations[key]


class _EndothelialSearch(_SweepBookkeeping):
    """Mutable state for one call to :func:`optimise_endothelial_settings`."""

    def __init__(
        self,
        template: nx.MultiGraph,
        *,
        raw_volume: np.ndarray,
        starting_values: Mapping[str, Any],
        config: HaemodynamicsApplyConfig,
        sample_edges: Iterable[tuple],
        pool_size: Optional[int] = None,
        calibration_edges: Optional[list[tuple]] = None,
        calibrations: Optional[_Calibrations] = None,
        progress: Optional[ProgressCallback] = None,
        enabled_groups: Optional[Iterable[str]] = None,
        vessel_mask: Optional[np.ndarray] = None,
        planted_sample_size: int = endothelial_planted.PLANTED_SAMPLE_SIZE,
        decoy_sample_size: int = fwhm_decoys.DECOY_SAMPLE_SIZE,
    ) -> None:
        #: The vessels a run measures, as its ring fit finds them; never
        #: changed by a trial, each of which measures a copy.
        self.template = template
        self.sample_edges: list[tuple] = [tuple(edge) for edge in sample_edges]
        self.pool_size = len(self.sample_edges) if pool_size is None else int(pool_size)
        self.raw_volume = raw_volume
        self.config = config
        #: What the run calibrates the blur and wall on (``None``: every vessel).
        self.calibration_edges = calibration_edges
        self.current: dict[str, Any] = dict(starting_values)
        #: Every calibration a trial asks for (the caller's, when it has
        #: already made some).
        self.calibrations = calibrations or _Calibrations(
            template, config, raw_volume, calibration_edges
        )
        super().__init__(
            progress=progress,
            enabled_groups=enabled_groups,
            group_total=3,
        )
        self.passes_run = 0
        self._outcomes = TrialCache(max_entries=None, max_bytes=None)
        #: What the image's rings say with the blur free (see
        #: `image_calibration_gate`).
        self.image_gate = image_calibration_gate(self.current.get("endothelial_min_ring_contrast"))
        self.image_psf, self.image_wall, self.image_details = _calibrated_values(
            self._calibration({
                **self.current, **_UNSET_BLUR_AND_WALL, "endothelial_min_ring_contrast": self.image_gate,
            })
        )
        self.decoy_probe: Optional[nx.MultiGraph] = None
        self.planted: Optional[endothelial_planted.PlantedRings] = None
        #: Why there are no planted vessels, when there are none, and whose
        #: blur and wall they were drawn with, when there are.
        self.unplanted_reason: Optional[str] = None
        self.planting_source = ""
        if vessel_mask is None:
            self.unplanted_reason = "no segmentation to find vessel-free tissue in"
        else:
            self._prepare_checks(vessel_mask, planted_sample_size, decoy_sample_size)

    # -- what a trial measures with -------------------------------------------
    def _config(self, settings: Mapping[str, Any]) -> HaemodynamicsApplyConfig:
        return _with_ring_settings(self.config, settings)

    def _calibration(self, settings: Mapping[str, Any]):
        """The run's calibration of the blur and wall for *settings* --
        ``None`` when they set both -- its rings fitted once per blur."""
        return self.calibrations(settings)

    def _measure(
        self,
        graph: nx.MultiGraph,
        settings: Mapping[str, Any],
        *,
        volume: Optional[np.ndarray] = None,
        edges: Optional[Iterable[tuple]] = None,
    ) -> dict[str, Any]:
        """The run's ring fit of *graph* (of its *edges*, among all of them,
        when given) with *settings* on *volume* (the image unless given),
        with the blur and wall the run would calibrate on the image."""
        return apply.measure_endothelial_diameters(
            graph,
            self._config(settings),
            raw_volume=self.raw_volume if volume is None else volume,
            edges=edges,
            calibration_edges=self.calibration_edges,
            calibration=self._calibration(settings),
        )

    def _planting_blur_and_wall(self) -> tuple[Optional[tuple], Optional[float], str]:
        """The blur and wall to draw planted vessels with, and where they
        came from: the image's rings with the blur free; failing that, the
        settings' blur and the wall the rings give at it, or the settings'."""
        if self.image_psf is not None and self.image_wall is not None:
            return self.image_psf, self.image_wall, "the image's rings"
        psf, wall = apply.endothelial_psf_and_wall(self._config(self.current))
        if psf is None:
            return None, None, ""
        _psf, at_psf, _details = _calibrated_values(
            self._calibration({
                **self.current, "endothelial_wall_thickness_um": None,
                "endothelial_min_ring_contrast": self.image_gate,
            })
        )
        wall = at_psf if at_psf is not None else wall
        return (psf, wall, "your blur setting") if wall is not None else (None, None, "")

    def _prepare_checks(
        self, vessel_mask: np.ndarray, planted_sample_size: int, decoy_sample_size: int
    ) -> None:
        """Place the sampled vessels' decoys and plant their known-lumen
        vessels -- once, before any trial -- each guided by its vessel's mask
        width: the run's own, or, where the run measures none, one measured
        here as a run would, on a copy no trial reads."""
        guide = self.template.edge_subgraph(self.sample_edges)
        if not _has_mask_widths(guide, self.sample_edges):
            guide = guide.copy()
            apply.measure_edt_diameters(guide, self.config, mask_volume=vessel_mask)
        voxel_size_zyx = tuple(
            float(v) for v in self.template.graph.get("image_voxel_size_zyx", self.config.voxel_size_zyx)
        )
        rng = np.random.default_rng(0)
        order = rng.permutation(len(self.sample_edges))
        ordered = [self.sample_edges[i] for i in order]
        self.decoy_probe = fwhm_decoys.decoy_probe_graph(
            guide, ordered[: max(0, int(decoy_sample_size))], vessel_mask, voxel_size_zyx,
            rng=rng, guide_attribute="edt_diameter_um",
        )
        psf, wall, source = self._planting_blur_and_wall()
        if psf is None:
            self.unplanted_reason = (
                "the image's rings gave no blur and wall to draw them with ("
                + self.image_details.get("reason", "no calibration")
                + "), and the settings set none"
            )
            return
        self.planting_source = source
        self.planted = endothelial_planted.plant_rings(
            self.raw_volume, guide, ordered, vessel_mask, voxel_size_zyx,
            psf_sigma_zyx=psf, wall_um=wall, rng=np.random.default_rng(1),
            sample_size=planted_sample_size,
        )
        if self.planted is None:
            self.unplanted_reason = (
                "no sampled vessel showed a ring to draw them like, or none fitted beside one"
            )

    # -- trials ---------------------------------------------------------------
    def _run_trial(self, overrides: Mapping[str, Any]) -> tuple[nx.MultiGraph, dict[str, Any]]:
        """The run's ring fit of the sampled vessels, among all of them, on a
        fresh copy of the template with ``self.current`` plus *overrides*."""
        settings = {**self.current, **overrides}
        graph = self.template.copy()
        summary = self._measure(graph, settings, edges=self.sample_edges)
        return graph, summary

    def _trial_outcome(self, overrides: Mapping[str, Any]) -> RingMeasurementQuality:
        """The quality of ``self.current`` plus *overrides*, measured once per
        distinct set of the settings this search decides: nothing else a
        trial reads changes during a search."""
        settings = {**self.current, **overrides}
        key = {name: settings.get(name) for name in ENDOTHELIAL_SETTING_NAMES}

        def measure() -> RingMeasurementQuality:
            graph, summary = self._run_trial(overrides)
            planted_report = None
            if self.planted is not None:
                probe = self.planted.probe.copy()
                self._measure(probe, settings, volume=self.planted.volume)
                planted_report = endothelial_planted.planted_lumen_report(probe)
            decoys = None
            if self.decoy_probe is not None:
                decoys = self.decoy_probe.copy()
                self._measure(decoys, settings)
            return ring_measurement_quality(
                graph, self.sample_edges, calibrated=not summary.get("skipped", False),
                decoys=decoys, planted_report=planted_report,
            )

        return self._outcomes.get_or_compute((), key, measure)

    def _quality(self, overrides: Mapping[str, Any]) -> RingMeasurementQuality:
        return self._trial_outcome(overrides)

    def _cost(self, overrides: Mapping[str, Any]) -> float:
        quality = self._quality(overrides)
        self._note(
            measured_fraction=quality.measured_fraction,
            diameter_cv=quality.median_diameter_cv,
            **({"decoy_false_positive_rate": quality.decoy_false_positive_rate} if quality.n_decoys else {}),
            **({"planted_lumen_error": quality.planted_error} if quality.n_planted else {}),
        )
        return quality.score

    # -- groups ---------------------------------------------------------------
    def _group_blur_and_wall(self) -> None:
        group = "blur_and_wall"
        incumbent = (
            self.current.get("endothelial_psf_sigma_xy_um"),
            self.current.get("endothelial_psf_sigma_z_um"),
        )
        # The image's values at the current gate; where it is too strict for
        # them, the image's own (at `image_calibration_gate`).
        _psf, _wall, free = _calibrated_values(
            self._calibration({**self.current, **_UNSET_BLUR_AND_WALL})
        )
        if "sigma_xy_um" not in free:
            free = self.image_details
        chosen = self._run_candidates(
            group, PSF_SWEEP_NAME, psf_candidates(incumbent, free),
            lambda pair: self._cost(
                {"endothelial_psf_sigma_xy_um": pair[0], "endothelial_psf_sigma_z_um": pair[1]}
            ),
            incumbent=incumbent,
        )
        (
            self.current["endothelial_psf_sigma_xy_um"],
            self.current["endothelial_psf_sigma_z_um"],
        ) = chosen
        self._emit(
            GROUP_FINISHED, group,
            winner={"endothelial_psf_sigma_xy_um": chosen[0], "endothelial_psf_sigma_z_um": chosen[1]},
        )
        self._group_index += 1
        _psf, _wall, at_blur = _calibrated_values(
            self._calibration({**self.current, "endothelial_wall_thickness_um": None})
        )
        if "wall_um" not in at_blur and chosen == (None, None):
            at_blur = self.image_details
        self._sweep(
            group, "endothelial_wall_thickness_um",
            wall_candidates(self.current.get("endothelial_wall_thickness_um"), at_blur),
            lambda wall: self._cost({"endothelial_wall_thickness_um": wall}),
        )

    def _group_ring_gate(self) -> None:
        self._sweep(
            "ring_gate", "endothelial_min_ring_contrast",
            ring_gate_candidates(self.current.get("endothelial_min_ring_contrast")),
            lambda gate: self._cost({"endothelial_min_ring_contrast": gate}),
        )

    def run(self, *, max_passes: int = 1) -> None:
        """Run the groups in order, repeating them up to *max_passes* times
        until a pass changes nothing (see ``.search._Search.run``)."""
        max_passes = max(1, int(max_passes))
        if max_passes > 1:
            self._group_total *= max_passes
        for _pass in range(max_passes):
            before = dict(self.current)
            self.passes_run += 1
            self._run_group("ring_gate", self._group_ring_gate)
            self._run_group("blur_and_wall", self._group_blur_and_wall)
            if self.current == before:
                break


#: The blur and wall left to the image.
_UNSET_BLUR_AND_WALL = {
    "endothelial_psf_sigma_xy_um": None,
    "endothelial_psf_sigma_z_um": None,
    "endothelial_wall_thickness_um": None,
}


def _sample_count_for_budget(
    measure: Callable[[list[tuple]], Any],
    G: nx.MultiGraph,
    pool: list[tuple],
    *,
    target_seconds: float,
    planted_sample_size: int,
    decoy_sample_size: int,
) -> tuple[int, Optional[float]]:
    """How many of *pool*'s vessels to sample -- at most
    :data:`_MAX_AUTO_SAMPLE_EDGE_COUNT` -- so the trials of a pass take
    about *target_seconds*, timing ``measure(edges)`` -- one ring fit of
    *edges* among every vessel -- on a probe of ten and on none; and the
    time the trials are estimated to take at that count (``None`` without a
    timing). A trial also measures the planted vessels (up to three per
    sampled one) and the decoys (one each), which cost as a sampled vessel
    does. Every vessel when the probe cannot run."""
    total = len(pool)
    if total <= _MIN_SAMPLE_EDGE_COUNT:
        return total, None
    probe = _representative_edges(G, min(10, total), pool=pool)
    try:
        measure(probe)  # untimed: a fresh process's first call pays its set-up
        start = time.perf_counter()
        measure([])
        fixed = time.perf_counter() - start
        start = time.perf_counter()
        measure(probe)
        per_vessel = max(time.perf_counter() - start - fixed, 0.0) / len(probe)
    except Exception:  # noqa: BLE001 - estimating runtime must never block a real run
        return total, None

    def trial_seconds(count: int) -> float:
        measured = (
            count
            + min(endothelial_planted.PLANTED_PER_VESSEL * count, planted_sample_size)
            + min(count, decoy_sample_size)
        )
        return 3.0 * fixed + per_vessel * measured

    most = min(total, _MAX_AUTO_SAMPLE_EDGE_COUNT)
    count = min(_MIN_SAMPLE_EDGE_COUNT, most)
    while count < most and _TRIALS_PER_PASS * trial_seconds(count + 1) <= target_seconds:
        count += 1
    return count, _TRIALS_PER_PASS * trial_seconds(count)


def _prepared_search(
    G: nx.MultiGraph,
    *,
    starting_values: Mapping[str, Any],
    voxel_size_zyx: tuple[float, float, float],
    raw_tiff_path: str | Path | None = None,
    raw_channel: int | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    config: Optional[HaemodynamicsApplyConfig] = None,
    vessel_mask: Optional[np.ndarray] = None,
    calibration_edges: Optional[Iterable[tuple]] = None,
    sample_edge_count: Optional[int] = None,
    auto_sample_target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    progress: Optional[ProgressCallback] = None,
    groups: Optional[Iterable[str]] = None,
    planted_sample_size: int = endothelial_planted.PLANTED_SAMPLE_SIZE,
    decoy_sample_size: int = fwhm_decoys.DECOY_SAMPLE_SIZE,
) -> tuple[_EndothelialSearch, Optional[float]]:
    """The search :func:`optimise_endothelial_settings` runs, with everything
    it measures with read and made once -- the image and segmentation as a
    run reads them, the vessels as its ring fit finds them, the sample, the
    image's calibration, the planted vessels and decoys -- and the time
    "Auto" estimated it to take (``None`` when the sample size was given)."""
    if config is None:
        if raw_tiff_path is None:
            raise ValueError(
                "The endothelial settings optimiser needs the raw image: raw_tiff_path or a config."
            )
        config = _measurement_config(
            starting_values, voxel_size_zyx=voxel_size_zyx, axis_order=axis_order,
            raw_tiff_path=raw_tiff_path, raw_channel=raw_channel,
        )
    config = _with_ring_settings(config, starting_values)
    vessels = apply.measured_vessels(G)
    pool = (
        list(vessels.edges(keys=True)) if calibration_edges is None
        else [tuple(edge) for edge in calibration_edges if vessels.has_edge(*edge)]
    )
    if not pool and calibration_edges is not None and vessels.number_of_edges():
        raise ValueError(
            "The run calibrates its widths on none of the network's vessels: with "
            "boundary_handling=remove_disconnected, no vessel lies on a path from an inlet to "
            "an outlet."
        )
    if not pool:
        raise ValueError("Graph has no vessels to read internal diameters on.")

    raw_volume = apply.load_endothelial_volume_like_graph(config, G)
    if raw_volume is None:
        raise ValueError("No endothelial image to read: the config names none.")
    if vessel_mask is not None:
        # Read as a run reads it: an ilastik "Simple Segmentation" is labels
        # 1/2, both true as bool.
        vessel_mask = _to_binary_volume_for_skeletonization(
            vessel_mask, use_memmap=config.use_memmap, memmap_directory=config.memmap_directory
        )
    template = _fresh_template(vessels, config, vessel_mask)
    calibration = None if calibration_edges is None else pool
    calibrations = _Calibrations(template, config, raw_volume, calibration)

    estimated_seconds: Optional[float] = None
    if sample_edge_count is None:
        # Timed reading with the image's own blur and wall, calibrated once
        # here (the search reuses it): a probe that calibrated would time the
        # calibration, a minute on E14.5, as if every vessel cost it; one
        # whose gate leaves the rings uncalibrated would read nothing.
        probe_settings = dict(starting_values)
        image_psf, image_wall, _details = _calibrated_values(calibrations({
            **starting_values, **_UNSET_BLUR_AND_WALL,
            "endothelial_min_ring_contrast": image_calibration_gate(
                starting_values.get("endothelial_min_ring_contrast")
            ),
        }))
        if image_psf is not None and image_wall is not None:
            probe_settings.update(
                endothelial_psf_sigma_xy_um=image_psf[1], endothelial_psf_sigma_z_um=image_psf[0],
                endothelial_wall_thickness_um=image_wall,
            )
        probe_config = _with_ring_settings(config, probe_settings)
        sample_edge_count, estimated_seconds = _sample_count_for_budget(
            lambda edges: apply.measure_endothelial_diameters(
                template.copy(), probe_config, raw_volume=raw_volume, edges=edges,
                calibration_edges=calibration, calibration=calibrations(probe_settings),
            ),
            vessels, pool, target_seconds=auto_sample_target_seconds,
            planted_sample_size=planted_sample_size if vessel_mask is not None else 0,
            decoy_sample_size=decoy_sample_size if vessel_mask is not None else 0,
        )
    search = _EndothelialSearch(
        template,
        raw_volume=raw_volume,
        starting_values=starting_values,
        config=config,
        sample_edges=_representative_edges(vessels, sample_edge_count, pool=pool),
        pool_size=len(pool),
        calibration_edges=calibration,
        calibrations=calibrations,
        progress=progress,
        enabled_groups=groups,
        vessel_mask=vessel_mask,
        planted_sample_size=planted_sample_size,
        decoy_sample_size=decoy_sample_size,
    )
    if estimated_seconds is not None:
        estimated_seconds += _CALIBRATIONS_PER_PASS * max(calibrations.seconds.values(), default=0.0)
    return search, estimated_seconds


# -- the report ---------------------------------------------------------------------


def _blur_and_wall_text(psf, wall, details: Mapping[str, Any]) -> str:
    parts = []
    if psf is not None:
        xy, z = psf[1], psf[0]
        xy_interval, z_interval = details.get("sigma_xy_interval_um"), details.get("sigma_z_interval_um")
        parts.append(
            f"sigma_xy {xy:.2f} um"
            + (f" (95% {xy_interval[0]:.2f}-{xy_interval[1]:.2f})" if xy_interval else "")
            + f", sigma_z {z:.2f} um"
            + (f" (95% {z_interval[0]:.2f}-{z_interval[1]:.2f})" if z_interval else "")
        )
    if wall is not None:
        interval = details.get("wall_interval_um")
        parts.append(
            f"wall {wall:.2f} um" + (f" (95% {interval[0]:.2f}-{interval[1]:.2f})" if interval else "")
        )
    return ", ".join(parts)


def _notes(
    search: _EndothelialSearch,
    before: Optional[RingMeasurementQuality],
    after: Optional[RingMeasurementQuality],
) -> list[str]:
    """What the report says about how the trials were judged, and why the
    ring fit could not read the image where it could not."""
    notes = [
        f"Measured a sample of {len(search.sample_edges)} of the {search.pool_size} vessels the run "
        "reads, each among the whole network as a run reads it."
    ]
    details = search.image_details
    at_gate = f"With the blur free and a ring contrast gate of {search.image_gate:g}, the image's rings"
    if search.image_psf is not None and "reason" not in details:
        notes.append(
            f"{at_gate} calibrated at "
            + _blur_and_wall_text(search.image_psf, search.image_wall, details) + "."
        )
    elif search.image_psf is not None:
        notes.append(
            f"{at_gate} will not calibrate on their own ("
            + details["reason"] + "); fixed at what they read, "
            + _blur_and_wall_text(search.image_psf, search.image_wall, details)
            + ", the ring fit can measure."
        )
    else:
        notes.append(
            f"{at_gate} gave no blur and wall (" + details.get("reason", "no calibration") + ")."
        )
        if not search.config.use_edt_diameter_crosscheck:
            notes.append(
                "use_edt_diameter_crosscheck is off, so the ring fit has no mask width to find the "
                "wide vessels it calibrates on by: turn it on, or set endothelial_psf_sigma_xy_um, "
                "endothelial_psf_sigma_z_um and endothelial_wall_thickness_um."
            )
    if search.planted is not None:
        planted = search.planted
        notes.append(
            f"Every trial was scored on {planted.probe.number_of_edges()} vessels of known lumen "
            f"drawn into a copy of the image beside sampled ones, like {planted.reference_rings} "
            "sampled vessels' rings, with the blur and wall of "
            + search.planting_source + " ("
            + _blur_and_wall_text(planted.psf_sigma_zyx, planted.wall_um, {})
            + "): they favour those two values, so between the image's estimate and fixed "
            "values the real vessels decide."
        )
    else:
        notes.append(
            "No vessels of known lumen were planted (" + str(search.unplanted_reason)
            + "): the trials were scored on the real vessels"
            + (" and decoys" if search.decoy_probe is not None else "") + " alone."
        )
    if search.decoy_probe is not None:
        notes.append(
            f"A ring found in any of {search.decoy_probe.number_of_edges()} decoys -- sampled vessels "
            "moved into vessel-free tissue -- counted against a trial."
        )
    if before is not None and after is not None and not (before.n_edges_measured or after.n_edges_measured):
        notes.append(
            "WARNING: no trial measured any sampled vessel, so nothing here was chosen on the "
            "image; check the blur and wall above, and use_edt_diameter_crosscheck."
        )
    return notes


def optimise_endothelial_settings(
    G: nx.MultiGraph,
    *,
    starting_values: Mapping[str, Any],
    voxel_size_zyx: tuple[float, float, float],
    raw_tiff_path: str | Path | None = None,
    raw_channel: int | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    config: Optional[HaemodynamicsApplyConfig] = None,
    vessel_mask: Optional[np.ndarray] = None,
    calibration_edges: Optional[Iterable[tuple]] = None,
    sample_edge_count: Optional[int] = None,
    auto_sample_target_seconds: float = DEFAULT_AUTO_SAMPLE_TARGET_SECONDS,
    progress: Optional[ProgressCallback] = None,
    groups: Optional[Iterable[str]] = None,
    max_passes: int = 1,
    planted_sample_size: int = endothelial_planted.PLANTED_SAMPLE_SIZE,
) -> OptimisationResult:
    """Choose the endothelial ring fit's settings (:data:`ENDOTHELIAL_SETTING_NAMES`)
    for *G* against its endothelial image (see the module docstring).

    *config* is the run's haemodynamics config, as the Diameters stage
    builds it: every trial reads with it -- the image it names, its channel,
    axis order and low-RAM option, the mask-width settings -- holding the
    trial's own values of :data:`ENDOTHELIAL_SETTING_NAMES`. Without it one
    is made from *starting_values*, the image at *raw_tiff_path*
    (*raw_channel*, *axis_order*) and *voxel_size_zyx*. *starting_values*
    holds the four settings' starting values.

    *vessel_mask*, the segmentation *G* was built from, gives the planted
    vessels and the decoys their vessel-free tissue; without it a trial is
    scored on the real vessels alone. *calibration_edges* (``(u, v, key)``)
    are the vessels the run calibrates on -- network handling's keeps
    (``pipeline.stages.edges_network_handling_keeps``); the sample and the
    calibrations come from these alone, while every vessel is still read
    round them. Zero-resistance bridges are never sampled.

    *sample_edge_count* ``None`` picks the sample by timing a trial
    (*auto_sample_target_seconds* for a pass's trials; the calibrations come
    on top); a value fixes it. *groups* restricts the search to some of
    :data:`ENDOTHELIAL_GROUP_NAMES`; *max_passes* repeats them until a pass
    changes nothing. *G* is never changed.
    """
    search, estimated_seconds = _prepared_search(
        G,
        starting_values=starting_values,
        voxel_size_zyx=voxel_size_zyx,
        raw_tiff_path=raw_tiff_path,
        raw_channel=raw_channel,
        axis_order=axis_order,
        config=config,
        vessel_mask=vessel_mask,
        calibration_edges=calibration_edges,
        sample_edge_count=sample_edge_count,
        auto_sample_target_seconds=auto_sample_target_seconds,
        progress=progress,
        groups=groups,
        planted_sample_size=planted_sample_size,
    )
    try:
        before: Optional[RingMeasurementQuality] = search._quality({})
    except Exception:  # noqa: BLE001 - a scorecard must never fail the search it reports on
        before = None
    start = time.perf_counter()
    search.run(max_passes=max_passes)
    seconds = time.perf_counter() - start
    after: Optional[RingMeasurementQuality] = None
    scorecard: tuple[ScoreRow, ...] = ()
    try:
        if before is not None:
            after = search._quality({})
            scorecard = _scorecard(before, after)
    except Exception:  # noqa: BLE001
        scorecard = ()
    return OptimisationResult(
        settings={name: search.current.get(name) for name in ENDOTHELIAL_SETTING_NAMES},
        trials=tuple(search.trials),
        groups_run=tuple(dict.fromkeys(search.groups_run)),
        passes_run=search.passes_run,
        seconds=seconds,
        estimated_seconds=estimated_seconds,
        group_seconds=dict(search.group_seconds),
        scorecard=scorecard,
        notes=tuple(_notes(search, before, after)),
    )
