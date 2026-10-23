"""The endothelial settings optimiser (haemolynx.optimisation.endothelial_search):
the Diameters tab's "Optimise endothelial settings", which chooses the ring
fit's blur, wall and contrast gate for a raw image that stains the wall.

A trial reads what the run reads -- its widths equal assign_edge_diameters'
on the same vessels and settings -- and is scored on vessels of known lumen
planted in the image, the sample's real vessels and decoys. The search logic
(ties, failures, candidates, the calibrations it makes once) is checked on a
stand-in ring fit; the end-to-end searches on synthetic stacks at several
voxel sizes, walls and blurs are marked slow.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.haemodynamics import apply
from haemolynx.optimisation import endothelial_search as s
from haemolynx.optimisation.report import build_report_text

from endothelial_fixtures import hollow_vessel_scene

#: The settings the panel starts the search from, at their schema defaults.
DEFAULTS = {
    "endothelial_psf_sigma_xy_um": None,
    "endothelial_psf_sigma_z_um": None,
    "endothelial_wall_thickness_um": None,
    "endothelial_min_ring_contrast": 4.0,
}


def _run_settings(tmp_path: Path, scene, **overrides):
    """The scene on disk, and the run's settings and haemodynamics config
    for it, made by the pipeline as a run makes them."""
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.stages import _haemodynamics_apply_config

    raw_path, mask_path = tmp_path / "raw.tif", tmp_path / "segmented.tif"
    tifffile.imwrite(str(raw_path), scene.volume)
    tifffile.imwrite(str(mask_path), scene.mask.astype(np.uint8))
    schema = default_schema()
    settings = resolve_settings(
        {
            "input_path": mask_path,
            "raw_image_path": raw_path,
            "raw_image_stain": "endothelial",
            "measure_diameters_from_raw_image": True,
            "run_haemodynamics": True,
            "use_edt_diameter_crosscheck": True,
            **overrides,
        },
        schema=schema,
        config_path=None,
    )
    config = _haemodynamics_apply_config(settings, schema, voxel_size_zyx=scene.voxel_size_zyx)
    return settings, config


def _starting(settings):
    return {name: settings[name] for name in s.ENDOTHELIAL_SETTING_NAMES}


@pytest.fixture(scope="module")
def crossing(tmp_path_factory):
    """Three hollow 9 um vessels crossing, each cut into six connected
    edges: enough wide rings, some tilted towards z, for the image to
    calibrate its own blur and wall."""
    return hollow_vessel_scene(), tmp_path_factory.mktemp("crossing")


_RESULTS = (
    "endothelial_status",
    "endothelial_diameter_um",
    "endothelial_diameter_samples_um",
    "endothelial_contrast_samples",
)


@pytest.mark.parametrize(
    "case",
    ["estimated", "blur_set", "calibrated_on_the_kept_vessels"],
)
def test_a_trials_widths_are_the_runs_on_the_same_vessels(crossing, case):
    """A trial reads its sample among the whole network through the run's
    own call, with the blur and wall the run calibrates on the run's
    vessels, from a template cleared of the run's results: its widths are
    assign_edge_diameters', to the bit -- with the blur and wall estimated,
    with the blur set (the wall estimated at it), and calibrated on the
    vessels network handling keeps."""
    scene, tmp_path = crossing
    overrides = (
        {"endothelial_psf_sigma_xy_um": scene.psf_seen_zyx[1],
         "endothelial_psf_sigma_z_um": scene.psf_seen_zyx[0]}
        if case == "blur_set" else {}
    )
    settings, config = _run_settings(tmp_path, scene, **overrides)
    calibration = (
        [edge for edge in scene.graph.edges(keys=True) if edge[0] < 18]
        if case == "calibrated_on_the_kept_vessels" else None
    )
    G_run = scene.graph.copy()
    apply.assign_edge_diameters(G_run, config, mask_volume=scene.mask, calibration_edges=calibration)

    search, _estimate = s._prepared_search(
        G_run,  # the run's graph, its results on it, as the panel hands it over
        starting_values=_starting(settings),
        voxel_size_zyx=scene.voxel_size_zyx,
        config=config,
        vessel_mask=scene.mask,
        calibration_edges=calibration,
        sample_edge_count=6,
        planted_sample_size=2,
        decoy_sample_size=2,
    )
    trial, summary = search._run_trial({})

    assert "skipped" not in summary
    assert len(search.sample_edges) == 6
    if calibration is not None:
        assert set(search.sample_edges) <= set(calibration)
    measured = [edge for edge in search.sample_edges
                if G_run.edges[edge]["endothelial_status"] == "measured"]
    assert len(measured) >= 3
    for edge in search.sample_edges:
        for name in _RESULTS:
            assert trial.edges[edge].get(name) == G_run.edges[edge].get(name), (edge, name)


# --- the search, on a stand-in ring fit ---------------------------------------------


#: What the stand-in calibration finds with the blur free: the image's blur
#: and wall and their 95% intervals.
_IMAGE = {
    "sigma_xy_um": 0.6, "sigma_z_um": 2.0, "wall_um": 1.0,
    "sigma_xy_interval_um": (0.55, 0.66), "sigma_z_interval_um": (1.8, 2.3),
    "wall_interval_um": (0.9, 1.15),
}


def _ring_settings(config):
    return {name: config.endothelial.get(name) for name in s.ENDOTHELIAL_SETTING_NAMES}


class _StandIn:
    """The ring fit and its calibration, replaced: *reader(settings,
    calibration, edge)* gives an edge's readings (``None``: it fails), and
    *calibrate(psf, gate)* the calibration for a blur (``None``: left free)
    and a gate."""

    def __init__(self, monkeypatch, reader, calibrate=None):
        self.reader = reader
        self.calibrate = calibrate or (
            lambda psf, gate: ((2.0, 0.6, 0.6), 1.0, dict(_IMAGE)) if psf is None
            else (psf, 1.0, {"wall_um": 1.0, "wall_interval_um": (0.9, 1.15)})
        )
        self.fits: list = []
        self.calibrations: list = []
        monkeypatch.setattr(apply, "endothelial_calibration_fits", self._fits)
        monkeypatch.setattr(apply, "calibrate_endothelial_rings", self._calibration)
        monkeypatch.setattr(apply, "measure_endothelial_diameters", self._measure)

    def _fits(self, G, config, *, raw_volume, calibration_edges=None):
        psf, _wall = apply.endothelial_psf_and_wall(config)
        self.fits.append((psf, None if calibration_edges is None else list(calibration_edges)))
        return ("fits", psf)

    def _calibration(self, G, config, *, raw_volume, calibration_edges=None, fits=None):
        psf, wall = apply.endothelial_psf_and_wall(config)
        if psf is not None and wall is not None:
            return None
        assert fits == ("fits", psf)
        self.calibrations.append((psf, config.endothelial["endothelial_min_ring_contrast"]))
        return self.calibrate(psf, config.endothelial["endothelial_min_ring_contrast"])

    def _measure(self, G, config, *, raw_volume=None, edges=None, calibration_edges=None,
                 calibration=None):
        settings = _ring_settings(config)
        if calibration is not None and calibration[0] is None:
            return {"skipped": True, "reason": "could not calibrate"}
        for edge in (list(G.edges(keys=True)) if edges is None else edges):
            readings = self.reader(settings, calibration, edge)
            data = G.edges[edge]
            if readings:
                data.update(
                    endothelial_status="measured", endothelial_diameter_um=float(np.median(readings)),
                    endothelial_diameter_samples_um=list(readings),
                )
            else:
                data["endothelial_status"] = "failed:low_contrast"
        return {"edges_measured": 0}


def _chain(n=4):
    """*n* vessels end to end along x, each with a mask width."""
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    for i in range(n + 1):
        G.add_node(i, pos=np.array([10.0, 20.0, 2.0 + 8.0 * i]))
    for i in range(n):
        G.add_edge(i, i + 1, voxels=[(10.0, 20.0, 2.0 + 8.0 * i + x) for x in range(9)],
                   length=8.0, edt_diameter_um=6.0)
    return G


#: The volume the chain lies in, and the chain filled out to its mask width.
_CHAIN_SHAPE = (20, 40, 44)


def _chain_mask():
    z, y, _x = np.indices(_CHAIN_SHAPE)
    mask = (z - 10.0) ** 2 + (y - 20.0) ** 2 <= 9.0
    mask[:, :, 35:] = False
    return mask


@pytest.fixture()
def raw_path(tmp_path):
    path = tmp_path / "raw.tif"
    tifffile.imwrite(str(path), np.zeros(_CHAIN_SHAPE, dtype=np.float32))
    return path


def _optimise(G, raw_path, **kwargs):
    kwargs.setdefault("starting_values", dict(DEFAULTS))
    kwargs.setdefault("sample_edge_count", 100)
    return s.optimise_endothelial_settings(
        G, voxel_size_zyx=(1.0, 1.0, 1.0), raw_tiff_path=raw_path, **kwargs
    )


def test_a_tie_keeps_every_value_the_user_had(monkeypatch, raw_path):
    _StandIn(monkeypatch, lambda settings, calibration, edge: [6.0, 6.0])

    result = _optimise(_chain(), raw_path)

    assert result.settings == DEFAULTS
    assert result.groups_run == s.ENDOTHELIAL_GROUP_NAMES
    sweeps = {trial.sweep for trial in result.trials}
    assert len(sweeps) == 3  # the blur, the wall, the gate
    for sweep in sweeps:
        chosen = [trial for trial in result.trials if trial.sweep == sweep and trial.chosen]
        assert len(chosen) == 1 and chosen[0].incumbent


def test_a_candidate_that_fails_is_scored_as_a_loss_and_the_sweep_goes_on(monkeypatch, raw_path):
    def reader(settings, calibration, edge):
        gate = settings["endothelial_min_ring_contrast"]
        if gate == 2.0:
            raise RuntimeError("the fit blew up")
        # Gate 3 reads every vessel; any other gate loses the first.
        return [6.0, 6.0] if gate == 3.0 or edge != (0, 1, 0) else None

    _StandIn(monkeypatch, reader)

    result = _optimise(_chain(), raw_path, groups=("ring_gate",))

    gate_trials = {trial.value: trial for trial in result.trials if trial.group == "ring_gate"}
    assert gate_trials[2.0].note.startswith("failed: the fit blew up")
    assert gate_trials[2.0].score == float("inf")
    assert gate_trials[3.0].chosen
    assert result.settings["endothelial_min_ring_contrast"] == 3.0
    assert result.groups_run == ("ring_gate",)


def test_a_ring_found_in_vessel_free_tissue_counts_against_a_trial(monkeypatch, raw_path):
    """Given the segmentation, the sampled vessels are copied into
    vessel-free tissue: a gate loose enough to find a ring there loses to
    one that does not, however many real vessels both read."""
    def reader(settings, calibration, edge_or_decoy):
        return [6.0, 6.0]

    def measure(G, config, **kwargs):
        decoy = any(abs(p[1] - 20.0) > 4.0 or abs(p[0] - 10.0) > 4.0 for p in G.edges[next(iter(
            G.edges(keys=True)))]["voxels"])
        gate = config.endothelial["endothelial_min_ring_contrast"]
        for edge in kwargs.get("edges") or list(G.edges(keys=True)):
            reads = not decoy or gate < 4.0
            G.edges[edge].update(
                endothelial_status="measured" if reads else "failed:low_contrast",
                **({"endothelial_diameter_samples_um": [6.0, 6.0]} if reads else {}),
            )
        return {}

    _StandIn(monkeypatch, reader)
    monkeypatch.setattr(apply, "measure_endothelial_diameters", measure)
    starting = {**DEFAULTS, "endothelial_min_ring_contrast": 2.0}

    result = _optimise(_chain(), raw_path, starting_values=starting, vessel_mask=_chain_mask(),
                       groups=("ring_gate",))

    trials = {trial.value: trial for trial in result.trials}
    assert trials[2.0].metrics["decoy_false_positive_rate"] == 1.0
    assert trials[4.0].metrics["decoy_false_positive_rate"] == 0.0
    assert result.settings["endothelial_min_ring_contrast"] == 4.0
    decoys = next(row for row in result.scorecard if row.name == "decoys given a width")
    assert (decoys.before, decoys.after) == (1.0, 0.0)


def test_the_rings_are_fitted_once_per_blur_and_calibrated_once_per_gate(monkeypatch, raw_path):
    """A calibration's fits depend on the blur alone, its answer on the blur
    and the gate: each is made once however many trials ask for it."""
    stand_in = _StandIn(monkeypatch, lambda settings, calibration, edge: [6.0, 6.0])

    _optimise(_chain(), raw_path)

    blurs = [psf for psf, _edges in stand_in.fits]
    assert len(blurs) == len(set(blurs))
    assert None in blurs and (2.0, 0.6, 0.6) in blurs  # free, and the estimate fixed
    assert len(stand_in.calibrations) == len(set(stand_in.calibrations))
    gates = {gate for psf, gate in stand_in.calibrations if psf is None}
    assert gates == set(s.ring_gate_candidates(4.0))


def test_when_the_rings_will_not_calibrate_on_their_own_their_values_are_fixed(monkeypatch, raw_path):
    """Rings that disagree on the blur leave the run nothing to measure with;
    fixed at what they read, the ring fit can measure, and the search
    proposes them and says why."""
    refused = dict(_IMAGE, reason="the wide vessels' rings do not agree on the blur")
    _StandIn(
        monkeypatch,
        lambda settings, calibration, edge: [6.0, 6.2],
        calibrate=lambda psf, gate: (None, None, dict(refused)) if psf is None
        else (psf, 1.0, {"wall_um": 1.0, "wall_interval_um": (0.9, 1.15)}),
    )

    result = _optimise(_chain(), raw_path)

    assert result.settings["endothelial_psf_sigma_xy_um"] == 0.6
    assert result.settings["endothelial_psf_sigma_z_um"] == 2.0
    measured = next(row for row in result.scorecard if row.name == "sampled vessels measured")
    assert (measured.before, measured.after) == (0.0, 1.0)
    assert any("will not calibrate on their own" in note for note in result.notes)


def test_a_gate_too_strict_for_the_rings_to_calibrate_still_has_the_image_to_judge_by(
    monkeypatch, raw_path
):
    """Regression: the image's blur and wall were calibrated at the gate
    under test; set too strict for any ring to pass, there was nothing to
    plant vessels with, and halving it still read nothing, so it stayed."""
    stand_in = _StandIn(
        monkeypatch,
        lambda settings, calibration, edge: [6.0, 6.0],
        calibrate=lambda psf, gate: (
            (None, None, {"reason": "too few rings"}) if gate > 8.0
            else (((2.0, 0.6, 0.6) if psf is None else psf), 1.0, dict(_IMAGE))
        ),
    )
    starting = {**DEFAULTS, "endothelial_min_ring_contrast": 32.0}

    search, _estimate = s._prepared_search(
        _chain(), starting_values=starting, voxel_size_zyx=(1.0, 1.0, 1.0),
        raw_tiff_path=raw_path, sample_edge_count=100,
    )
    assert search.image_gate == 4.0 and search.image_psf == (2.0, 0.6, 0.6)
    search.run()

    assert search.current["endothelial_min_ring_contrast"] == 4.0
    assert (None, 4.0) in stand_in.calibrations


def test_with_nothing_to_calibrate_on_the_report_says_what_to_set(monkeypatch, raw_path):
    _StandIn(
        monkeypatch,
        lambda settings, calibration, edge: [6.0, 6.0],
        calibrate=lambda psf, gate: (None, None, {"reason": "0 rings of 0 wide vessels showed their wall"}),
    )

    result = _optimise(_chain(), raw_path)

    assert result.settings == DEFAULTS
    text = "\n".join(result.notes)
    assert "use_edt_diameter_crosscheck is off" in text
    assert "WARNING: no trial measured any sampled vessel" in text
    report = build_report_text(result, group_names=s.ENDOTHELIAL_GROUP_NAMES, heading="Endothelial:")
    assert report.startswith("Endothelial:") and "0 rings of 0 wide vessels" in report


def test_the_trials_start_from_vessels_with_none_of_the_runs_results(monkeypatch, raw_path):
    """The panel hands over the run's graph: a trial must not be credited
    with the widths the run left on a vessel it fails."""
    G = _chain()
    for _u, _v, data in G.edges(data=True):
        data.update(endothelial_status="measured", endothelial_diameter_um=42.0,
                    endothelial_diameter_samples_um=[42.0, 42.0], diameter_um=42.0,
                    diameter_source="endothelial", resistance=1.0)
    _StandIn(monkeypatch, lambda settings, calibration, edge: None if edge == (0, 1, 0) else [6.0, 6.0])

    search, _estimate = s._prepared_search(
        G, starting_values=dict(DEFAULTS), voxel_size_zyx=(1.0, 1.0, 1.0),
        raw_tiff_path=raw_path, sample_edge_count=100,
    )
    trial, _summary = search._run_trial({})

    for _u, _v, data in search.template.edges(data=True):
        assert not {"endothelial_status", "endothelial_diameter_um", "diameter_um", "resistance"} & set(data)
        assert data["edt_diameter_um"] == 6.0  # the mask's width the ring fit is guided by
    assert trial.edges[0, 1, 0]["endothelial_status"].startswith("failed")
    assert "endothelial_diameter_um" not in trial.edges[0, 1, 0]
    assert search._quality({}).n_edges_measured == 3
    assert G.edges[0, 1, 0]["endothelial_diameter_um"] == 42.0  # the caller's graph untouched


def test_bridges_and_vessels_the_run_removes_are_never_sampled_nor_calibrated_on(monkeypatch, raw_path):
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    G = _chain()
    G.add_edge(1, 3, voxels=[(10.0, 20.0 + y, 10.0) for y in range(6)], length=5.0,
               **{IS_ZERO_RESISTANCE: True})
    removed = (3, 4, 0)
    kept = [edge for edge in G.edges(keys=True) if edge != removed]
    stand_in = _StandIn(monkeypatch, lambda settings, calibration, edge: [6.0, 6.0])

    search, _estimate = s._prepared_search(
        G, starting_values=dict(DEFAULTS), voxel_size_zyx=(1.0, 1.0, 1.0),
        raw_tiff_path=raw_path, sample_edge_count=100, calibration_edges=kept,
    )

    vessels = {(0, 1, 0), (1, 2, 0), (2, 3, 0)}
    assert set(search.sample_edges) == vessels
    assert (1, 3, 0) not in search.template.edges
    assert removed in search.template.edges  # still read round the sample, as in a run
    assert stand_in.fits and all(set(edges) == vessels for _psf, edges in stand_in.fits)


def test_a_run_calibrating_on_no_vessel_is_refused_with_its_reason(raw_path):
    with pytest.raises(ValueError, match="no vessel lies on a path from an inlet to an outlet"):
        _optimise(_chain(), raw_path, calibration_edges=[])


def test_auto_picks_the_most_vessels_whose_trials_fit_the_budget(monkeypatch):
    """Each trial reads its sample, up to three planted vessels per sampled
    one and a decoy each, and pays a fixed cost three times: the count is
    the largest whose pass of trials fits the budget."""
    clock = {"now": 0.0}
    monkeypatch.setattr(s.time, "perf_counter", lambda: clock["now"])

    def measure(edges):
        clock["now"] += 0.2 + 0.05 * len(edges)

    G = _chain(200)
    pool = list(G.edges(keys=True))

    count, estimate = s._sample_count_for_budget(
        measure, G, pool, target_seconds=60.0, planted_sample_size=60, decoy_sample_size=100
    )

    def trial(n):
        return 3 * 0.2 + 0.05 * (n + min(3 * n, 60) + min(n, 100))

    assert s._TRIALS_PER_PASS * trial(count) <= 60.0 < s._TRIALS_PER_PASS * trial(count + 1)
    assert estimate == pytest.approx(s._TRIALS_PER_PASS * trial(count))

    def failing(edges):
        raise RuntimeError("no")

    assert s._sample_count_for_budget(
        failing, G, pool, target_seconds=60.0, planted_sample_size=60, decoy_sample_size=100
    ) == (len(pool), None)


# --- candidates, quality, the scorecard (pure) ---------------------------------------


@pytest.mark.parametrize("gate", [4.0, 8.0, 1.5, 32.0])
def test_the_gate_candidates_are_half_to_twice_its_value_on_any_scale(gate):
    """And the ring fit's own default, which a gate too strict for the
    image's rings to calibrate at would never reach by halving once."""
    from haemolynx.haemodynamics.endothelial import ENDOTHELIAL_MIN_RING_CONTRAST

    assert s.ring_gate_candidates(gate) == pytest.approx(
        sorted({gate * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)} | {ENDOTHELIAL_MIN_RING_CONTRAST})
    )


@pytest.mark.parametrize("gate, expected", [(None, 4.0), (32.0, 4.0), (3.0, 3.0), (0.5, 2.0)])
def test_the_image_is_calibrated_at_the_default_gate_or_a_weaker_one(gate, expected):
    assert s.image_calibration_gate(gate) == expected


@pytest.mark.parametrize("gate", [None, 0.0])
def test_an_unset_or_zero_gate_is_tried_from_the_ring_fits_own(gate):
    from haemolynx.haemodynamics.endothelial import ENDOTHELIAL_MIN_RING_CONTRAST

    candidates = s.ring_gate_candidates(gate)
    assert ENDOTHELIAL_MIN_RING_CONTRAST in candidates
    assert (gate in candidates) is (gate is not None)


def test_the_blur_candidates_are_the_images_estimate_fixed_and_its_interval():
    assert s.psf_candidates((None, None), _IMAGE) == [
        (None, None), (0.6, 2.0), (0.55, 1.8), (0.66, 2.3),
    ]
    assert s.psf_candidates((0.7, 2.5), {}) == [(0.7, 2.5), (None, None)]


def test_the_wall_candidates_are_the_images_estimate_fixed_and_its_interval():
    assert s.wall_candidates(None, _IMAGE) == [None, 1.0, 0.9, 1.15]
    assert s.wall_candidates(1.5, {"wall_um": 1.0}) == [1.5, None, 1.0]
    assert s.wall_candidates(None, {}) == [None]


def test_a_vessel_without_a_width_costs_as_much_as_all_of_it_wrong():
    G = nx.MultiGraph()
    G.add_edge(0, 1, endothelial_status="measured", endothelial_diameter_samples_um=[10.0, 10.0])
    G.add_edge(1, 2, endothelial_status="measured", endothelial_diameter_samples_um=[8.0, 12.0])
    G.add_edge(2, 3, endothelial_status="failed:low_contrast")
    decoys = nx.MultiGraph()
    decoys.add_edge(0, 1, endothelial_status="measured")
    decoys.add_edge(2, 3, endothelial_status="failed:no_dark_lumen")

    quality = s.ring_measurement_quality(
        G, G.edges(keys=True), decoys=decoys,
        planted_report={"planted": 4, "relative_error": 0.1, "measured_fraction": 0.75},
    )

    assert quality.measured_fraction == pytest.approx(2 / 3)
    assert quality.median_diameter_cv == pytest.approx(0.1)  # 0 and 0.2
    assert quality.real_cost == pytest.approx((0.0 + 0.2 + 1.0) / 3)
    assert quality.decoy_false_positive_rate == 0.5
    assert quality.score == pytest.approx(
        quality.real_cost + s.PLANTED_ERROR_WEIGHT * 0.1 + s.DECOY_WEIGHT * 0.5
    )
    bare = s.ring_measurement_quality(G, G.edges(keys=True))
    assert bare.score == pytest.approx(bare.real_cost)


def test_the_scorecard_shows_the_checks_a_search_had():
    plain = s.RingMeasurementQuality(
        n_edges_total=10, n_edges_measured=5, median_diameter_cv=0.05, real_cost=0.5
    )
    checked = s.RingMeasurementQuality(
        n_edges_total=10, n_edges_measured=6, median_diameter_cv=0.04, real_cost=0.4,
        n_decoys=5, decoy_false_positive_rate=0.2, n_planted=8, planted_error=0.1,
    )

    assert [row.name for row in s._scorecard(plain, plain)] == [
        "sampled vessels measured", "width spread along a vessel (CV)",
    ]
    rows = s._scorecard(checked, checked)
    assert [row.name for row in rows][2:] == ["decoys given a width", "planted vessels' lumen error"]
    assert rows[0].tolerance == pytest.approx(0.1)  # one vessel of ten


# --- end to end, on synthetic stacks at other voxel sizes, walls and blurs (slow) ---

#: (voxel size zyx, optics zyx, wall): a claudin-5-like stack and an
#: anisotropic two-photon one -- the search carries no constant of either.
_STACKS = [
    ((1.0, 0.98, 0.98), (2.0, 0.5, 0.5), 1.0),
    ((2.0, 0.5, 0.5), (2.5, 0.4, 0.4), 0.8),
]


def _dim_scene(voxel, optics, wall):
    """Rings 30 counts over a background of 20: dim, as a weak stain's are."""
    return hollow_vessel_scene(
        voxel_size_zyx=voxel, optics_um=optics, wall_um=wall, peak=30.0, background=20.0,
        extent_um=(56.0, 120.0, 120.0),
    )


@pytest.mark.slow
@pytest.mark.parametrize("voxel, optics, wall", _STACKS, ids=["claudin", "anisotropic"])
def test_a_gate_too_strict_for_the_image_is_lowered_whatever_its_voxels(tmp_path, voxel, optics, wall):
    """At a gate of 32 these rings do not calibrate the image and the ring
    fit reads nothing; the search lowers the gate until it reads the real
    vessels and the planted ones at their lumen, and no decoy."""
    scene = _dim_scene(voxel, optics, wall)
    settings, config = _run_settings(tmp_path, scene, endothelial_min_ring_contrast=32.0)

    result = s.optimise_endothelial_settings(
        scene.graph, starting_values=_starting(settings), voxel_size_zyx=scene.voxel_size_zyx,
        config=config, vessel_mask=scene.mask, sample_edge_count=8, planted_sample_size=6,
        groups=("ring_gate",), max_passes=3,
    )

    assert result.settings["endothelial_min_ring_contrast"] <= 16.0
    rows = {row.name: row for row in result.scorecard}
    assert rows["sampled vessels measured"].before == 0.0
    assert rows["sampled vessels measured"].after >= 0.75
    assert rows["planted vessels' lumen error"].before == 1.0
    assert rows["planted vessels' lumen error"].after < 0.05
    assert rows["decoys given a width"].after == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("voxel, optics, wall", _STACKS, ids=["claudin", "anisotropic"])
def test_a_whole_search_is_no_worse_than_where_it_started(tmp_path, voxel, optics, wall):
    """Every group from the defaults: the image calibrates its own blur and
    wall, so any value proposed for them is the image's own (unset, or
    within its rings' interval), and nothing on the scorecard gets worse."""
    from haemolynx.optimisation.scorecard import worse_rows

    scene = _dim_scene(voxel, optics, wall)
    settings, config = _run_settings(tmp_path, scene)

    result = s.optimise_endothelial_settings(
        scene.graph, starting_values=_starting(settings), voxel_size_zyx=scene.voxel_size_zyx,
        config=config, vessel_mask=scene.mask, sample_edge_count=8, planted_sample_size=6,
    )

    assert not worse_rows(result.scorecard)
    xy, z = (result.settings["endothelial_psf_sigma_xy_um"], result.settings["endothelial_psf_sigma_z_um"])
    if xy is not None:
        assert xy == pytest.approx(scene.psf_seen_zyx[1], rel=0.25)
        assert z == pytest.approx(scene.psf_seen_zyx[0], rel=0.25)
    if result.settings["endothelial_wall_thickness_um"] is not None:
        assert result.settings["endothelial_wall_thickness_um"] == pytest.approx(wall, rel=0.3)
    assert any("calibrated at" in note for note in result.notes)
