"""End-to-end tests for haemolynx.optimisation.fwhm_search -- a small
synthetic multi-vessel fixture, reusing the same cylinder-Gaussian style of
volume as tests/test_haemodynamics_automated_fwhm.py.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.optimisation import fwhm_search as s
from haemolynx.pipeline.schema import SCHEMA

#: (y_center, x_start, x_end) for four well-separated, differently-sized
#: parallel vessels -- separated enough in y that one vessel's own Gaussian
#: tail never reaches another's centerline.
_VESSEL_SPECS = [
    (8.0, 2, 10),
    (24.0, 2, 20),
    (40.0, 2, 30),
    (56.0, 2, 38),
]
_SIGMA = 1.5
_Z_CENTER = 5.0
_SHAPE = (11, 60, 40)


def _multi_vessel_raw_volume() -> np.ndarray:
    nz, ny, nx_dim = _SHAPE
    z = np.arange(nz, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    x = np.arange(nx_dim, dtype=float)[None, None, :]
    raw = np.zeros(_SHAPE, dtype=np.float32)
    for yc, x0, x1 in _VESSEL_SPECS:
        r2 = (y - yc) ** 2 + (z - _Z_CENTER) ** 2
        gauss = 100.0 * np.exp(-r2 / (2.0 * _SIGMA**2))
        mask = (x >= x0) & (x <= x1)
        raw = np.where(mask, np.maximum(raw, gauss), raw)
    return raw.astype(np.float32)


def _multi_vessel_graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    node_id = 0
    for i, (yc, x0, x1) in enumerate(_VESSEL_SPECS):
        u, v = node_id, node_id + 1
        node_id += 2
        G.add_node(u, pos=np.array([_Z_CENTER, yc, float(x0)], dtype=float))
        G.add_node(v, pos=np.array([_Z_CENTER, yc, float(x1)], dtype=float))
        voxels = [(_Z_CENTER, yc, float(x)) for x in range(x0, x1 + 1)]
        G.add_edge(
            u, v, weight=1.0, length=float(x1 - x0), branch_order=f"B{i:02d}", voxels=voxels
        )
    return G


def _fwhm_starting_values(**overrides) -> dict:
    defaults = SCHEMA.defaults()
    values = {k: v for k, v in defaults.items() if k.startswith("fwhm_")}
    values["fwhm_sample_spacing_along_edge_um"] = 2.0
    values["fwhm_transverse_profile_step_um"] = 0.25
    values["fwhm_diameter_guess_um"] = 3.0
    values.update(overrides)
    return values


@pytest.fixture()
def multi_vessel(tmp_path: Path):
    raw = _multi_vessel_raw_volume()
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)
    G = _multi_vessel_graph()
    return G, raw_path


# ---------------------------------------------------------------------------
# _representative_subgraph
# ---------------------------------------------------------------------------
def test_representative_subgraph_returns_full_graph_when_count_covers_it(multi_vessel):
    G, _raw_path = multi_vessel
    sample = s._representative_subgraph(G, 100)
    assert sample.number_of_edges() == G.number_of_edges()


def test_representative_subgraph_respects_requested_count(multi_vessel):
    G, _raw_path = multi_vessel
    sample = s._representative_subgraph(G, 2)
    assert sample.number_of_edges() == 2


def test_representative_subgraph_does_not_mutate_the_caller_graph(multi_vessel):
    G, raw_path = multi_vessel
    sample = s._representative_subgraph(G, 2)
    starting_values = _fwhm_starting_values()
    import haemolynx.haemodynamics.automated as automated

    raw_volume = automated.load_single_channel_tiff_volume(raw_path)
    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        sample,
        raw_volume=raw_volume,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        store_profile_debug=True,
        **s._measurement_kwargs(starting_values),
    )
    assert all("fwhm_status" not in data for _u, _v, data in G.edges(data=True))


# ---------------------------------------------------------------------------
# optimise_fwhm_settings -- end to end
# ---------------------------------------------------------------------------
def test_optimise_fwhm_settings_leaves_the_callers_graph_alone_by_default(multi_vessel):
    """Regression: the search measured every edge of the caller's graph with
    the winners at the end, and the panel -- its only caller -- passed a copy
    and threw it away: on a large network, a full FWHM pass for nothing."""
    G, raw_path = multi_vessel
    starting_values = _fwhm_starting_values()
    result = s.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=starting_values,
        sample_edge_count=4,
    )
    assert result.settings  # at least one setting decided
    assert result.trials
    assert all("fwhm_status" not in data for _u, _v, data in G.edges(data=True))
    assert result.seconds > 0.0
    assert result.estimated_seconds is None  # the sample size was given: nothing estimated
    assert set(result.group_seconds) == set(result.groups_run)
    # No vessel mask, so no vessels planted: no planted-width row.
    assert [row.name for row in result.scorecard] == [
        name for name, field, _higher, _tolerance in s._SCORECARD_MEASURES
        if field != "planted_error"
    ]


def test_the_fwhm_scorecard_compares_the_starting_and_chosen_quality():
    from haemolynx.optimisation.fwhm_metrics import FwhmMeasurementQuality

    def quality(measured: int, r2: float) -> FwhmMeasurementQuality:
        return FwhmMeasurementQuality(
            n_edges_total=10, n_edges_measured=measured, measured_fraction=measured / 10,
            mean_fit_r2=r2, median_fit_r2=r2, mean_achieved_extent_ratio=1.0,
            median_diameter_cv=0.1,
        )

    rows = {row.name: row for row in s._scorecard(quality(10, 0.95), quality(8, 0.90))}

    usable = rows["sampled vessels with a usable width"]
    assert (usable.before, usable.after) == (1.0, 0.8)
    assert usable.worse  # two edges of ten: past the one-and-a-half the guard allows
    assert rows["mean fit R2"].worse
    assert not rows["width spread along a vessel (CV)"].worse


def test_optimise_fwhm_settings_measures_the_full_graph_when_asked(multi_vessel):
    G, raw_path = multi_vessel
    s.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=_fwhm_starting_values(),
        sample_edge_count=4,
        groups=["exclusion_zones"],
        measure_full_graph=True,
    )
    assert all(data.get("fwhm_status") == "measured" for _u, _v, data in G.edges(data=True))


def test_optimise_fwhm_settings_restricts_to_requested_groups(multi_vessel):
    G, raw_path = multi_vessel
    starting_values = _fwhm_starting_values()
    result = s.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=starting_values,
        sample_edge_count=4,
        groups=["exclusion_zones"],
    )
    assert result.groups_run == ("exclusion_zones",)
    tried_settings = {trial.setting for trial in result.trials}
    assert tried_settings <= {
        "fwhm_branch_endpoint_exclusion_um",
        "fwhm_junction_proximity_exclusion_um",
    }


def test_optimise_fwhm_settings_raises_on_empty_graph(tmp_path: Path):
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)
    with pytest.raises(ValueError):
        s.optimise_fwhm_settings(
            nx.MultiGraph(),
            raw_tiff_path=raw_path,
            voxel_size_zyx=(1.0, 1.0, 1.0),
            starting_values=_fwhm_starting_values(),
        )


def test_optimise_fwhm_settings_a_failing_candidate_is_scored_as_a_loss(multi_vessel, monkeypatch):
    """A candidate whose real measurement call raises must not end the
    sweep or crash the run -- the setting simply keeps its incoming value
    if every candidate fails."""
    G, raw_path = multi_vessel
    starting_values = _fwhm_starting_values()

    import haemolynx.optimisation.fwhm_search as search_mod

    real_run_trial = search_mod._FwhmSearch._run_trial
    call_count = {"n": 0}

    def _flaky_run_trial(self, overrides):
        call_count["n"] += 1
        if "fwhm_branch_endpoint_exclusion_um" in overrides:
            raise RuntimeError("synthetic failure")
        return real_run_trial(self, overrides)

    monkeypatch.setattr(search_mod._FwhmSearch, "_run_trial", _flaky_run_trial)

    result = search_mod.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=starting_values,
        sample_edge_count=4,
        groups=["exclusion_zones"],
    )
    # Every branch_endpoint_exclusion_um candidate failed -> kept its
    # starting value.
    assert result.settings["fwhm_branch_endpoint_exclusion_um"] == pytest.approx(
        starting_values["fwhm_branch_endpoint_exclusion_um"]
    )
    failed_trials = [
        t for t in result.trials
        if t.setting == "fwhm_branch_endpoint_exclusion_um" and t.note.startswith("failed")
    ]
    assert failed_trials
    assert call_count["n"] > 0


def test_optimise_fwhm_settings_progress_events_reach_the_callback(multi_vessel):
    G, raw_path = multi_vessel
    starting_values = _fwhm_starting_values()
    events = []
    s.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=starting_values,
        sample_edge_count=4,
        groups=["exclusion_zones"],
        progress=events.append,
    )
    assert events
    kinds = {event.kind for event in events}
    assert "group_started" in kinds
    assert "group_finished" in kinds
    assert "candidate_evaluated" in kinds


def test_sample_edge_count_from_probe_seconds_never_exceeds_total_edges():
    result = s._sample_edge_count_from_probe_seconds(
        probe_seconds=0.001, probe_edge_count=5, total_edges=1000, target_seconds=180.0
    )
    assert result <= 1000


def test_sample_edge_count_from_probe_seconds_zero_probe_time_uses_every_edge():
    result = s._sample_edge_count_from_probe_seconds(
        probe_seconds=0.0, probe_edge_count=5, total_edges=42, target_seconds=180.0
    )
    assert result == 42


def test_the_estimate_auto_reports_is_the_one_it_chose_the_sample_by():
    count = s._sample_edge_count_from_probe_seconds(
        probe_seconds=0.1, probe_edge_count=10, total_edges=1000, target_seconds=180.0
    )
    assert 10 < count < 1000
    assert s._estimated_search_seconds(0.1, 10, count) <= 180.0
    assert s._estimated_search_seconds(0.1, 10, count + 1) > 180.0


def test_fwhm_search_shares_its_bookkeeping_base_class_with_search():
    """Regression: _FwhmSearch used to copy-paste _emit/_record/_group_enabled
    from search._Search verbatim instead of sharing them. Both now subclass
    the same _SweepBookkeeping, so a future bookkeeping fix only needs to
    land once."""
    from haemolynx.optimisation.search import _Search, _SweepBookkeeping

    assert issubclass(s._FwhmSearch, _SweepBookkeeping)
    assert issubclass(_Search, _SweepBookkeeping)
    for method in ("_emit", "_record", "_group_enabled", "_sweep", "_run_candidates", "_run_group"):
        assert getattr(s._FwhmSearch, method) is getattr(_Search, method), (
            f"{method} is no longer the shared _SweepBookkeeping implementation"
        )


def test_fwhm_search_group_enabled_and_emit_still_work_through_the_base_class():
    graph = _multi_vessel_graph()
    events = []
    search = s._FwhmSearch(
        graph,
        raw_volume=_multi_vessel_raw_volume(),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values={},
        progress=events.append,
        enabled_groups=("exclusion_and_extent",),
    )
    assert search._group_enabled("exclusion_and_extent") is True
    assert search._group_enabled("baseline_estimation") is False
    assert search.groups_run == ["exclusion_and_extent"]

    search._emit("group_started", "exclusion_and_extent")
    assert len(events) == 1
    assert events[0].group_name == "exclusion_and_extent"

    search._record("exclusion_and_extent", "some_setting", 1.0, 0.5, note="ok")
    assert len(search.trials) == 1
    assert search.trials[0].setting == "some_setting"


# ---------------------------------------------------------------------------
# The run's own checks on every trial: image PSF, decoys, EDT cross-check
# ---------------------------------------------------------------------------
def _vessel_mask() -> np.ndarray:
    """The four vessels as the segmentation would have them."""
    return _multi_vessel_raw_volume() > 50.0


def _search(G, raw, **kwargs) -> s._FwhmSearch:
    return s._FwhmSearch(
        G,
        raw_volume=raw,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=_fwhm_starting_values(),
        progress=None,
        **kwargs,
    )


@pytest.mark.parametrize("fixed", [True, False])
def test_every_trial_holds_the_blur_at_the_image_psf(multi_vessel, monkeypatch, fixed):
    """Regression: the optimiser's trials fitted each profile's blur freely
    while the run holds it at one image PSF, so it tuned a different
    measurement from the one the run makes."""
    import functools

    G, raw_path = multi_vessel
    seen = []
    real = s.automated.measure_edge_diameters_fwhm_from_raw_tiff

    @functools.wraps(real)  # the search reads its keyword arguments off the signature
    def recording(graph, **kwargs):
        seen.append(kwargs.get("profile_psf_sigma_zyx"))
        return real(graph, **kwargs)

    monkeypatch.setattr(s.automated, "measure_edge_diameters_fwhm_from_raw_tiff", recording)
    s.optimise_fwhm_settings(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        starting_values=_fwhm_starting_values(
            fwhm_fix_blur_to_image_psf=fixed,
            raw_section_psf_sigma_xy_um=0.8,
            raw_section_psf_sigma_z_um=1.6,
        ),
        sample_edge_count=4,
        groups=["exclusion_zones"],
    )

    assert seen
    assert set(seen) == ({(1.6, 0.8, 0.8)} if fixed else {None})


def _speckled(clean: np.ndarray) -> np.ndarray:
    """*clean* with blurred specks, as bright as the vessels, in the tissue
    the mask leaves out."""
    from scipy import ndimage

    rng = np.random.default_rng(1)
    specks = ndimage.gaussian_filter((rng.random(clean.shape) < 0.05).astype(float), 1.2)
    specks *= 100.0 / specks.max()
    specks[ndimage.binary_dilation(_vessel_mask(), iterations=3)] = 0.0
    return np.maximum(clean, specks).astype(np.float32)


def test_decoys_in_unsegmented_speckle_count_against_a_trial():
    """With the vessel mask, each trial measures decoys beside the sampled
    vessels; FWHM fitting specks there costs the trial."""
    G = _multi_vessel_graph()
    clean = _multi_vessel_raw_volume()

    on_clean = _search(G.copy(), clean, vessel_mask=_vessel_mask())._quality({})
    on_speckled = _search(G.copy(), _speckled(clean), vessel_mask=_vessel_mask())._quality({})

    assert on_clean.decoy_false_positive_rate == pytest.approx(0.0)
    assert on_speckled.decoy_false_positive_rate > 0.0
    assert on_speckled.score > on_clean.score


def test_a_width_disagreeing_with_the_mask_is_not_usable():
    G = _multi_vessel_graph()
    for _u, _v, data in G.edges(data=True):
        data["edt_diameter_um"] = 20.0  # far from the ~3.5 um FWHM reads

    quality = _search(G, _multi_vessel_raw_volume())._quality({})

    assert quality.n_edges_measured > 0
    assert quality.n_edges_demoted == quality.n_edges_measured
    assert quality.usable_fraction == pytest.approx(0.0)
    assert quality.edt_disagreement_fraction == pytest.approx(1.0)


def test_the_mask_gives_the_sample_its_own_width_without_touching_the_callers_graph():
    G = _multi_vessel_graph()
    search = _search(s._representative_subgraph(G, 4), _multi_vessel_raw_volume(),
                     vessel_mask=_vessel_mask())

    assert all(
        float(data.get("edt_diameter_um") or 0.0) > 0.0
        for _u, _v, data in search.sample_graph_template.edges(data=True)
    )
    assert all("edt_diameter_um" not in data for _u, _v, data in G.edges(data=True))
    assert search.decoy_probe is not None and search.decoy_probe.number_of_edges() > 0


def test_the_guard_judges_usable_widths_not_measured_ones(monkeypatch):
    """A gate that stops FWHM fitting specks measures fewer edges and loses
    no usable width; the guard used to veto it for measuring fewer."""
    baseline = FwhmQuality(measured=10, demoted=4, fp=0.5)
    gated = FwhmQuality(measured=6, demoted=0, fp=0.0)
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume())
    monkeypatch.setattr(
        search, "_quality", lambda overrides: gated if overrides else baseline
    )

    chosen = search._guarded_sweep(
        "rejection_gates", "fwhm_reject_samples_with_low_fit_r2", [True],
    )

    assert chosen is True
    assert search.trials[-1].score < 100.0  # no guard penalty


def FwhmQuality(*, measured: int, demoted: int, fp: float):
    from haemolynx.optimisation.fwhm_metrics import FwhmMeasurementQuality

    return FwhmMeasurementQuality(
        n_edges_total=10, n_edges_measured=measured, measured_fraction=measured / 10,
        mean_fit_r2=0.9, median_fit_r2=0.9, mean_achieved_extent_ratio=1.0,
        median_diameter_cv=0.0, n_edges_demoted=demoted, decoy_false_positive_rate=fp,
    )


def test_the_guard_lets_a_small_sample_lose_one_edge_but_not_two(monkeypatch):
    """Regression: on a sample of ten edges the 5% tolerance was half an
    edge, so a candidate measuring one edge fewer was vetoed while one
    measuring an edge more was rewarded -- a pull towards looser settings."""
    baseline = FwhmQuality(measured=10, demoted=0, fp=0.0)
    by_value = {
        "one_fewer": FwhmQuality(measured=9, demoted=0, fp=0.0),
        "two_fewer": FwhmQuality(measured=8, demoted=0, fp=0.0),
    }
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume())
    monkeypatch.setattr(
        search, "_quality",
        lambda overrides: by_value[overrides["fwhm_profile_baseline_mode"]] if overrides else baseline,
    )

    search._guarded_sweep(
        "baseline_estimation", "fwhm_profile_baseline_mode", ["one_fewer", "two_fewer"],
    )

    scores = {t.value: t.score for t in search.trials}
    assert scores["one_fewer"] < 100.0
    assert scores["two_fewer"] > 500.0


def test_the_guard_tolerance_is_five_percent_on_a_large_sample():
    assert s._FwhmSearch._usable_fraction_tolerance(10) == pytest.approx(0.15)
    assert s._FwhmSearch._usable_fraction_tolerance(200) == pytest.approx(0.05)


def test_a_sweeps_baseline_is_its_current_values_own_trial_measured_once(monkeypatch):
    """Regression: every sweep measured its current value twice -- once as
    the baseline and once as a candidate -- about a quarter of every FWHM
    search. A trial asked for again, in a later sweep or pass, is free."""
    import functools

    calls = {"n": 0}
    real = s.automated.measure_edge_diameters_fwhm_from_raw_tiff

    @functools.wraps(real)  # the search reads its keyword arguments off the signature
    def counting(graph, **kwargs):
        calls["n"] += 1
        return real(graph, **kwargs)

    monkeypatch.setattr(s.automated, "measure_edge_diameters_fwhm_from_raw_tiff", counting)
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume())
    current = float(search.current["fwhm_branch_endpoint_exclusion_um"])
    candidates = sorted({0.0, current, current + 2.0})

    search._guarded_sweep("exclusion_zones", "fwhm_branch_endpoint_exclusion_um", candidates)
    assert calls["n"] == len(candidates)

    search._guarded_sweep("exclusion_zones", "fwhm_branch_endpoint_exclusion_um", candidates)
    assert calls["n"] == len(candidates)


def test_the_baseline_diameters_are_the_current_settings_trial(monkeypatch):
    import functools

    calls = {"n": 0}
    real = s.automated.measure_edge_diameters_fwhm_from_raw_tiff

    @functools.wraps(real)
    def counting(graph, **kwargs):
        calls["n"] += 1
        return real(graph, **kwargs)

    monkeypatch.setattr(s.automated, "measure_edge_diameters_fwhm_from_raw_tiff", counting)
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume())

    diameters = search._baseline_diameters_um()
    quality = search._quality({})

    assert calls["n"] == 1
    assert diameters.size > 0 and np.all(diameters > 0)
    assert quality.n_edges_measured > 0


# ---------------------------------------------------------------------------
# Accuracy: vessels of known width planted beside the sampled ones
# ---------------------------------------------------------------------------
def test_the_measurement_model_is_left_alone_with_nothing_to_judge_it_by():
    """Coverage and fit quality cannot tell a right width from a wrong one;
    without the segmentation to plant vessels beside, these settings keep
    their values."""
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume())

    search._group_measurement_model()

    assert search.planted is None
    assert [(t.group, t.note) for t in search.trials] == [
        ("measurement_model",
         "skipped: no vessels planted to judge accuracy by (needs the run's segmentation)")
    ]


def test_with_the_segmentation_every_trial_is_scored_on_planted_vessels():
    search = _search(_multi_vessel_graph(), _multi_vessel_raw_volume(), vessel_mask=_vessel_mask())

    quality = search._quality({})

    assert search.planted is not None
    assert quality.n_planted == search.planted.probe.number_of_edges() > 0
    assert 0.0 <= quality.planted_error <= 1.0


def test_the_scorecard_shows_the_planted_vessels_error_when_there_are_some():
    from haemolynx.optimisation.fwhm_metrics import FwhmMeasurementQuality

    def quality(error):
        return FwhmMeasurementQuality(
            n_edges_total=10, n_edges_measured=10, measured_fraction=1.0, mean_fit_r2=0.9,
            median_fit_r2=0.9, mean_achieved_extent_ratio=1.0, median_diameter_cv=0.1,
            planted_error=error, n_planted=9,
        )

    rows = {row.name: row for row in s._scorecard(quality(0.05), quality(0.12))}
    assert rows["planted vessels' width error"].worse


def test_the_sample_is_spread_over_the_vessels_widths():
    """Regression: spread over lengths, a sample of a network of capillaries
    of every length rarely drew one of its few wide vessels."""
    G = nx.MultiGraph()
    for index in range(40):
        wide = index in (7, 23)
        G.add_edge(2 * index, 2 * index + 1, length=10.0 + index, edt_diameter_um=12.0 if wide else 3.0)

    sample = s._representative_subgraph(G, 5)

    widths = [data["edt_diameter_um"] for _u, _v, data in sample.edges(data=True)]
    assert 12.0 in widths


def test_without_widths_the_sample_is_spread_over_lengths():
    G = nx.MultiGraph()
    for index in range(40):
        G.add_edge(2 * index, 2 * index + 1, length=float(index))
    lengths = sorted(d["length"] for _u, _v, d in s._representative_subgraph(G, 5).edges(data=True))
    assert lengths[0] < 8 and lengths[-1] >= 32  # one from each fifth
