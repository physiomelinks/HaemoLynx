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
def test_optimise_fwhm_settings_measures_the_full_graph(multi_vessel):
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
    # The full caller's graph must be measured for real at the end.
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


def test_fwhm_search_shares_its_bookkeeping_base_class_with_search():
    """Regression: _FwhmSearch used to copy-paste _emit/_record/_group_enabled
    from search._Search verbatim instead of sharing them. Both now subclass
    the same _SweepBookkeeping, so a future bookkeeping fix only needs to
    land once."""
    from haemolynx.optimisation.search import _Search, _SweepBookkeeping

    assert issubclass(s._FwhmSearch, _SweepBookkeeping)
    assert issubclass(_Search, _SweepBookkeeping)
    for method in ("_emit", "_record", "_group_enabled"):
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
