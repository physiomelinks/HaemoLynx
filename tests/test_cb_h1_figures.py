"""`cb_h1_figures.py` draws every number from the batch and sensitivity outputs (re-run package D).

Figure 1 used to draw its junction and length panels from a table copied out of an old run's
logs, and kept showing SHR +34% / +27% after the data had moved to +6% / +2%. Figure 2's gap,
Figure 3's series and every p-value label were literals too. These tests build tiny per-edge
tables and check that each figure reads them.
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cb_h1_batch
import cb_h1_figures
from ImageLynx import cb_settings
from ImageLynx.batch_outputs import EDGE_TABLE_NAME, BatchRun
from ImageLynx.specimens import SPECIMENS
from ImageLynx.statistics.cohort_split import assess_cohort_split

FIELDS = ["u", "v", "key", "length_um", "edt_diameter_um", "edt_junction_trim"]

# A triangle 0-1-2 with a parallel 0-1 edge and a tail 2-3: E = 5, V = 4, so beta-1 = 2;
# nodes 0, 1 and 2 have degree 3 on the MultiGraph, node 3 degree 1.
BASE_EDGES = [(0, 1), (0, 1), (1, 2), (2, 0), (2, 3)]


def _write(run_dir, specimen_id, edges, length_um=10.0, diameter_um=6.0, trimmed=1):
    folder = run_dir / specimen_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / EDGE_TABLE_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, (u, v) in enumerate(edges):
            writer.writerow({"u": u, "v": v, "key": i, "length_um": length_um,
                             "edt_diameter_um": diameter_um,
                             "edt_junction_trim": "trimmed" if i < trimmed
                             else "untrimmed_too_short"})


def _ring(n_extra):
    """BASE_EDGES plus n_extra more triangles hung off node 3: each adds one loop."""
    edges = list(BASE_EDGES)
    for k in range(n_extra):
        a, b = 100 + 2 * k, 101 + 2 * k
        edges += [(3, a), (a, b), (b, 3)]
    return edges


class _FakeReader:
    """Stands in for ``open_batch_run``: a real ``BatchRun`` over the tiny CSVs these tests
    write, without the placed-ROI and cache checks.

    ``run_dir`` None is the batch (``batch_root / specimen``); otherwise the folder passed.
    ``opened`` records every (specimen, folder) so a test can see what was opened how.
    """

    def __init__(self, batch_root):
        self.batch_root = batch_root
        self.opened = []

    def __call__(self, specimen, run_dir=None):
        self.opened.append((specimen.specimen_id, run_dir))
        folder = self.batch_root / specimen.specimen_id if run_dir is None else Path(run_dir)
        path = folder / EDGE_TABLE_NAME
        if not path.exists():
            raise FileNotFoundError(f"{specimen.specimen_id} ({folder}): no {path.name}.")
        return BatchRun(specimen, folder, None, None)


@pytest.fixture(autouse=True)
def _fresh_tables():
    """The driver caches each opened run; a test's tmp folders must not see the last one's."""
    cb_h1_figures._run.cache_clear()
    yield
    cb_h1_figures._run.cache_clear()


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """Batch plus both sensitivity runs; SHR carries one more loop than WKY at every threshold."""
    results = tmp_path / "cb_h1_batch"
    sensitivity = tmp_path / "cb_h1_sensitivity"
    low, high = cb_h1_batch.sensitivity_thresholds(cb_settings.FROZEN_THRESHOLD)
    runs = {low: sensitivity / f"t{low:.2f}", cb_settings.FROZEN_THRESHOLD: results,
            high: sensitivity / f"t{high:.2f}"}
    for step, run_dir in enumerate(runs.values()):
        for i, specimen in enumerate(SPECIMENS):
            extra = i % 3 + (1 + step if specimen.group == "SHR" else 0)
            diameter = 7.0 + 0.1 * (i % 3) if specimen.group == "WKY" else 6.0 + 0.1 * (i % 3)
            _write(run_dir, specimen.specimen_id, _ring(extra), diameter_um=diameter,
                   trimmed=2 + i % 2)
    (results / "threshold_selection.json").write_text(json.dumps({
        "frozen": cb_settings.FROZEN_THRESHOLD,
        "fragmentation_onset": {s.specimen_id: (high if i < 4 else None)
                                for i, s in enumerate(SPECIMENS)}}))
    reader = _FakeReader(results)
    monkeypatch.setattr(cb_h1_figures, "OUTPUT_DIR", results)
    monkeypatch.setattr(cb_h1_figures, "SENSITIVITY_DIR", sensitivity)
    monkeypatch.setattr(cb_h1_figures, "open_batch_run", reader)
    return SimpleNamespace(results=results, sensitivity=sensitivity, runs=runs, reader=reader)


def test_the_hand_copied_topology_table_is_gone():
    assert not hasattr(cb_h1_figures, "TOPOLOGY")


def test_network_measures_counts_loops_junctions_and_length_on_the_multigraph(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_h1_figures, "open_batch_run", _FakeReader(tmp_path))
    for specimen in SPECIMENS:
        _write(tmp_path, specimen.specimen_id, BASE_EDGES, length_um=10.0)
    measures = cb_h1_figures.network_measures(tmp_path)
    roi = cb_h1_figures.ROI_MM3
    for values in measures.values():
        assert values["beta1"] == pytest.approx(2 / roi)
        assert values["junctions"] == pytest.approx(3 / roi)   # the parallel edge counts
        assert values["length"] == pytest.approx(50.0 / roi)


def test_a_missing_table_raises_rather_than_dropping_the_specimen(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_h1_figures, "open_batch_run", _FakeReader(tmp_path))
    for specimen in SPECIMENS[1:]:
        _write(tmp_path, specimen.specimen_id, BASE_EDGES)
    with pytest.raises(FileNotFoundError, match=SPECIMENS[0].specimen_id):
        cb_h1_figures.network_measures(tmp_path)


@pytest.mark.parametrize("column, read", [
    ("length_um", lambda: cb_h1_figures.network_measures()),
    ("edt_diameter_um", lambda: cb_h1_figures._load_diameters()),
    ("length_um", lambda: cb_h1_figures._degree_and_length()),
], ids=["density length", "diameter figure", "segment-length figure"])
def test_a_blank_cell_stops_the_figures_rather_than_being_skipped(outputs, column, read):
    path = outputs.results / "SHR-C" / EDGE_TABLE_NAME
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1][column] = ""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=rf"SHR-C.*empty or non-finite {column}.*\(0, 1, 1\)"):
        read()


def test_the_batch_is_opened_by_default_and_each_sensitivity_run_by_its_folder(outputs):
    cb_h1_figures.sensitivity_series()
    low, high = cb_h1_batch.sensitivity_thresholds(cb_settings.FROZEN_THRESHOLD)
    for specimen in SPECIMENS:
        sid = specimen.specimen_id
        assert (sid, None) in outputs.reader.opened
        assert (sid, outputs.sensitivity / f"t{low:.2f}" / sid) in outputs.reader.opened
        assert (sid, outputs.sensitivity / f"t{high:.2f}" / sid) in outputs.reader.opened
    # Opened once each, however many figures read the run.
    assert len(outputs.reader.opened) == len(set(outputs.reader.opened)) == 3 * len(SPECIMENS)
    cb_h1_figures.junction_trim_shares()
    cb_h1_figures.network_measures()
    assert len(outputs.reader.opened) == 3 * len(SPECIMENS)


def test_a_missing_batch_table_stops_the_figures(outputs):
    (outputs.results / "SHR-C" / EDGE_TABLE_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="SHR-C"):
        cb_h1_figures.network_measures()


@pytest.mark.parametrize("values, state", [
    ({"WKY-A": 1, "WKY-B": 2, "WKY-C": 5, "SHR-A": 3, "SHR-B": 4, "SHR-C": 6}, "groups overlap"),
    ({"WKY-A": 1, "WKY-B": 2, "WKY-C": 3, "SHR-A": 4, "SHR-B": 5, "SHR-C": 6}, "groups separate"),
])
def test_the_p_value_label_is_the_exact_permutation_p(values, state):
    summary = cb_h1_figures.group_summary(values)
    expected = assess_cohort_split(values).permutation_p
    assert summary.split.permutation_p == expected
    assert state in summary.text
    assert f"exact p = {expected:.2f}" in summary.text
    assert summary.ratio == pytest.approx(13 / 8 if state == "groups overlap" else 2.5)


def test_sensitivity_series_reads_the_frozen_value_and_its_grid_neighbours(outputs):
    thresholds, series = cb_h1_figures.sensitivity_series()
    assert thresholds == sorted(outputs.runs)
    for t, summary in zip(thresholds, series["beta1"]):
        measures = cb_h1_figures.network_measures(outputs.runs[t])
        wky = np.mean([measures[s.specimen_id]["beta1"] for s in SPECIMENS if s.group == "WKY"])
        shr = np.mean([measures[s.specimen_id]["beta1"] for s in SPECIMENS if s.group == "SHR"])
        assert summary.ratio == pytest.approx(shr / wky)
    ratios = [s.ratio for s in series["beta1"]]
    assert ratios == sorted(ratios) and ratios[0] > 1


def test_fragmentation_onsets_keep_null_and_raise_without_the_key(outputs):
    onsets = cb_h1_figures.fragmentation_onsets()
    assert sum(o is None for o in onsets.values()) == 2
    path = outputs.results / "threshold_selection.json"
    path.write_text(json.dumps({"frozen": cb_settings.FROZEN_THRESHOLD}))
    with pytest.raises(KeyError, match="--stage threshold"):
        cb_h1_figures.fragmentation_onsets()
    path.write_text(json.dumps({"fragmentation_onset": {"WKY-A": 0.97}}))
    with pytest.raises(KeyError, match="WKY-B"):
        cb_h1_figures.fragmentation_onsets()


def _figure_texts(monkeypatch):
    """Capture every text drawn on each figure as it is closed."""
    captured = []
    real_close = cb_h1_figures.plt.close

    def close(fig):
        captured.append([t.get_text() for t in fig.findobj(
            lambda artist: hasattr(artist, "get_text") and artist.get_text())])
        real_close(fig)

    monkeypatch.setattr(cb_h1_figures.plt, "close", close)
    return captured


@pytest.mark.plotting
def test_main_writes_every_figure_with_text_computed_from_the_tables(outputs, monkeypatch):
    captured = _figure_texts(monkeypatch)
    cb_h1_figures.main()

    for name in ("figure1_network_density", "figure2_diameter_distribution",
                 "figure3_threshold_sensitivity", "figure7_node_degree",
                 "figure8_segment_length"):
        assert (outputs.results / f"{name}.png").exists()

    density, diameter, sensitivity, _, length = captured
    measures = cb_h1_figures.network_measures(outputs.results)
    for key, title, *_ in cb_h1_figures.MEASURES:
        expected = cb_h1_figures.group_summary({s: m[key] for s, m in measures.items()}).text
        assert expected in density
    _, series = cb_h1_figures.sensitivity_series()
    smallest = min(g.split.permutation_p for summaries in series.values() for g in summaries)
    assert any(f"smallest exact p {smallest:.2f}" in text for text in sensitivity)

    medians = cb_h1_figures.median_diameters(cb_h1_figures._load_diameters())
    gap = assess_cohort_split(medians).gap
    assert gap == pytest.approx(0.8)
    assert f"gap {gap:.2f} µm = {gap / cb_h1_figures.VOXEL_UM:.2f} of one step" in diameter

    assert any("(4 of 6 specimens)" in text for text in sensitivity)
    assert any("SHR exceeds WKY in 9 of 9 comparisons" in text for text in sensitivity)

    shares = cb_h1_figures.junction_trim_shares()
    low, high = 100 * min(shares.values()), 100 * max(shares.values())
    assert any(f"{low:.0f}-{high:.0f}% of edges" in text for text in length)


def test_the_junction_exclusion_is_the_pipeline_default():
    from carotid_image_to_model import HaemodynamicsConfig
    assert cb_h1_figures.junction_exclusion_um() == pytest.approx(
        HaemodynamicsConfig.__dataclass_fields__["edt_junction_proximity_exclusion_um"].default)


def test_the_threshold_stage_records_each_fragmentation_onset(monkeypatch, tmp_path):
    onsets = {s.specimen_id: (0.97 if i < 4 else None) for i, s in enumerate(SPECIMENS)}
    calls = iter(SPECIMENS)

    def fake_select(samples):
        specimen = next(calls)
        return SimpleNamespace(threshold=cb_settings.FROZEN_THRESHOLD,
                               fragmentation_onset=onsets[specimen.specimen_id],
                               format_table=lambda: "")

    sample = SimpleNamespace(threshold=cb_settings.FROZEN_THRESHOLD, foreground_fraction=0.2,
                             median_diameter_um=5.0, median_network_diameter_um=7.0)
    monkeypatch.setattr(cb_h1_batch, "_predicted", lambda: list(SPECIMENS))
    monkeypatch.setattr(cb_h1_batch, "place_roi",
                        lambda specimen, roi: SimpleNamespace(bounds=slice(None)))
    monkeypatch.setattr(cb_h1_batch, "read_ilastik_probabilities",
                        lambda *a, **k: np.zeros((2, 2, 2)))
    monkeypatch.setattr(cb_h1_batch, "sweep_thresholds", lambda *a, **k: [sample])
    monkeypatch.setattr(cb_h1_batch, "select_threshold", fake_select)
    monkeypatch.setattr(cb_h1_batch, "select_on_network_calibre",
                        lambda *a, **k: SimpleNamespace(threshold=None, reason="none"))
    monkeypatch.setattr(cb_h1_batch, "OUTPUT_DIR", tmp_path)

    cb_h1_batch.stage_threshold(cb_settings.ROI_VOXELS, [cb_settings.FROZEN_THRESHOLD])
    saved = json.loads((tmp_path / "threshold_selection.json").read_text())
    assert saved["fragmentation_onset"] == onsets
