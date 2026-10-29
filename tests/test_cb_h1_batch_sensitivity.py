"""`cb_h1_batch.py --stage sensitivity` moves the flood threshold and nothing else.

Given ``--hysteresis-low`` alone the pipeline seeds at low + 0.05, so the 0.93 run used to seed at
0.98 while the frozen run seeds at 0.999. The stage now passes the frozen seed explicitly.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cb_h1_batch
from ImageLynx import cb_settings
from ImageLynx.statistics.threshold_selection import ThresholdSample


def test_the_neighbours_are_the_frozen_value_s_grid_neighbours():
    grid = sorted(cb_settings.THRESHOLD_GRID)
    i = grid.index(cb_settings.FROZEN_THRESHOLD)
    assert cb_h1_batch.sensitivity_thresholds(cb_settings.FROZEN_THRESHOLD) == (
        grid[i - 1], grid[i + 1])


@pytest.mark.parametrize("bad", [0.94, min(cb_settings.THRESHOLD_GRID),
                                 max(cb_settings.THRESHOLD_GRID)])
def test_a_value_off_the_grid_or_at_its_end_raises(bad):
    with pytest.raises(ValueError):
        cb_h1_batch.sensitivity_thresholds(bad)


def test_every_run_passes_the_frozen_seed_explicitly(monkeypatch, tmp_path):
    commands = []

    class _Done:
        returncode = 1          # non-zero, so no ROI check runs on the fake output

    def fake_run(command, **kwargs):
        commands.append(command)
        return _Done()

    class _Placement:
        centre_zyx = (0, 0, 0)

    specimens = [s for s in cb_h1_batch.SPECIMENS][:2]
    monkeypatch.setattr(cb_h1_batch.subprocess, "run", fake_run)
    monkeypatch.setattr(cb_h1_batch, "_predicted", lambda: specimens)
    monkeypatch.setattr(cb_h1_batch, "place_roi", lambda specimen, roi: _Placement())
    monkeypatch.setattr(cb_h1_batch, "SENSITIVITY_DIR", tmp_path)

    cb_h1_batch.stage_sensitivity(cb_settings.ROI_VOXELS, cb_settings.FROZEN_THRESHOLD)

    lows = cb_h1_batch.sensitivity_thresholds(cb_settings.FROZEN_THRESHOLD)
    assert len(commands) == len(lows) * len(specimens)
    for command in commands:
        low = float(command[command.index("--hysteresis-low") + 1])
        high = float(command[command.index("--hysteresis-high") + 1])
        out = Path(command[command.index("--output-dir") + 1])
        specimen = command[command.index("--specimen") + 1]
        assert low in lows
        assert high == cb_settings.HYSTERESIS_HIGH
        assert out == tmp_path / f"t{low:.2f}" / specimen


def test_the_frozen_run_command_is_unchanged(monkeypatch, tmp_path):
    """--stage run still passes the low bound alone, so its outputs stay byte-identical."""
    commands = []

    class _Done:
        returncode = 1

    class _Placement:
        centre_zyx = (0, 0, 0)

    monkeypatch.setattr(cb_h1_batch.subprocess, "run",
                        lambda command, **kw: commands.append(command) or _Done())
    monkeypatch.setattr(cb_h1_batch, "_predicted", lambda: cb_h1_batch.SPECIMENS[:1])
    monkeypatch.setattr(cb_h1_batch, "place_roi", lambda specimen, roi: _Placement())
    monkeypatch.setattr(cb_h1_batch, "OUTPUT_DIR", tmp_path)

    cb_h1_batch.stage_run(cb_settings.ROI_VOXELS, cb_settings.FROZEN_THRESHOLD)
    assert "--hysteresis-high" not in commands[0]
    assert commands[0][commands[0].index("--hysteresis-low") + 1] == str(
        cb_settings.FROZEN_THRESHOLD)


@pytest.fixture
def threshold_stage(monkeypatch, tmp_path):
    """Run ``--stage threshold`` over every specimen on a stubbed sweep, without the data.

    Call it with the sweep each specimen returns and the grid; it returns the frozen value and
    the ``threshold_selection.json`` record the stage wrote.
    """
    monkeypatch.setattr(cb_h1_batch, "_predicted", lambda: list(cb_h1_batch.SPECIMENS))
    monkeypatch.setattr(cb_h1_batch, "place_roi",
                        lambda specimen, roi: SimpleNamespace(bounds=(slice(None),) * 3))
    monkeypatch.setattr(cb_h1_batch, "read_ilastik_probabilities",
                        lambda *a, **k: np.zeros((2, 2, 2)))
    monkeypatch.setattr(cb_h1_batch, "OUTPUT_DIR", tmp_path)

    def run(sweep, grid):
        monkeypatch.setattr(cb_h1_batch, "sweep_thresholds", lambda *a, **k: sweep)
        frozen = cb_h1_batch.stage_threshold(cb_settings.ROI_VOXELS, grid)
        return frozen, json.loads((tmp_path / "threshold_selection.json").read_text())

    return run


def test_the_network_calibre_check_is_recorded_but_never_frozen(threshold_stage):
    """--stage threshold freezes on the plain cut; the d_net check sits beside it."""
    def sample(threshold, d_med, d_net):
        return ThresholdSample(
            threshold=threshold, foreground_fraction=0.3, median_diameter_um=d_med,
            p90_diameter_um=2 * d_med, median_voxel_diameter_um=d_med,
            mask_components=10, mask_components_above_floor=10,
            largest_mask_component_share=0.99, skeleton_length_mm=1.0, endpoints=4,
            endpoint_density_per_mm=4.0, skeleton_components=1,
            median_network_diameter_um=d_net)

    # Plain cut picks 0.95; d_net less one voxel picks 0.97 (d_net 6.46 -> 4.59 um).
    sweep = [sample(0.93, 5.27, 7.46), sample(0.95, 5.27, 7.46), sample(0.97, 3.73, 6.46)]

    frozen, record = threshold_stage(sweep, [0.93, 0.95, 0.97])

    assert frozen == record["frozen"] == 0.95
    assert set(record["per_specimen"].values()) == {0.95}
    check = record["robustness_network_calibre_less_voxel"]
    assert check["frozen"] == 0.97
    assert set(check["per_specimen"].values()) == {0.97}


def test_the_threshold_stage_records_each_fragmentation_onset(threshold_stage, monkeypatch):
    onsets = {s.specimen_id: (0.97 if i < 4 else None)
              for i, s in enumerate(cb_h1_batch.SPECIMENS)}
    calls = iter(cb_h1_batch.SPECIMENS)

    def fake_select(samples):
        specimen = next(calls)
        return SimpleNamespace(threshold=cb_settings.FROZEN_THRESHOLD,
                               fragmentation_onset=onsets[specimen.specimen_id],
                               format_table=lambda: "")

    sample = SimpleNamespace(threshold=cb_settings.FROZEN_THRESHOLD, foreground_fraction=0.2,
                             median_diameter_um=5.0, median_network_diameter_um=7.0)
    monkeypatch.setattr(cb_h1_batch, "select_threshold", fake_select)
    monkeypatch.setattr(cb_h1_batch, "select_on_network_calibre",
                        lambda *a, **k: SimpleNamespace(threshold=None, reason="none"))

    _, saved = threshold_stage([sample], [cb_settings.FROZEN_THRESHOLD])
    assert saved["fragmentation_onset"] == onsets
