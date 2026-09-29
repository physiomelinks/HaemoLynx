"""`cb_h1_batch.py --stage sensitivity` moves the flood threshold and nothing else (open item 41).

Given ``--hysteresis-low`` alone the pipeline seeds at low + 0.05, so the 0.93 run used to seed at
0.98 while the frozen run seeds at 0.999. The stage now passes the frozen seed explicitly.
"""
from pathlib import Path

import pytest

import cb_h1_batch
from ImageLynx import cb_settings


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
