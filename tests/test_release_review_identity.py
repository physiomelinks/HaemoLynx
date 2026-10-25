"""The same input and settings, run twice, give identical outputs
(RELEASE_REVIEW.md `repro.6c`), end to end on the seven-vessel fixture.

Both runs write into the same folder, one after the other, so a path a file
records is the same in both; the release review's comparison then reads
every output: the CSVs byte for byte, the graph node for node and edge for
edge with every attribute, the VTK files array for array, and the manifest
outside the times and memory it exists to record. Plots may differ in their
bytes (render IDs) and are not counted.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from haemolynx.pipeline.schema import SCHEMA
from haemolynx.pipeline.settings import resolve_settings
from haemolynx.pipeline.stages import run_pipeline_stages
from release_review.exports import compare_folders

FIXTURE = Path(__file__).resolve().parent / "data" / "seven_vessel_noisy_3d.tif"


def _run_into(folder: Path) -> None:
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update({
        "input_path": FIXTURE,
        "vtk_output_prefix": folder / "run",
        "plot_dir": folder / "plots",
        "statistics": True,
        "show_plots_in_ide": False,
        "interactive_plots": False,
    })
    run_pipeline_stages(resolve_settings(values, schema=SCHEMA, config_path=None), SCHEMA)


@pytest.mark.slow
@pytest.mark.integration
def test_a_run_repeated_gives_identical_outputs(tmp_path):
    same = tmp_path / "run"
    _run_into(same)
    first = shutil.move(str(same), str(tmp_path / "first"))
    _run_into(same)
    second = shutil.move(str(same), str(tmp_path / "second"))

    differences = compare_folders(Path(first), Path(second))

    assert any(Path(first).glob("*_statistics.csv")) and any(Path(first).glob("*_graph.pkl"))
    assert [f"{each.file}: {each.detail}" for each in differences if each.counted] == []
