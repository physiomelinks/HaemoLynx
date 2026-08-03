"""Integration test: export pipeline tutorial notebook to Python and execute it."""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "tutorials"
TUTORIAL_NOTEBOOK = TUTORIAL_DIR / "pipeline_tutorial.ipynb"
GENERATED_SCRIPT = TUTORIAL_DIR / "pipeline_tutorial.py"
INPUT_TIFF = REPO_ROOT / "tests" / "data" / "Nerve_capillaries_cropped.tif"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_tutorial_notebook_converts_and_runs(tmp_path):
    """Export pipeline_tutorial.ipynb to pipeline_tutorial.py and run it."""
    pytest.importorskip("pyvista")
    pytest.importorskip("nbconvert")
    pytest.importorskip("nbformat")

    if not TUTORIAL_NOTEBOOK.exists():
        pytest.skip(f"Missing tutorial notebook: {TUTORIAL_NOTEBOOK}")
    if not INPUT_TIFF.exists():
        pytest.skip(f"Missing tutorial input TIFF: {INPUT_TIFF}")

    if str(TUTORIAL_DIR) not in sys.path:
        sys.path.insert(0, str(TUTORIAL_DIR))
    from export_notebook import export_pipeline_tutorial_script

    export_pipeline_tutorial_script(TUTORIAL_NOTEBOOK, GENERATED_SCRIPT)
    assert GENERATED_SCRIPT.exists()
    assert GENERATED_SCRIPT.stat().st_size > 0
    assert "# AUTO-GENERATED" in GENERATED_SCRIPT.read_text(encoding="utf-8")

    output_dir = tmp_path / "outputs"
    plot_dir = tmp_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    existing_pythonpath = __import__("os").environ.get("PYTHONPATH", "")
    extra_paths = f"{TUTORIAL_DIR}:{REPO_ROOT / 'src'}:{REPO_ROOT / 'examples'}"
    if existing_pythonpath:
        extra_paths = f"{extra_paths}:{existing_pythonpath}"
    env = {
        **dict(__import__("os").environ),
        "IMAGELYNX_REPO_ROOT": str(REPO_ROOT),
        "IMAGELYNX_TUTORIAL_OUTPUT_DIR": str(output_dir),
        "IMAGELYNX_TUTORIAL_PLOT_DIR": str(plot_dir),
        "PYTHONPATH": extra_paths,
    }
    result = subprocess.run(
        [sys.executable, str(GENERATED_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("stdout:\n", result.stdout)
        print("stderr:\n", result.stderr)
    assert result.returncode == 0, (
        f"Tutorial script failed with exit code {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )

    stem = INPUT_TIFF.stem
    graph_path = output_dir / f"{stem}_graph.pkl"
    vtk_prefix = output_dir / f"{stem}_tutorial"

    assert graph_path.exists(), f"Missing graph pickle: {graph_path}"

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[tutorial_integration] n_nodes={n_nodes}, n_edges={n_edges}")
    assert n_nodes > 50, f"Expected at least 50 nodes, got {n_nodes}"
    assert n_edges > 50, f"Expected at least 50 edges, got {n_edges}"
    assert n_nodes < 1000, f"Expected less than 1000 nodes, got {n_nodes}"
    assert n_edges < 1000, f"Expected less than 1000 edges, got {n_edges}"
