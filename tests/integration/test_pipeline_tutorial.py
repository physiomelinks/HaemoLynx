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
        "HAEMOLYNX_REPO_ROOT": str(REPO_ROOT),
        "HAEMOLYNX_TUTORIAL_OUTPUT_DIR": str(output_dir),
        "HAEMOLYNX_TUTORIAL_PLOT_DIR": str(plot_dir),
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


@pytest.mark.integration
@pytest.mark.slow
def test_the_tutorial_runs_without_the_repository(tmp_path):
    """The notebook promises `pip install HaemoLynx` is enough. Check that.

    The exported script is copied out of the repository and run from a
    directory that contains nothing else, so the checkout is not discoverable:
    no `tests/data` mask, no `examples/` config, no `tutorial_plots`. It has to
    fall back to the schema's default settings and the synthetic vessel volume
    it builds itself, which is what a pip-installed user gets.
    """
    import os
    import shutil

    pytest.importorskip("pyvista")
    pytest.importorskip("nbconvert")

    if not GENERATED_SCRIPT.exists():
        pytest.skip("run test_pipeline_tutorial_notebook_converts_and_runs first")

    sandbox = tmp_path / "elsewhere"
    sandbox.mkdir()
    script = sandbox / GENERATED_SCRIPT.name
    shutil.copy(GENERATED_SCRIPT, script)

    env = {
        **os.environ,
        "HAEMOLYNX_TUTORIAL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "HAEMOLYNX_TUTORIAL_PLOT_DIR": str(tmp_path / "plots"),
        # Only the library, exactly as an install provides it.
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "MPLBACKEND": "Agg",
        "PYVISTA_OFF_SCREEN": "true",
    }
    env.pop("HAEMOLYNX_REPO_ROOT", None)
    env.pop("HAEMOLYNX_TUTORIAL_INPUT_TIFF", None)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(sandbox), env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"the tutorial needs the repository after all:\n{result.stdout}\n{result.stderr}"
    )
    assert "synthetic volume" in result.stdout, (
        "expected the fallback volume to be used when there is no mask to hand"
    )

    graph_path = tmp_path / "outputs" / "synthetic_vessels_graph.pkl"
    assert graph_path.exists(), f"missing graph from the synthetic run: {graph_path}"
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    # The phantom is a trunk splitting twice: 7 vessels, 8 nodes.
    assert graph.number_of_edges() == 7
    assert graph.number_of_nodes() == 8
    # The graph is pickled at Stage 2, before haemodynamics; the later stages
    # are evidenced by what they write.
    assert (tmp_path / "outputs" / "tutorial_vessels.vtp").exists(), (
        "the VTK export did not run on the synthetic network"
    )
    assert (tmp_path / "outputs" / "synthetic_vessels_statistics.csv").exists(), (
        "the statistics export did not run on the synthetic network"
    )
    assert "Equivalent resistance" in result.stdout
