"""Integration test for the example image-to-model pipeline."""
from __future__ import annotations

import importlib.util
import pickle
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
TESTS_DIR = REPO_ROOT / "tests"
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "examples_resistance_network_pipeline",
        PIPELINE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.slow
def test_image_to_model_pipeline_end_to_end_on_static_tiff(tmp_path):
    """Run full pipeline against committed synthetic TIFF fixture."""
    pytest.importorskip("pyvista")

    input_tiff = tmp_path / "seven_vessel_noisy_3d.tif"
    shutil.copy(FIXTURE_TIFF, input_tiff)

    plot_dir = TESTS_DIR / "plots" / "plots_image_to_model_static"
    output_dir = TESTS_DIR / "outputs" / "image_to_model_static"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "integration_network"

    pipeline = _load_pipeline_module()
    pipeline.image_to_model_pipeline(
        image_path=input_tiff,
        plot_dir=plot_dir,
        vtk_output_prefix=vtk_prefix,
        verbose_logging=False,
        do_skeletonize=True,
        do_graph_building=True,
        do_equiv_resistance_calculation=False,
        skeleton_closing_radius=1,
        skeleton_bridge_gap_size=1,
        skeleton_min_branch_length=3,
        skeleton_max_bridge_distance=2,
        skeleton_component_connectivity=3,
        skeleton_min_component_percent=1.0,
        set_input_node_method="edge_percent",
        set_output_node_method="edge_percent",
        edge_percent=30.0,
        end_percent=20.0,
        node_edge_axis=1,
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=1000.0,
        output_p_bc=500.0,
        min_stub_length=3.0,
        visualize_results=False,
        visualize_vtk=False,
    )

    skeleton_path = output_dir / f"{input_tiff.stem}_skeleton.npy"
    graph_path = output_dir / f"{input_tiff.stem}_graph.pkl"
    vessels_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels.vtp")
    vessels_flow_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels_flow.vtp")
    pericytes_path = vtk_prefix.with_name(vtk_prefix.name + "_pericytes.vtp")
    nodes_path = vtk_prefix.with_name(vtk_prefix.name + "_nodes.vtp")

    assert skeleton_path.exists()
    assert graph_path.exists()
    assert vessels_path.exists()
    assert vessels_flow_path.exists()
    assert pericytes_path.exists()
    assert nodes_path.exists()

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_static] n_nodes={n_nodes}, n_edges={n_edges}")

    assert n_nodes == 8, f"Expected 8 nodes, got {n_nodes}"
    assert n_edges == 7, f"Expected 7 edges, got {n_edges}"


@pytest.mark.integration
@pytest.mark.slow
def test_image_to_model_pipeline_coordinate_input_volume_output(tmp_path):
    """Run pipeline with coordinates for inlets and volume boxes for outlets."""
    pytest.importorskip("pyvista")

    input_tiff = tmp_path / "seven_vessel_noisy_3d_coords_volume.tif"
    shutil.copy(FIXTURE_TIFF, input_tiff)

    plot_dir = TESTS_DIR / "plots" / "plots_image_to_model_coords_volume"
    output_dir = TESTS_DIR / "outputs" / "image_to_model_coords_volume"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "integration_coords_volume"

    pipeline = _load_pipeline_module()
    pipeline.image_to_model_pipeline(
        image_path=input_tiff,
        plot_dir=plot_dir,
        vtk_output_prefix=vtk_prefix,
        verbose_logging=False,
        do_skeletonize=True,
        do_graph_building=True,
        do_equiv_resistance_calculation=False,
        skeleton_closing_radius=1,
        skeleton_bridge_gap_size=1,
        skeleton_min_branch_length=3,
        skeleton_max_bridge_distance=2,
        skeleton_component_connectivity=3,
        skeleton_min_component_percent=1.0,
        set_input_node_method="coordinates",
        set_output_node_method="volume",
        starting_node_coordinates=[(24.0, 24.0, 24.0)],
        output_node_volumes=[((0.0, 0.0, 0.0), (47.0, 47.0, 47.0))],
        edge_percent=30.0,
        end_percent=20.0,
        node_edge_axis=1,
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=1000.0,
        output_p_bc=500.0,
        min_stub_length=3.0,
        visualize_results=False,
        visualize_vtk=False,
    )

    graph_path = output_dir / f"{input_tiff.stem}_graph.pkl"
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_coords_volume] n_nodes={n_nodes}, n_edges={n_edges}")

    assert n_nodes > 0
    assert n_edges > 0
    assert vtk_prefix.with_name(vtk_prefix.name + "_vessels_flow.vtp").exists()
