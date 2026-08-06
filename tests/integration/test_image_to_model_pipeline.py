"""Integration test for the example image-to-model pipeline."""
from __future__ import annotations

import importlib.util
import pickle
import shutil
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
TESTS_DIR = REPO_ROOT / "tests"
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"
FIXTURE_H5 = REPO_ROOT / "tests" / "data" / "bundled_vessels_8_to_2.h5"


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
        input_path=input_tiff,
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
        starting_node_selection_method="coordinates",
        output_node_selection_method="coordinates",
        starting_node_coordinates=[(5.0, 5.0, 5.0)],
        output_node_coordinates=[(42.0, 42.0, 42.0)],
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
    pericytes_path = vtk_prefix.with_name(vtk_prefix.name + "_pericytes.vtp")
    nodes_path = vtk_prefix.with_name(vtk_prefix.name + "_nodes.vtp")

    assert skeleton_path.exists()
    assert graph_path.exists()
    assert vessels_path.exists()
    assert pericytes_path.exists()
    assert nodes_path.exists()

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_static] n_nodes={n_nodes}, n_edges={n_edges}")

    assert n_nodes == 11, f"Expected 11 nodes, got {n_nodes}"
    assert n_edges == 10, f"Expected 10 edges, got {n_edges}"


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
        input_path=input_tiff,
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
        starting_node_selection_method="coordinates",
        output_node_selection_method="volume",
        starting_node_coordinates=[(24.0, 24.0, 24.0)],
        output_node_volumes=[((0.0, 0.0, 0.0), (47.0, 47.0, 47.0))],
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
    # One export, after the solve: vessels and flow in a single file.
    vessels = vtk_prefix.with_name(vtk_prefix.name + "_vessels.vtp")
    assert vessels.exists()
    import pyvista as pv

    assert "flow_abs" in pv.read(str(vessels)).cell_data


@pytest.mark.integration
@pytest.mark.slow
def test_image_to_model_pipeline_probabilistic_artificial_comparison_cohort_reuse(tmp_path):
    """Ensure probabilistic artificial cohort is reused from comparison in final run."""
    pytest.importorskip("pyvista")

    input_tiff = tmp_path / "seven_vessel_noisy_3d_probabilistic.tif"
    shutil.copy(FIXTURE_TIFF, input_tiff)

    plot_dir = TESTS_DIR / "plots" / "plots_image_to_model_probabilistic_reuse"
    output_dir = TESTS_DIR / "outputs" / "image_to_model_probabilistic_reuse"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "integration_probabilistic_reuse"

    pipeline = _load_pipeline_module()
    from ImageLynx.haemodynamics import probability as probability_mod

    # Force uniform 0.8 map for final run so comparison constrained value aligns.
    # The diameter table is derived from the config, not a module constant.
    diameter_by_branch_order = pipeline.resolve_settings()["diameter_by_branch_order"]
    constriction_uniform_08 = {
        str(branch_order): 0.8 for branch_order in diameter_by_branch_order
    }
    probabilistic_call_args: list[dict | None] = []

    # The comparison scenarios and the final run all reach the periodic model
    # through this one module attribute, so a single patch records all three.
    original_probabilistic = (
        probability_mod.set_poiseuille_resistances_with_probabilistic_periodic_constrictions
    )

    def _recording_probabilistic(original_fn, call_log: list[dict | None]):
        def _wrapper(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["constriction_length"] = 8.0
            kwargs["constriction_spacing"] = 10.0
            active_map = kwargs.get("active_center_indices_by_edge")
            call_log.append(deepcopy(active_map) if active_map is not None else None)
            return original_fn(*args, **kwargs)

        return _wrapper

    probability_mod.set_poiseuille_resistances_with_probabilistic_periodic_constrictions = (  # type: ignore[attr-defined]
        _recording_probabilistic(original_probabilistic, probabilistic_call_args)
    )
    try:
        pipeline.image_to_model_pipeline(
            input_path=input_tiff,
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
            starting_node_coordinates=[(5.0, 5.0, 5.0)],
            output_node_coordinates=[(42.0, 42.0, 42.0)],
            starting_nodes=[],
            output_nodes=[],
            input_p_bc=1000.0,
            output_p_bc=500.0,
            min_stub_length=3.0,
            visualize_results=False,
            visualize_vtk=False,
            do_pericyte_construction=True,
            use_pericyte_mask_constriction=False,
            use_probabilistic_pericyte_constriction=True,
            pericyte_constriction_probability=0.8,
            run_pericyte_resistance_comparison=True,
            pericyte_comparison_baseline_value=1.0,
            pericyte_comparison_constricted_value=0.8,
            reuse_comparison_pericyte_cohort_for_main_run=True,
            constriction_by_branch_order=constriction_uniform_08,
        )
    finally:
        probability_mod.set_poiseuille_resistances_with_probabilistic_periodic_constrictions = (  # type: ignore[attr-defined]
            original_probabilistic
        )

    # Expect 3 calls:
    # 1) comparison baseline (None),
    # 2) comparison constricted (fixed non-empty map),
    # 3) final main run (same fixed map when reuse toggle is True).
    assert len(probabilistic_call_args) >= 3
    assert probabilistic_call_args[0] is None
    assert isinstance(probabilistic_call_args[1], dict)
    assert isinstance(probabilistic_call_args[2], dict)
    assert probabilistic_call_args[1] == probabilistic_call_args[2]
    assert any(len(v) > 0 for v in probabilistic_call_args[1].values())

    comparison_csv = output_dir / f"{input_tiff.stem}_pericyte_resistance_comparison.csv"
    assert comparison_csv.exists()


@pytest.mark.integration
@pytest.mark.slow
def test_image_to_model_pipeline_end_to_end_on_h5_bundle_fixture():
    """Run full pipeline against committed H5 fixture with bundled vessels."""
    pytest.importorskip("pyvista")
    h5py = pytest.importorskip("h5py")

    input_h5 = FIXTURE_H5
    if not input_h5.exists():
        pytest.skip(f"Missing H5 fixture: {input_h5}")

    plot_dir = TESTS_DIR / "plots" / "plots_image_to_model_h5_bundle"
    output_dir = TESTS_DIR / "outputs" / "image_to_model_h5_bundle"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "integration_h5_bundle"

    pipeline = _load_pipeline_module()
    previous_dataset_name = getattr(pipeline, "H5_DATASET_NAME", None)
    pipeline.H5_DATASET_NAME = "data"
    with h5py.File(input_h5, "r") as handle:
        shape = tuple(int(v) for v in handle["data"].shape)
    z_max = max(0, shape[0] - 1)
    y_max = max(0, shape[1] - 1)
    x_max = max(0, shape[2] - 1)
    input_x_hi = max(0, int(round(0.2 * x_max)))
    output_x_lo = max(0, int(round(0.8 * x_max)))
    try:
        pipeline.image_to_model_pipeline(
            input_path=input_h5,
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
            starting_node_selection_method="volume",
            output_node_selection_method="volume",
            starting_node_volumes=[
                (
                    (0.0, 0.0, 0.0),
                    (float(z_max), float(y_max), float(input_x_hi)),
                )
            ],
            output_node_volumes=[
                (
                    (0.0, 0.0, float(output_x_lo)),
                    (float(z_max), float(y_max), float(x_max)),
                )
            ],
            starting_nodes=[],
            output_nodes=[],
            input_p_bc=1000.0,
            output_p_bc=500.0,
            min_stub_length=3.0,
            visualize_results=False,
            visualize_vtk=False,
        )
    finally:
        pipeline.H5_DATASET_NAME = previous_dataset_name

    skeleton_path = output_dir / f"{input_h5.stem}_skeleton.npy"
    graph_path = output_dir / f"{input_h5.stem}_graph.pkl"
    vessels_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels.vtp")
    pericytes_path = vtk_prefix.with_name(vtk_prefix.name + "_pericytes.vtp")
    nodes_path = vtk_prefix.with_name(vtk_prefix.name + "_nodes.vtp")

    assert skeleton_path.exists()
    assert graph_path.exists()
    assert vessels_path.exists()
    assert pericytes_path.exists()
    assert nodes_path.exists()

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_h5_bundle] n_nodes={n_nodes}, n_edges={n_edges}")

    assert n_nodes == 10, f"Expected 10 nodes, got {n_nodes}"
    assert n_edges == 9, f"Expected 9 edges, got {n_edges}"
