"""Integration test for cropped Nerve capillaries pipeline run."""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

from haemolynx import haemodynamics
from haemolynx.io import crop_tiff_volume_from_corners


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
NERVE_TIFF = REPO_ROOT / "examples" / "images" / "Nerve_capillaries.tif"


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
def test_nerve_pipeline_on_cropped_last_z_quarter_bottom_y_half():
    """Crop Nerve TIFF then run full pipeline with range-based checks."""
    pytest.importorskip("pyvista")
    tifffile = pytest.importorskip("tifffile")
    if not NERVE_TIFF.exists():
        pytest.skip(f"Missing test input TIFF: {NERVE_TIFF}")

    source = tifffile.imread(NERVE_TIFF)
    z, y, x = source.shape[:3]
    z_start = int(0.75 * z)
    y_start = int(0.5 * y)
    corner_a = (z_start, y_start, 0)
    corner_b = (z - 1, y - 1, x - 1)

    tests_data_path = REPO_ROOT / "tests" / "data"
    cropped_tiff = tests_data_path / "Nerve_capillaries_cropped.tif"
    crop_info = crop_tiff_volume_from_corners(
        NERVE_TIFF,
        cropped_tiff,
        corner_a=corner_a,
        corner_b=corner_b,
    )
    assert cropped_tiff.exists()
    assert tuple(crop_info["cropped_shape"]) == (z - z_start, y - y_start, x)
    cropped_z, cropped_y, cropped_x = tuple(crop_info["cropped_shape"])
    # The crop is disconnected: after the 30% small-component filter the
    # surviving connected network sits in the low-y part of the image, so the
    # outlet band must reach 60% in from the far end to touch it, and the
    # inlet band is the complementary first 40%.
    #
    # The boxes below are in microns, like every node position -- these bounds
    # are voxel counts, so they are a generous over-estimate of the crop rather
    # than a measurement of it, which is all the split needs. Deriving them
    # from the physical extent would move the split and reselect the boundary
    # nodes, so it is left for whoever wants that on purpose.
    y_split = max(1, int(0.4 * cropped_y))

    plot_dir = REPO_ROOT / "tests" / "plots" / "plots_nerve"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir = REPO_ROOT / "tests" / "outputs" / "nerve_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "nerve_pipeline_test"
    pipeline = _load_pipeline_module()
    solved_graph = pipeline.image_to_model_pipeline(
        input_path=cropped_tiff,
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
        skeleton_min_component_percent=30.0,
        starting_node_selection_method="volume",
        output_node_selection_method="volume",
        starting_node_volumes=[((0.0, 0.0, 0.0), (float(cropped_z - 1), float(y_split - 1), float(cropped_x - 1)))],
        output_node_volumes=[((0.0, float(y_split), 0.0), (float(cropped_z - 1), float(cropped_y - 1), float(cropped_x - 1)))],
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=1000.0,
        output_p_bc=500.0,
        min_stub_length=3.0,
        visualize_results=False,
        visualize_vtk=False,
    )

    graph_path = output_dir / f"{cropped_tiff.stem}_graph.pkl"
    assert graph_path.exists()

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_nerve] n_nodes={n_nodes}, n_edges={n_edges}")
    assert n_nodes > 30, f"Expected at least 30 nodes, got {n_nodes}"
    assert n_edges > 30, f"Expected at least 30 edges, got {n_edges}"
    assert n_nodes < 1000, f"Expected less than 1000 nodes, got {n_nodes}"
    assert n_edges < 1000, f"Expected less than 1000 edges, got {n_edges}"

    # With the boundary conditions on the connected network, every edge must
    # conduct and the solved flows must be real, not roundoff on a dead
    # network (a broken BC placement solves to a uniform pressure and flows
    # ~14 orders of magnitude smaller).
    flows = [
        abs(data["flow_signed"])
        for _, _, data in solved_graph.edges(data=True)
        if "flow_signed" in data
    ]
    assert len(flows) == solved_graph.number_of_edges()
    max_flow = max(flows)
    assert max_flow > 1e-18, f"Solved flows are numerical noise: max {max_flow:.3e}"

    # Kirchhoff's current law at every node without an imposed pressure.
    boundary_nodes = {
        node
        for node, data in solved_graph.nodes(data=True)
        if data.get("pressure") in (1000.0, 500.0)
    }
    assert boundary_nodes, "No boundary nodes found at the imposed pressures"
    residuals = haemodynamics.flow_conservation_residuals(
        solved_graph, boundary_nodes=boundary_nodes
    )
    assert residuals, "No interior nodes to check conservation on"
    worst = max(abs(r) for r in residuals.values())
    assert worst < 1e-9 * max_flow, (
        f"Flow not conserved: worst residual {worst:.3e} vs max flow {max_flow:.3e}"
    )
