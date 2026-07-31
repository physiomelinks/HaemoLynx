"""Integration tests for user-selectable input axis order in the example pipeline."""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "resistance_network_pipeline.py"
TESTS_DIR = REPO_ROOT / "tests"
FIXTURE_TIFF = REPO_ROOT / "tests" / "data" / "seven_vessel_noisy_3d.tif"

VOXEL_SIZE_XYZ = (0.4, 0.5, 2.0)


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "examples_resistance_network_pipeline_axis_order",
        PIPELINE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_pipeline(pipeline, *, image_path, plot_dir, output_dir, axis_order):
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "axis_order_network"
    pipeline.image_to_model_pipeline(
        image_path=image_path,
        plot_dir=plot_dir,
        vtk_output_prefix=vtk_prefix,
        axis_order=axis_order,
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
        # "coordinates" snaps to the nearest terminal node, so opposite physical
        # corners always resolve to a node regardless of voxel anisotropy.
        starting_node_selection_method="coordinates",
        output_node_selection_method="coordinates",
        starting_node_coordinates=[(0.0, 0.0, 0.0)],
        output_node_coordinates=[(200.0, 100.0, 100.0)],
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=1000.0,
        output_p_bc=500.0,
        min_stub_length=3.0,
        run_haemodynamics=False,
        visualize_results=False,
        visualize_vtk=False,
    )
    with (output_dir / f"{image_path.stem}_graph.pkl").open("rb") as fh:
        return pickle.load(fh)


def _write_anisotropic_tiff(volume: np.ndarray, path: Path) -> None:
    tifffile.imwrite(
        str(path),
        volume,
        imagej=True,
        resolution=(1.0 / VOXEL_SIZE_XYZ[0], 1.0 / VOXEL_SIZE_XYZ[1]),
        metadata={"spacing": VOXEL_SIZE_XYZ[2], "unit": "um"},
    )


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_uses_array_axis_spacing_for_anisotropic_voxels(tmp_path):
    """Graph geometry must be scaled with (z, y, x) spacing, not metadata (x, y, z)."""
    pytest.importorskip("skan")

    volume = tifffile.imread(str(FIXTURE_TIFF))
    input_tiff = tmp_path / "anisotropic_axis_order.tif"
    _write_anisotropic_tiff(volume, input_tiff)

    pipeline = _load_pipeline_module()
    output_dir = TESTS_DIR / "outputs" / "axis_order_anisotropic"
    graph = _run_pipeline(
        pipeline,
        image_path=input_tiff,
        plot_dir=TESTS_DIR / "plots" / "plots_axis_order_anisotropic",
        output_dir=output_dir,
        axis_order="zyx",
    )

    # Metadata is recorded in (x, y, z) as reported by the file.
    voxel_meta = json.loads((output_dir / f"{input_tiff.stem}_voxel_size.json").read_text())
    assert tuple(voxel_meta["voxel_size"]) == pytest.approx(VOXEL_SIZE_XYZ)
    # Graph geometry is scaled with the reverse — spacing per array axis.
    assert graph.graph["voxel_size"] == pytest.approx((2.0, 0.5, 0.4))

    positions = np.array([data["pos"] for _, data in graph.nodes(data=True)])
    # Axis 0 spans 2.0 um per voxel, axis 2 only 0.4 um, so the physical extent of
    # axis 0 must exceed that of axis 2 for a roughly cubic volume.
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert extent[0] > extent[2]


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_axis_order_transposed_input_matches_canonical_run(tmp_path):
    """An (x, y, z)-ordered file with axis_order='xyz' reproduces the canonical model."""
    pytest.importorskip("skan")

    volume = tifffile.imread(str(FIXTURE_TIFF))
    canonical_tiff = tmp_path / "canonical.tif"
    transposed_tiff = tmp_path / "transposed.tif"
    _write_anisotropic_tiff(volume, canonical_tiff)
    # Same data stored as (x, y, z) instead of (z, y, x).
    _write_anisotropic_tiff(np.transpose(volume, (2, 1, 0)), transposed_tiff)

    pipeline = _load_pipeline_module()
    canonical_graph = _run_pipeline(
        pipeline,
        image_path=canonical_tiff,
        plot_dir=TESTS_DIR / "plots" / "plots_axis_order_canonical",
        output_dir=TESTS_DIR / "outputs" / "axis_order_canonical",
        axis_order="zyx",
    )
    transposed_graph = _run_pipeline(
        pipeline,
        image_path=transposed_tiff,
        plot_dir=TESTS_DIR / "plots" / "plots_axis_order_transposed",
        output_dir=TESTS_DIR / "outputs" / "axis_order_transposed",
        axis_order="xyz",
    )

    assert transposed_graph.number_of_nodes() == canonical_graph.number_of_nodes()
    assert transposed_graph.number_of_edges() == canonical_graph.number_of_edges()

    def total_length(graph):
        return sum(float(d["length"]) for _, _, d in graph.edges(data=True))

    assert total_length(transposed_graph) == pytest.approx(
        total_length(canonical_graph), rel=1e-9
    )

    def sorted_positions(graph):
        pos = np.array([data["pos"] for _, data in graph.nodes(data=True)])
        return pos[np.lexsort(pos.T[::-1])]

    np.testing.assert_allclose(
        sorted_positions(transposed_graph),
        sorted_positions(canonical_graph),
        rtol=1e-9,
        atol=1e-9,
    )
