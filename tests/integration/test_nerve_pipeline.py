"""Integration test for cropped Nerve capillaries pipeline run."""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
import tifffile

from ImageLynx.io import crop_tiff_volume_from_corners


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


def _write_zsplit_large_vessel_masks(
    out_dir: Path,
    shape_zyx: tuple[int, ...],
    *,
    stem: str,
) -> tuple[Path, Path]:
    """Half-volume Z split: arteriole low-Z, venule high-Z (disjoint, no fast-mode overlap)."""
    z, y, x = (int(v) for v in shape_zyx[:3])
    art = np.zeros((z, y, x), dtype=np.uint8)
    ven = np.zeros((z, y, x), dtype=np.uint8)
    split = max(1, z // 2)
    art[:split, :, :] = 1
    ven[split:, :, :] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / f"{stem}_dummy_art.tif"
    ven_path = out_dir / f"{stem}_dummy_ven.tif"
    tifffile.imwrite(art_path, art)
    tifffile.imwrite(ven_path, ven)
    return art_path, ven_path


@pytest.mark.integration
@pytest.mark.slow
def test_nerve_pipeline_on_cropped_last_z_quarter_bottom_y_half(tmp_path):
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
    shape_zyx = tuple(int(v) for v in tifffile.imread(cropped_tiff).shape[:3])
    art_path, ven_path = _write_zsplit_large_vessel_masks(
        tmp_path / "nerve_masks", shape_zyx, stem="nerve"
    )

    plot_dir = REPO_ROOT / "tests" / "plots" / "plots_nerve"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_dir = REPO_ROOT / "tests" / "outputs" / "nerve_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "nerve_pipeline_test"
    pipeline = _load_pipeline_module()
    pipeline.image_to_model_pipeline(
        image_path=cropped_tiff,
        plot_dir=plot_dir,
        vtk_output_prefix=vtk_prefix,
        verbose_logging=False,
        use_ilastik_segmentation=False,
        use_large_vessel_masks=True,
        use_ilastik_large_vessel_segmentation=False,
        automated_vessel_assignment=True,
        automated_vessel_assignment_use_legacy_mode=True,
        large_vessel_mask_dilation_microns=40.0,
        large_arteriole_mask_path=art_path,
        large_venule_mask_path=ven_path,
        do_skeletonize=True,
        do_graph_building=True,
        do_equiv_resistance_calculation=False,
        skeleton_closing_radius=1,
        skeleton_bridge_gap_size=1,
        skeleton_min_branch_length=3,
        skeleton_max_bridge_distance=2,
        skeleton_component_connectivity=3,
        skeleton_min_component_percent=1.0,
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=1000.0,
        output_p_bc=500.0,
        min_stub_length=3.0,
        final_render_mode="2d",
        visualize_results=False,
        visualize_vtk=False,
    )

    graph_path = output_dir / f"{cropped_tiff.stem}_graph.pkl"
    vessels_flow_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels_flow.vtp")
    final_graph_projection_path = plot_dir / "final_graph.png"
    assert graph_path.exists()
    assert vessels_flow_path.exists()
    assert final_graph_projection_path.exists()

    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"[integration_nerve] n_nodes={n_nodes}, n_edges={n_edges}")
    assert n_nodes > 50, f"Expected at least 50 nodes, got {n_nodes}"
    assert n_edges > 50, f"Expected at least 50 edges, got {n_edges}"
    assert n_nodes < 1000, f"Expected less than 1000 nodes, got {n_nodes}"
    assert n_edges < 1000, f"Expected less than 1000 edges, got {n_edges}"
