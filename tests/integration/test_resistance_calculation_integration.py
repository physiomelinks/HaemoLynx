"""Integration tests for resistance/flow/pressure on a diagonal branching network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "resistance_network_pipeline.py"

INPUT_PRESSURE_PA = 1000.0
OUTPUT_PRESSURE_PA = 400.0
DELTA_P_PA = INPUT_PRESSURE_PA - OUTPUT_PRESSURE_PA
DESIGN_RADIUS_VOX = 2.5
EXPECTED_ORDER_COUNTS = {"B01": 1, "B02": 2, "B03": 4, "B04": 2, "B05": 1}


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


def _build_diagonal_synthetic_mask(shape: tuple[int, int, int] = (96, 96, 96)) -> np.ndarray:
    """Create a segmented vessel mask with symmetric diagonal branches."""
    mask = np.zeros(shape, dtype=bool)
    node_pos = {
        0: (48.0, 48.0, 8.0),
        1: (48.0, 48.0, 20.0),
        2: (40.0, 36.0, 32.0),
        3: (56.0, 60.0, 32.0),
        4: (34.0, 30.0, 44.0),
        5: (46.0, 42.0, 44.0),
        6: (50.0, 54.0, 44.0),
        7: (62.0, 66.0, 44.0),
        8: (40.0, 36.0, 60.0),
        9: (56.0, 60.0, 60.0),
        10: (48.0, 48.0, 72.0),
        11: (48.0, 48.0, 84.0),
    }
    centerline_edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7),
        (4, 8),
        (5, 8),
        (6, 9),
        (7, 9),
        (8, 10),
        (9, 10),
        (10, 11),
    ]

    def draw_tube(
        p0_xyz: tuple[float, float, float],
        p1_xyz: tuple[float, float, float],
        radius_vox: float,
    ) -> None:
        p0 = np.asarray(p0_xyz, dtype=float)
        p1 = np.asarray(p1_xyz, dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        n_samples = int(np.ceil(seg_len * 2.0)) + 1
        r = int(np.ceil(radius_vox))
        zz_max, yy_max, xx_max = np.asarray(mask.shape) - 1
        for t in np.linspace(0.0, 1.0, n_samples, dtype=float):
            p = (1.0 - t) * p0 + t * p1
            zc, yc, xc = p
            z0 = max(0, int(np.floor(zc - r)))
            z1 = min(zz_max, int(np.ceil(zc + r)))
            y0 = max(0, int(np.floor(yc - r)))
            y1 = min(yy_max, int(np.ceil(yc + r)))
            x0 = max(0, int(np.floor(xc - r)))
            x1 = min(xx_max, int(np.ceil(xc + r)))
            zz, yy, xx = np.ogrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            inside = (
                (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= radius_vox**2
            )
            mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] |= inside

    for u, v in centerline_edges:
        draw_tube(node_pos[u], node_pos[v], DESIGN_RADIUS_VOX)
    return mask


def _build_diagonal_synthetic_raw(mask: np.ndarray) -> np.ndarray:
    """Create a smooth raw-intensity proxy from the same vessel pattern."""
    raw = gaussian_filter(mask.astype(np.float32), sigma=1.1)
    if float(np.max(raw)) > 0.0:
        raw = raw / float(np.max(raw))
    raw = (raw * 4095.0).astype(np.uint16)
    return raw


def _branch_order_diameters_from_radius(radius_vox: float) -> dict[str, float]:
    diameter = 2.0 * float(radius_vox)
    out: dict[str, float] = {}
    for i in range(1, 26):
        out[f"B{i:02d}"] = diameter
        out[f"Art{i}"] = diameter
        out[f"Ven{i}"] = diameter
    return out


def _percent_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else float("inf")
    return abs(actual - expected) / abs(expected) * 100.0


def _run_hemodynamics_assertions(
    vessels,
    *,
    diameter_each: np.ndarray,
    network_scalar_diameter: float,
    report_title: str,
    radius_dependent: bool,
    resistance_rel_tol: float = 5e-3,
    bo3_rel_tol: float = 5e-2,
    pressure_rel_tol: float = 4e-2,
) -> None:
    orders = np.asarray(vessels.cell_data["branch_order"]).astype(str)
    resistances = np.asarray(vessels.cell_data["resistance"], dtype=float)
    flow_abs = np.asarray(vessels.cell_data["flow_abs"], dtype=float)
    edge_u = np.asarray(vessels.cell_data["edge_u"], dtype=int)
    edge_v = np.asarray(vessels.cell_data["edge_v"], dtype=int)
    p_u = np.asarray(vessels.cell_data["pressure_u"], dtype=float)
    p_v = np.asarray(vessels.cell_data["pressure_v"], dtype=float)

    assert len(orders) == vessels.n_cells
    assert len(resistances) == vessels.n_cells
    assert len(diameter_each) == vessels.n_cells

    unique, counts = np.unique(orders, return_counts=True)
    count_map = {str(k): int(v) for k, v in zip(unique, counts)}
    filtered_counts = {k: count_map.get(k, 0) for k in EXPECTED_ORDER_COUNTS}
    assert filtered_counts == EXPECTED_ORDER_COUNTS

    sized = vessels.compute_cell_sizes(length=True, area=False, volume=False)
    lengths = np.asarray(sized.cell_data["Length"], dtype=float)
    assert len(lengths) == vessels.n_cells

    mu_each = 1.0 / (diameter_each**1.647)
    r_expected_each = (128.0 * mu_each * lengths) / (np.pi * (diameter_each**4))

    d_scalar = float(network_scalar_diameter)
    mu_scalar = 1.0 / (d_scalar**1.647)
    r_expected_each_scalar = (128.0 * mu_scalar * lengths) / (np.pi * (d_scalar**4))

    resistance_err_individual_pct = float(
        np.max(
            np.abs(
                (resistances - r_expected_each)
                / np.maximum(np.abs(r_expected_each), 1e-12)
            )
        )
        * 100.0
    )
    resistance_err_total_pct = float(
        np.max(
            np.abs(
                (resistances - r_expected_each_scalar)
                / np.maximum(np.abs(r_expected_each_scalar), 1e-12)
            )
        )
        * 100.0
    )

    if radius_dependent:
        r_expected = r_expected_each
        r_by_order_source = r_expected_each
        q_model_label = "radius_dependent"
    else:
        r_expected = r_expected_each_scalar
        r_by_order_source = r_expected_each_scalar
        q_model_label = "radius_independent"

    assert np.allclose(resistances, r_expected, rtol=resistance_rel_tol, atol=1e-9)

    r_by_order: dict[str, float] = {}
    for order in EXPECTED_ORDER_COUNTS:
        idx = np.where(orders == order)[0]
        vals = r_by_order_source[idx]
        assert len(vals) == EXPECTED_ORDER_COUNTS[order]
        r_by_order[order] = float(np.mean(vals))

    req_analytic = (
        r_by_order["B01"]
        + 0.5 * (r_by_order["B02"] + 0.5 * r_by_order["B03"] + r_by_order["B04"])
        + r_by_order["B05"]
    )
    q_total_analytic = DELTA_P_PA / req_analytic
    q_bo3_analytic = q_total_analytic / 4.0

    bo3_idx = int(np.where(orders == "B03")[0][0])
    bo3_flow_err_pct = _percent_error(
        float(flow_abs[bo3_idx]),
        float(q_bo3_analytic),
    )
    assert flow_abs[bo3_idx] == pytest.approx(q_bo3_analytic, rel=bo3_rel_tol, abs=1e-8)

    p_final_analytic = OUTPUT_PRESSURE_PA + q_total_analytic * r_by_order["B05"]

    incident_orders: dict[int, list[str]] = {}
    incident_pressures: dict[int, list[float]] = {}
    for i in range(vessels.n_cells):
        ou = str(orders[i])
        uu = int(edge_u[i])
        vv = int(edge_v[i])
        incident_orders.setdefault(uu, []).append(ou)
        incident_orders.setdefault(vv, []).append(ou)
        incident_pressures.setdefault(uu, []).append(float(p_u[i]))
        incident_pressures.setdefault(vv, []).append(float(p_v[i]))

    inlet_bo3_nodes = [
        node_id
        for node_id, ords in incident_orders.items()
        if ords.count("B02") == 1 and ords.count("B03") == 2
    ]
    assert inlet_bo3_nodes
    inlet_bo3_node = int(min(inlet_bo3_nodes))
    p_inlet_bo3_solved = float(np.median(incident_pressures[inlet_bo3_node]))
    p_inlet_bo3_analytic = (
        INPUT_PRESSURE_PA
        - q_total_analytic * r_by_order["B01"]
        - (q_total_analytic / 2.0) * r_by_order["B02"]
    )
    p_inlet_bo3_err_pct = _percent_error(
        p_inlet_bo3_solved,
        p_inlet_bo3_analytic,
    )
    assert p_inlet_bo3_solved == pytest.approx(
        p_inlet_bo3_analytic,
        rel=pressure_rel_tol,
        abs=2e-2,
    )

    rows = [
        (
            f"resistance_{q_model_label}",
            resistance_err_individual_pct if radius_dependent else resistance_err_total_pct,
            resistance_rel_tol * 100.0,
        ),
        (f"flow_bo3_{q_model_label}", bo3_flow_err_pct, bo3_rel_tol * 100.0),
        (
            f"pressure_bo3_inlet_{q_model_label}",
            p_inlet_bo3_err_pct,
            pressure_rel_tol * 100.0,
        ),
    ]
    print(f"\nError summary (%) - {report_title}:")
    print("| metric | error_pct | threshold_pct | pass |")
    print("|---|---:|---:|:---:|")
    for name, err_pct, thr_pct in rows:
        if np.isfinite(thr_pct):
            status = "yes" if err_pct <= thr_pct else "no"
            thr_txt = f"{thr_pct:.4f}"
        else:
            status = "n/a"
            thr_txt = "n/a"
        print(f"| {name} | {err_pct:.4f} | {thr_txt} | {status} |")


def _base_pipeline_kwargs(
    *,
    image_path: Path,
    plot_dir: Path,
    vtk_prefix: Path,
) -> dict:
    return dict(
        image_path=image_path,
        plot_dir=plot_dir,
        vtk_output_prefix=vtk_prefix,
        verbose_logging=False,
        do_equiv_resistance_calculation=False,
        do_pericyte_constriction=False,
        skeleton_closing_radius=0,
        skeleton_bridge_gap_size=0,
        skeleton_min_branch_length=1,
        skeleton_max_bridge_distance=1,
        skeleton_component_connectivity=3,
        skeleton_min_component_percent=0.0,
        graph_reconnect_threshold=0.0,
        final_orphan_reconnect_threshold=0.0,
        min_stub_length=0.0,
        cluster_collapse_distance=0.0,
        starting_node_selection_method="coordinates",
        output_node_selection_method="coordinates",
        starting_node_coordinates=[(48.0, 48.0, 8.0)],
        output_node_coordinates=[(48.0, 48.0, 84.0)],
        starting_nodes=[],
        output_nodes=[],
        input_p_bc=INPUT_PRESSURE_PA,
        output_p_bc=OUTPUT_PRESSURE_PA,
        visualize_results=False,
        visualize_vtk=False,
        statistics_mode="fast",
    )


@pytest.mark.integration
@pytest.mark.slow
def test_resistance_calculation_on_diagonal_branching_network_from_mask(
    tmp_path: Path,
) -> None:
    pv = pytest.importorskip("pyvista")

    pipeline = _load_pipeline_module()
    mask = _build_diagonal_synthetic_mask()
    assert int(np.count_nonzero(mask)) > 0

    input_tiff = tmp_path / "diag_branching_segmentation_mask.tif"
    tifffile.imwrite(str(input_tiff), (mask.astype(np.uint8) * 255))

    output_dir = tmp_path / "pipeline_outputs_mask"
    plot_dir = tmp_path / "pipeline_plots_mask"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "diag_resistance_mask"
    flow_vtp_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels_flow.vtp")

    pipeline.image_to_model_pipeline(
        do_skeletonize=True,
        do_graph_building=True,
        run_haemodynamics=True,
        use_fwhm_edge_diameters=False,
        diameter_by_branch_order=_branch_order_diameters_from_radius(DESIGN_RADIUS_VOX),
        **_base_pipeline_kwargs(
            image_path=input_tiff,
            plot_dir=plot_dir,
            vtk_prefix=vtk_prefix,
        ),
    )
    assert flow_vtp_path.exists()

    vessels = pv.read(str(flow_vtp_path))
    diameter_each = np.full(vessels.n_cells, 2.0 * DESIGN_RADIUS_VOX, dtype=float)
    _run_hemodynamics_assertions(
        vessels,
        diameter_each=diameter_each,
        network_scalar_diameter=2.0 * DESIGN_RADIUS_VOX,
        report_title="mask_assigned_diameter",
        radius_dependent=False,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_resistance_calculation_on_diagonal_branching_network_from_raw_tiff(
    tmp_path: Path,
) -> None:
    pv = pytest.importorskip("pyvista")

    pipeline = _load_pipeline_module()
    mask = _build_diagonal_synthetic_mask()
    raw = _build_diagonal_synthetic_raw(mask)
    assert int(np.count_nonzero(mask)) > 0
    assert int(np.max(raw)) > 0

    input_tiff = tmp_path / "diag_branching_segmentation_for_raw.tif"
    raw_tiff = tmp_path / "diag_branching_raw_signal.tif"
    tifffile.imwrite(str(input_tiff), (mask.astype(np.uint8) * 255))
    tifffile.imwrite(str(raw_tiff), raw)

    output_dir = tmp_path / "pipeline_outputs_raw"
    plot_dir = tmp_path / "pipeline_plots_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    vtk_prefix = output_dir / "diag_resistance_raw"
    flow_vtp_path = vtk_prefix.with_name(vtk_prefix.name + "_vessels_flow.vtp")

    pipeline.image_to_model_pipeline(
        do_skeletonize=True,
        do_graph_building=True,
        run_haemodynamics=True,
        use_fwhm_edge_diameters=True,
        fwhm_raw_tiff_path=raw_tiff,
        fwhm_reject_samples_with_center_offset=False,
        fwhm_reject_samples_with_low_fit_r2=False,
        diameter_by_branch_order=_branch_order_diameters_from_radius(DESIGN_RADIUS_VOX),
        **_base_pipeline_kwargs(
            image_path=input_tiff,
            plot_dir=plot_dir,
            vtk_prefix=vtk_prefix,
        ),
    )
    assert flow_vtp_path.exists()

    vessels = pv.read(str(flow_vtp_path))
    assigned_diameter = np.asarray(vessels.cell_data["assigned_diameter_um"], dtype=float)
    assert len(assigned_diameter) == vessels.n_cells
    assert np.all(np.isfinite(assigned_diameter))
    assert np.all(assigned_diameter > 0.0)

    measured_radius_vox = float(np.median(assigned_diameter) / 2.0)
    assert measured_radius_vox == pytest.approx(DESIGN_RADIUS_VOX, abs=0.7)

    _run_hemodynamics_assertions(
        vessels,
        diameter_each=assigned_diameter,
        network_scalar_diameter=float(np.median(assigned_diameter)),
        report_title="raw_tiff_fwhm_assigned_diameter",
        radius_dependent=True,
    )
