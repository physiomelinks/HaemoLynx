"""3D object-to-vessel distance analysis for binary cell masks.

This module labels connected objects in a 3D binary mask (cells), computes each
object's nearest 3D distance to the edge of a vessel volume, and assigns the
nearest vessel segment identity (branch order) when a vessel graph is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

import networkx as nx
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial import cKDTree
import tifffile

from ImageLynx.io import (
    CANONICAL_AXIS_ORDER,
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    voxel_size_zyx_from_xyz,
)
from ImageLynx.haemodynamics.automated import build_graph_branch_label_volume


@dataclass(frozen=True)
class ObjectDistanceRecord:
    """Per-object nearest vessel-distance result."""

    object_id: int
    object_name: str
    voxel_count: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    edge_to_vessel_distance_microns: float
    edge_to_vessel_nearest_z: float
    edge_to_vessel_nearest_y: float
    edge_to_vessel_nearest_x: float
    edge_to_vessel_nearest_graph_edge_label_id: int | None
    edge_to_vessel_nearest_branch_order: str | None
    centroid_to_vessel_distance_microns: float
    centroid_to_vessel_nearest_z: float
    centroid_to_vessel_nearest_y: float
    centroid_to_vessel_nearest_x: float
    centroid_to_vessel_nearest_graph_edge_label_id: int | None
    centroid_to_vessel_nearest_branch_order: str | None


def _load_binary_mask_and_voxel_size(
    mask_path: str | Path,
    *,
    h5_dataset_name: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a binary mask from tif/tiff/h5 and return (mask_bool, voxel_size_xyz)."""
    path = Path(mask_path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, vx, vy, vz, _voxel_meta_status = load_3d_tif_with_voxel_size(
            str(path), axis_order=axis_order
        )
    elif suffix == ".h5":
        image, vx, vy, vz, _voxel_meta_status = load_3d_h5_with_voxel_size(
            str(path), dataset_name=h5_dataset_name, axis_order=axis_order
        )
    else:
        raise ValueError(
            f"Unsupported mask format '{path.suffix}'. Expected .tif, .tiff, or .h5."
        )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {image.shape}.")
    return image.astype(bool), (float(vx), float(vy), float(vz))


def _load_volume_shape_only(
    volume_path: str | Path,
    *,
    h5_dataset_name: str | None = None,
) -> tuple[int, int, int]:
    """Load only volume shape for tif/tiff/h5 reference image."""
    path = Path(volume_path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        vol = tifffile.imread(str(path))
    elif suffix == ".h5":
        vol, _, _, _, _ = load_3d_h5_with_voxel_size(
            str(path),
            dataset_name=h5_dataset_name,
        )
    else:
        raise ValueError(
            f"Unsupported reference format '{path.suffix}'. Expected .tif, .tiff, or .h5."
        )
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D reference volume, got shape {vol.shape}.")
    return (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))


def label_connected_objects(
    object_mask: np.ndarray,
    *,
    connectivity: int = 3,
    object_name_prefix: str = "Object",
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Label connected components in a binary 3D object mask."""
    if object_mask.ndim != 3:
        raise ValueError(f"Expected 3D object mask, got shape {object_mask.shape}.")
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be 1, 2, or 3; got {connectivity}.")
    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    labels, n_objects = label(object_mask.astype(bool), structure=structure)
    objects = [
        (i, f"{object_name_prefix}_{i:04d}") for i in range(1, int(n_objects) + 1)
    ]
    return labels.astype(np.int32), objects


def _compute_vessel_boundary_mask(vessel_mask: np.ndarray) -> np.ndarray:
    """Return vessel boundary voxels using 3D binary erosion."""
    if not np.any(vessel_mask):
        raise ValueError("Vessel mask is empty.")
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(vessel_mask.astype(bool), structure=structure)
    boundary = vessel_mask.astype(bool) & ~eroded
    if not np.any(boundary):
        # Thin vessel masks can vanish after erosion; fall back to whole vessel mask.
        boundary = vessel_mask.astype(bool)
    return boundary


def _compute_object_boundary_indices(object_region: np.ndarray) -> np.ndarray:
    """Return indices of boundary voxels for one connected object."""
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = binary_erosion(object_region.astype(bool), structure=structure)
    boundary = object_region.astype(bool) & ~eroded
    idx = np.argwhere(boundary)
    if idx.size == 0:
        idx = np.argwhere(object_region.astype(bool))
    return idx


def _edge_label_to_branch_order_map(
    G: nx.MultiGraph,
) -> dict[int, str | None]:
    """Map graph edge label ids to branch-order strings."""
    mapping: dict[int, str | None] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_label = data.get("graph_edge_label_id")
        if edge_label is None:
            continue
        try:
            edge_label_i = int(edge_label)
        except (TypeError, ValueError):
            continue
        branch_order = data.get("branch_order")
        mapping[edge_label_i] = None if branch_order is None else str(branch_order)
    return mapping


def _build_edge_voxel_lookup(
    G: nx.MultiGraph,
    *,
    volume_shape: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, dict[int, str | None]]:
    """Rasterize graph edges to a label volume and return branch-order lookup."""
    edge_label_volume, _ = build_graph_branch_label_volume(
        G,
        volume_shape=volume_shape,
        voxel_size_zyx=voxel_size_zyx,
        background_label=0,
        junction_label=-1,
    )
    return edge_label_volume, _edge_label_to_branch_order_map(G)


def compute_object_to_vessel_distances(
    *,
    object_mask: np.ndarray,
    vessel_mask: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    graph_edge_label_volume: np.ndarray | None = None,
    edge_label_to_branch_order: dict[int, str | None] | None = None,
    connectivity: int = 3,
    object_name_prefix: str = "Object",
) -> list[ObjectDistanceRecord]:
    """Compute two per-object nearest vessel-edge distances and branch identities.

    Measurements per object:
    1) From the closest object-edge voxel to nearest vessel-edge voxel.
    2) From object centroid to nearest vessel-edge voxel.
    """
    if object_mask.shape != vessel_mask.shape:
        raise ValueError(
            "object_mask and vessel_mask must share shape. "
            f"Got {object_mask.shape} and {vessel_mask.shape}."
        )
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(
            f"voxel_size_zyx must be 3 positive values, got {voxel_size_zyx}."
        )

    labels, objects = label_connected_objects(
        object_mask, connectivity=connectivity, object_name_prefix=object_name_prefix
    )
    if not objects:
        return []

    boundary_mask = _compute_vessel_boundary_mask(vessel_mask)
    boundary_idx = np.argwhere(boundary_mask)
    if boundary_idx.size == 0:
        raise ValueError("No vessel boundary voxels found.")
    boundary_phys = boundary_idx.astype(float) * spacing.reshape(1, 3)
    boundary_tree = cKDTree(boundary_phys)

    edge_tree: cKDTree | None = None
    edge_labels: np.ndarray | None = None
    if graph_edge_label_volume is not None:
        if graph_edge_label_volume.shape != object_mask.shape:
            raise ValueError(
                "graph_edge_label_volume shape must match object/vessel masks. "
                f"Got {graph_edge_label_volume.shape} and {object_mask.shape}."
            )
        valid_edge_idx = np.argwhere(graph_edge_label_volume > 0)
        if valid_edge_idx.size > 0:
            valid_edge_phys = valid_edge_idx.astype(float) * spacing.reshape(1, 3)
            edge_tree = cKDTree(valid_edge_phys)
            edge_labels = graph_edge_label_volume[
                valid_edge_idx[:, 0], valid_edge_idx[:, 1], valid_edge_idx[:, 2]
            ].astype(int)

    results: list[ObjectDistanceRecord] = []
    def _resolve_branch_identity_from_vessel_point(
        vessel_point_phys: np.ndarray,
    ) -> tuple[int | None, str | None]:
        nearest_edge_label_id: int | None = None
        nearest_branch_order: str | None = None
        if edge_tree is not None and edge_labels is not None and edge_labels.size > 0:
            _, edge_nn = edge_tree.query(vessel_point_phys.reshape(1, 3), k=1)
            nearest_edge_label_id = int(edge_labels[int(edge_nn[0])])
            if edge_label_to_branch_order is not None:
                nearest_branch_order = edge_label_to_branch_order.get(
                    nearest_edge_label_id
                )
        return nearest_edge_label_id, nearest_branch_order

    for object_id, object_name in objects:
        object_idx = np.argwhere(labels == int(object_id))
        if object_idx.size == 0:
            continue
        centroid_idx = np.mean(object_idx.astype(float), axis=0)
        centroid_phys = centroid_idx * spacing

        object_region = labels == int(object_id)
        object_boundary_idx = _compute_object_boundary_indices(object_region)
        object_boundary_phys = object_boundary_idx.astype(float) * spacing.reshape(1, 3)
        edge_dists, edge_nearest_indices = boundary_tree.query(object_boundary_phys, k=1)
        best_object_boundary_i = int(np.argmin(edge_dists))
        best_vessel_boundary_i_from_edge = int(edge_nearest_indices[best_object_boundary_i])
        edge_to_vessel_dist = float(edge_dists[best_object_boundary_i])
        nearest_vessel_idx_from_edge = boundary_idx[best_vessel_boundary_i_from_edge].astype(float)
        nearest_vessel_phys_from_edge = nearest_vessel_idx_from_edge * spacing
        (
            edge_nearest_edge_label_id,
            edge_nearest_branch_order,
        ) = _resolve_branch_identity_from_vessel_point(nearest_vessel_phys_from_edge)

        centroid_dist_arr, centroid_nearest_idx = boundary_tree.query(
            centroid_phys.reshape(1, 3), k=1
        )
        centroid_to_vessel_dist = float(centroid_dist_arr[0])
        nearest_vessel_idx_from_centroid = boundary_idx[int(centroid_nearest_idx[0])].astype(float)
        nearest_vessel_phys_from_centroid = nearest_vessel_idx_from_centroid * spacing
        (
            centroid_nearest_edge_label_id,
            centroid_nearest_branch_order,
        ) = _resolve_branch_identity_from_vessel_point(nearest_vessel_phys_from_centroid)

        results.append(
            ObjectDistanceRecord(
                object_id=int(object_id),
                object_name=object_name,
                voxel_count=int(object_idx.shape[0]),
                centroid_z=float(centroid_idx[0]),
                centroid_y=float(centroid_idx[1]),
                centroid_x=float(centroid_idx[2]),
                edge_to_vessel_distance_microns=edge_to_vessel_dist,
                edge_to_vessel_nearest_z=float(nearest_vessel_phys_from_edge[0]),
                edge_to_vessel_nearest_y=float(nearest_vessel_phys_from_edge[1]),
                edge_to_vessel_nearest_x=float(nearest_vessel_phys_from_edge[2]),
                edge_to_vessel_nearest_graph_edge_label_id=edge_nearest_edge_label_id,
                edge_to_vessel_nearest_branch_order=edge_nearest_branch_order,
                centroid_to_vessel_distance_microns=centroid_to_vessel_dist,
                centroid_to_vessel_nearest_z=float(nearest_vessel_phys_from_centroid[0]),
                centroid_to_vessel_nearest_y=float(nearest_vessel_phys_from_centroid[1]),
                centroid_to_vessel_nearest_x=float(nearest_vessel_phys_from_centroid[2]),
                centroid_to_vessel_nearest_graph_edge_label_id=centroid_nearest_edge_label_id,
                centroid_to_vessel_nearest_branch_order=centroid_nearest_branch_order,
            )
        )
    return results


def export_object_distance_details_csv(
    records: list[ObjectDistanceRecord],
    output_csv_path: str | Path,
) -> Path:
    """Write per-object dual-distance rows to CSV."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Object ID",
                "Object Name",
                "Voxel Count",
                "Centroid Z (voxel)",
                "Centroid Y (voxel)",
                "Centroid X (voxel)",
                "Edge-to-Vessel Distance (microns)",
                "Edge-to-Vessel Nearest Z (microns)",
                "Edge-to-Vessel Nearest Y (microns)",
                "Edge-to-Vessel Nearest X (microns)",
                "Edge-to-Vessel Nearest Coordinate (z,y,x microns)",
                "Edge-to-Vessel Nearest Graph Edge Label ID",
                "Edge-to-Vessel Nearest Branch Order",
                "Centroid-to-Vessel Distance (microns)",
                "Delta Distance (Centroid-Edge) (microns)",
                "Centroid-to-Vessel Nearest Z (microns)",
                "Centroid-to-Vessel Nearest Y (microns)",
                "Centroid-to-Vessel Nearest X (microns)",
                "Centroid-to-Vessel Nearest Coordinate (z,y,x microns)",
                "Centroid-to-Vessel Nearest Graph Edge Label ID",
                "Centroid-to-Vessel Nearest Branch Order",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.object_id,
                    rec.object_name,
                    rec.voxel_count,
                    f"{rec.centroid_z:.6g}",
                    f"{rec.centroid_y:.6g}",
                    f"{rec.centroid_x:.6g}",
                    f"{rec.edge_to_vessel_distance_microns:.6g}",
                    f"{rec.edge_to_vessel_nearest_z:.6g}",
                    f"{rec.edge_to_vessel_nearest_y:.6g}",
                    f"{rec.edge_to_vessel_nearest_x:.6g}",
                    f"({rec.edge_to_vessel_nearest_z:.6g}, {rec.edge_to_vessel_nearest_y:.6g}, {rec.edge_to_vessel_nearest_x:.6g})",
                    ""
                    if rec.edge_to_vessel_nearest_graph_edge_label_id is None
                    else rec.edge_to_vessel_nearest_graph_edge_label_id,
                    ""
                    if rec.edge_to_vessel_nearest_branch_order is None
                    else rec.edge_to_vessel_nearest_branch_order,
                    f"{rec.centroid_to_vessel_distance_microns:.6g}",
                    f"{(rec.centroid_to_vessel_distance_microns - rec.edge_to_vessel_distance_microns):.6g}",
                    f"{rec.centroid_to_vessel_nearest_z:.6g}",
                    f"{rec.centroid_to_vessel_nearest_y:.6g}",
                    f"{rec.centroid_to_vessel_nearest_x:.6g}",
                    f"({rec.centroid_to_vessel_nearest_z:.6g}, {rec.centroid_to_vessel_nearest_y:.6g}, {rec.centroid_to_vessel_nearest_x:.6g})",
                    ""
                    if rec.centroid_to_vessel_nearest_graph_edge_label_id is None
                    else rec.centroid_to_vessel_nearest_graph_edge_label_id,
                    ""
                    if rec.centroid_to_vessel_nearest_branch_order is None
                    else rec.centroid_to_vessel_nearest_branch_order,
                ]
            )
    return out


def export_object_distance_summary_csv(
    records: list[ObjectDistanceRecord],
    output_csv_path: str | Path,
) -> Path:
    """Write overall and per-branch dual-distance summary to CSV."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    edge_distances = np.asarray(
        [float(r.edge_to_vessel_distance_microns) for r in records], dtype=float
    )
    centroid_distances = np.asarray(
        [float(r.centroid_to_vessel_distance_microns) for r in records], dtype=float
    )
    grouped_edge: dict[str, list[float]] = {}
    grouped_centroid: dict[str, list[float]] = {}
    grouped_delta_by_edge_group: dict[str, list[float]] = {}
    grouped_delta_by_centroid_group: dict[str, list[float]] = {}
    for rec in records:
        group_edge = rec.edge_to_vessel_nearest_branch_order or "Unassigned"
        group_centroid = rec.centroid_to_vessel_nearest_branch_order or "Unassigned"
        delta = (
            float(rec.centroid_to_vessel_distance_microns)
            - float(rec.edge_to_vessel_distance_microns)
        )
        grouped_edge.setdefault(group_edge, []).append(
            float(rec.edge_to_vessel_distance_microns)
        )
        grouped_centroid.setdefault(group_centroid, []).append(
            float(rec.centroid_to_vessel_distance_microns)
        )
        grouped_delta_by_edge_group.setdefault(group_edge, []).append(delta)
        grouped_delta_by_centroid_group.setdefault(group_centroid, []).append(delta)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "Measurement Type",
                "Object Count",
                "Mean Distance (microns)",
                "Median Distance (microns)",
                "Std Distance (microns)",
                "Min Distance (microns)",
                "Max Distance (microns)",
                "Mean Delta (Centroid-Edge) (microns)",
                "Median Delta (Centroid-Edge) (microns)",
                "Std Delta (Centroid-Edge) (microns)",
            ]
        )
        if edge_distances.size == 0:
            writer.writerow(
                ["Overall", "edge_to_vessel", 0, "", "", "", "", "", "", "", ""]
            )
            writer.writerow(
                ["Overall", "centroid_to_vessel", 0, "", "", "", "", "", "", "", ""]
            )
            return out
        deltas = centroid_distances - edge_distances
        for metric_name, distances in (
            ("edge_to_vessel", edge_distances),
            ("centroid_to_vessel", centroid_distances),
        ):
            writer.writerow(
                [
                    "Overall",
                    metric_name,
                    int(distances.size),
                    f"{float(np.mean(distances)):.6g}",
                    f"{float(np.median(distances)):.6g}",
                    f"{float(np.std(distances)):.6g}",
                    f"{float(np.min(distances)):.6g}",
                    f"{float(np.max(distances)):.6g}",
                    f"{float(np.mean(deltas)):.6g}",
                    f"{float(np.median(deltas)):.6g}",
                    f"{float(np.std(deltas)):.6g}",
                ]
            )
        for metric_name, grouped in (
            ("edge_to_vessel", grouped_edge),
            ("centroid_to_vessel", grouped_centroid),
        ):
            for group in sorted(grouped):
                arr = np.asarray(grouped[group], dtype=float)
                if metric_name == "edge_to_vessel":
                    delta_group = np.asarray(
                        grouped_delta_by_edge_group.get(group, []), dtype=float
                    )
                else:
                    delta_group = np.asarray(
                        grouped_delta_by_centroid_group.get(group, []), dtype=float
                    )
                writer.writerow(
                    [
                        group,
                        metric_name,
                        int(arr.size),
                        f"{float(np.mean(arr)):.6g}",
                        f"{float(np.median(arr)):.6g}",
                        f"{float(np.std(arr)):.6g}",
                        f"{float(np.min(arr)):.6g}",
                        f"{float(np.max(arr)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.mean(delta_group)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.median(delta_group)):.6g}",
                        ""
                        if delta_group.size == 0
                        else f"{float(np.std(delta_group)):.6g}",
                    ]
                )
    return out


def run_3d_measurement_to_cell_mask(
    *,
    graph: nx.MultiGraph,
    cell_mask_path: str | Path,
    output_dir: str | Path,
    image_stem: str,
    voxel_size_xyz: tuple[float, float, float],
    vessel_mask_path: str | Path | None = None,
    vessel_reference_image_path: str | Path | None = None,
    cell_mask_h5_dataset_name: str | None = None,
    vessel_mask_h5_dataset_name: str | None = None,
    vessel_reference_h5_dataset_name: str | None = None,
    connectivity: int = 3,
    axis_order: str = CANONICAL_AXIS_ORDER,
) -> dict[str, Any]:
    """End-to-end 3D cell-mask distance analysis with CSV export.

    If ``vessel_mask_path`` is not provided, vessel volume is generated from graph
    edge voxels using the same rasterization logic as automated FWHM measurement.
    In that case, ``vessel_reference_image_path`` can be supplied to define the
    label volume shape; otherwise the cell-mask shape is used.

    *voxel_size_xyz* is the physical ``(x, y, z)`` voxel size reported by image
    metadata; it is compared against the mask metadata in the same order and
    converted to canonical array order ``(z, y, x)`` before any index scaling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_mask, cell_voxel_size = _load_binary_mask_and_voxel_size(
        cell_mask_path,
        h5_dataset_name=cell_mask_h5_dataset_name,
        axis_order=axis_order,
    )
    target_voxel_size_xyz = tuple(float(v) for v in voxel_size_xyz)
    target_voxel_size = voxel_size_zyx_from_xyz(target_voxel_size_xyz)
    if not np.allclose(cell_voxel_size, target_voxel_size_xyz, rtol=0.0, atol=0.0):
        raise ValueError(
            "Cell-mask voxel size does not match pipeline voxel size. "
            f"cell={cell_voxel_size}, pipeline={target_voxel_size_xyz}"
        )

    if vessel_mask_path is not None:
        vessel_mask, vessel_voxel_size = _load_binary_mask_and_voxel_size(
            vessel_mask_path,
            h5_dataset_name=vessel_mask_h5_dataset_name,
            axis_order=axis_order,
        )
        if vessel_mask.shape != cell_mask.shape:
            raise ValueError(
                "cell and vessel masks must share shape for distance analysis. "
                f"Got cell={cell_mask.shape}, vessel={vessel_mask.shape}"
            )
        if not np.allclose(vessel_voxel_size, target_voxel_size_xyz, rtol=0.0, atol=0.0):
            raise ValueError(
                "Vessel-mask voxel size does not match pipeline voxel size. "
                f"vessel={vessel_voxel_size}, pipeline={target_voxel_size_xyz}"
            )
        edge_label_volume, edge_to_bo = _build_edge_voxel_lookup(
            graph,
            volume_shape=vessel_mask.shape,
            voxel_size_zyx=target_voxel_size,
        )
    else:
        if vessel_reference_image_path is not None:
            volume_shape = _load_volume_shape_only(
                vessel_reference_image_path,
                h5_dataset_name=vessel_reference_h5_dataset_name,
            )
            if volume_shape != cell_mask.shape:
                raise ValueError(
                    "Cell-mask shape and reference-image shape must match when "
                    "building vessel volume from graph labels. "
                    f"Got cell={cell_mask.shape}, reference={volume_shape}"
                )
        else:
            volume_shape = (
                int(cell_mask.shape[0]),
                int(cell_mask.shape[1]),
                int(cell_mask.shape[2]),
            )
        edge_label_volume, edge_to_bo = _build_edge_voxel_lookup(
            graph,
            volume_shape=volume_shape,
            voxel_size_zyx=target_voxel_size,
        )
        vessel_mask = edge_label_volume > 0

    records = compute_object_to_vessel_distances(
        object_mask=cell_mask,
        vessel_mask=vessel_mask,
        voxel_size_zyx=target_voxel_size,
        graph_edge_label_volume=edge_label_volume,
        edge_label_to_branch_order=edge_to_bo,
        connectivity=connectivity,
    )

    details_csv = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances.csv"
    summary_csv = output_dir / f"{image_stem}_cell_to_vessel_3d_dual_distances_summary.csv"
    export_object_distance_details_csv(records, details_csv)
    export_object_distance_summary_csv(records, summary_csv)

    return {
        "object_count": len(records),
        "details_csv_path": details_csv,
        "summary_csv_path": summary_csv,
    }
