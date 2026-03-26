"""Coordinate conversion helpers for ImageLynx.

Physical coordinates are handled as (x, y, z).
NumPy image indices are handled as (z, y, x) to match array layout.
"""
from __future__ import annotations

import numpy as np


def physical_xyz_to_index_zyx(position_xyz, voxel_size_xyz) -> np.ndarray:
    """Convert physical coordinate (x,y,z) to nearest array index (z,y,x)."""
    x, y, z = np.asarray(position_xyz, dtype=float)
    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    return np.rint([z / vz, y / vy, x / vx]).astype(int)


def physical_xyz_to_continuous_index_zyx(points_xyz, voxel_size_xyz) -> np.ndarray:
    """Convert physical (x,y,z) points to continuous array indices (z,y,x)."""
    pts = np.asarray(points_xyz, dtype=float)
    one_point = pts.ndim == 1
    if one_point:
        pts = pts.reshape(1, 3)
    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    idx = np.empty_like(pts, dtype=float)
    idx[:, 0] = pts[:, 2] / vz
    idx[:, 1] = pts[:, 1] / vy
    idx[:, 2] = pts[:, 0] / vx
    return idx[0] if one_point else idx


def physical_xyz_delta_to_index_zyx_delta(delta_xyz, voxel_size_xyz) -> np.ndarray:
    """Convert physical displacement (x,y,z) to index displacement (z,y,x)."""
    dx, dy, dz = np.asarray(delta_xyz, dtype=float)
    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    return np.asarray([dz / vz, dy / vy, dx / vx], dtype=float)


def index_zyx_to_physical_xyz(index_zyx, voxel_size_xyz) -> np.ndarray:
    """Convert array index (z,y,x) to physical coordinate (x,y,z)."""
    z_idx, y_idx, x_idx = np.asarray(index_zyx, dtype=float)
    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    return np.asarray([x_idx * vx, y_idx * vy, z_idx * vz], dtype=float)


def indices_zyx_to_physical_xyz(indices_zyx, voxel_size_xyz) -> np.ndarray:
    """Convert (N,3) array indices (z,y,x) to physical coordinates (x,y,z)."""
    arr = np.asarray(indices_zyx, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    vx, vy, vz = np.asarray(voxel_size_xyz, dtype=float)
    out = np.empty_like(arr, dtype=float)
    out[:, 0] = arr[:, 2] * vx
    out[:, 1] = arr[:, 1] * vy
    out[:, 2] = arr[:, 0] * vz
    return out
