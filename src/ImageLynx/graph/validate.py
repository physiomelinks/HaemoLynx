"""Validate skeleton connections between positions."""
from typing import List, Tuple, Optional

import numpy as np

from ._helpers import get_line_points_3d


def validate_skeleton_connection(
    skeleton_data: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    max_gap: float = 3.0,
) -> Tuple[bool, Optional[List]]:
    """
    Validate that there's a skeleton path between two positions.
    Returns (is_valid, voxel_path or None).
    """
    try:
        p1 = np.round(pos1).astype(int)
        p2 = np.round(pos2).astype(int)
        if not (
            0 <= p1[0] < skeleton_data.shape[0]
            and 0 <= p1[1] < skeleton_data.shape[1]
            and 0 <= p1[2] < skeleton_data.shape[2]
        ) or not (
            0 <= p2[0] < skeleton_data.shape[0]
            and 0 <= p2[1] < skeleton_data.shape[1]
            and 0 <= p2[2] < skeleton_data.shape[2]
        ):
            return False, None

        line_points = get_line_points_3d(p1, p2)
        skeleton_nearby = 0
        for point in line_points:
            region = skeleton_data[
                max(0, point[0] - 1) : min(skeleton_data.shape[0], point[0] + 2),
                max(0, point[1] - 1) : min(skeleton_data.shape[1], point[1] + 2),
                max(0, point[2] - 1) : min(skeleton_data.shape[2], point[2] + 2),
            ]
            if np.any(region > 0):
                skeleton_nearby += 1
        connection_ratio = skeleton_nearby / len(line_points)
        is_valid = connection_ratio > 0.7
        if is_valid:
            return True, line_points
        return False, None
    except Exception:
        return False, None
