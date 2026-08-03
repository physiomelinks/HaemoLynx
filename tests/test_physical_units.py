"""Physical-unit vs voxel-index handling across the graph and benchmark layers.

Node ``pos`` and edge ``voxels`` are stored in physical units by
``build_graph_segment_skan_stitched_loops``, while ``image_shape``, array subscripts and
several thresholds are in voxels. At a voxel size of (1, 1, 1) the two are numerically
identical, so no fixture built at unit spacing can tell them apart - which is exactly why
this class of bug survived. Every test here therefore runs at a genuinely anisotropic
spacing, where a physical coordinate used as a subscript lands somewhere visibly wrong.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.graph.boundaries import select_boundary_terminal_nodes
from ImageLynx.statistics.benchmarking import (
    evaluate_completeness_and_overpruning,
    redilate_skeleton_to_volume,
)

# The measured acquisition voxel size for the WKY carotid body volume, (z, y, x) in microns.
SPACING = (1.8660, 1.8660, 1.8639)
SHAPE = (20, 20, 20)

# A straight centreline along z at voxel (y, x) = (10, 10), spanning voxels z = 5..15.
CENTRELINE_VOXELS = [(z, 10, 10) for z in range(5, 16)]


def _physical(voxel_coords):
    """Convert (z, y, x) voxel indices to the physical coordinates the graph actually stores."""
    return [
        [z * SPACING[0], y * SPACING[1], x * SPACING[2]] for z, y, x in voxel_coords
    ]


def _centreline_graph(diameter_um=2.0):
    G = nx.MultiGraph()
    physical = _physical(CENTRELINE_VOXELS)
    G.add_node(0, pos=np.array(physical[0]))
    G.add_node(1, pos=np.array(physical[-1]))
    G.add_edge(0, 1, voxels=physical, assigned_diameter_um=diameter_um)
    return G


def test_redilate_skeleton_to_volume_round_trips_anisotropic_spacing():
    """Physical edge coordinates must be divided by the spacing before being used as indices.

    Truncating them directly places the reconstruction at voxel (z * 1.866, 18, 18) instead
    of (z, 10, 10) - an offset that silently corrupts the Dice coefficient the skeletonisation
    tuner optimises against.
    """
    reconstructed = redilate_skeleton_to_volume(_centreline_graph(), SHAPE, SPACING)

    assert reconstructed[10, 10, 10], \
        "the true centreline voxel is not covered by the reconstruction"
    assert not reconstructed[10, 18, 18], \
        "reconstruction landed at int(physical), i.e. the coordinates were used as indices"

    # The reconstruction should sit on the centreline in y/x, not be smeared to the corner.
    occupied = np.argwhere(reconstructed)
    assert occupied.size > 0
    assert abs(occupied[:, 1].mean() - 10) < 1.5
    assert abs(occupied[:, 2].mean() - 10) < 1.5


def test_evaluate_completeness_and_overpruning_round_trips_anisotropic_spacing():
    """Orphaned volume is measured as a distance from the drawn skeleton to mask tissue.

    If the physical coordinates are used as subscripts the skeleton is drawn in the wrong
    place, every tissue voxel looks far from a centreline, and a healthy graph is reported
    as massively over-pruned.
    """
    mask = np.zeros(SHAPE, dtype=bool)
    for z, y, x in CENTRELINE_VOXELS:
        mask[z, y - 1:y + 2, x - 1:x + 2] = True

    result = evaluate_completeness_and_overpruning(
        _centreline_graph(), mask, SPACING, max_capillary_radius_um=5.0
    )

    # Every tissue voxel is within ~1 voxel (1.87 um) of the true centreline, so nothing is
    # orphaned. Mis-indexing puts the skeleton ~8 voxels (15 um) away and orphans everything.
    assert result["orphaned_volume_fraction"] < 0.05, (
        "tissue appears orphaned because the skeleton was drawn at the wrong coordinates: "
        f"{result['orphaned_volume_fraction']:.3f}"
    )


def test_select_boundary_terminal_nodes_compares_like_with_like():
    """Node positions are physical; image_shape is in voxels. The extent must be converted.

    Comparing a physical coordinate against a voxel extent shrinks the apparent volume by the
    voxel size, so interior nodes drift into the outlet band and are wired up as venous
    boundaries.
    """
    G = nx.Graph()
    # A hub at voxel z = 10 with three degree-1 terminals at voxels z = 0, 8 and 19.
    positions = {"hub": 10, "inlet": 0, "interior": 8, "outlet": 19}
    for name, z_voxel in positions.items():
        G.add_node(name, pos=np.array([z_voxel * SPACING[0], 10 * SPACING[1], 10 * SPACING[2]]))
    for terminal in ("inlet", "interior", "outlet"):
        G.add_edge("hub", terminal)

    starting, outputs = select_boundary_terminal_nodes(
        G, SHAPE, edge_percent=25.0, end_percent=25.0, axis=0, voxel_size=SPACING,
    )

    assert "inlet" in starting
    assert "outlet" in outputs
    # z = 8 of 19 is 42% of the way along, comfortably outside the bottom 25%. Against a
    # voxel-denominated extent it lands at 79% and is misclassified as an outlet.
    assert "interior" not in outputs, \
        "an interior node was selected as an outlet; the extent was not converted to physical units"
    assert "interior" not in starting


def test_resolve_voxel_size_prefers_explicit_configuration():
    """An explicitly configured voxel size must win over file metadata.

    The probability TIFFs declare no resolution, so metadata detection yields (1, 1, 1) and
    silently reports voxel counts as microns. Configuration has to override it.
    """
    import sys
    from pathlib import Path

    examples_path = Path(__file__).parent.parent / "examples"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    import carotid_image_to_model as C

    original = C.VOXEL_SIZE_UM
    try:
        C.VOXEL_SIZE_UM = SPACING
        # Deliberately a format with no metadata path at all: configuration must still win.
        assert C._resolve_voxel_size("no/such/file.npy", "npy") == SPACING

        C.VOXEL_SIZE_UM = None
        assert C._resolve_voxel_size("no/such/file.npy", "npy") == (1.0, 1.0, 1.0)
    finally:
        C.VOXEL_SIZE_UM = original


def test_unit_spacing_is_unchanged():
    """The fixes must be a no-op at (1, 1, 1), which is what every existing caller uses today."""
    unit = (1.0, 1.0, 1.0)
    G = nx.MultiGraph()
    voxels = [[float(z), 10.0, 10.0] for z in range(5, 16)]
    G.add_node(0, pos=np.array(voxels[0]))
    G.add_node(1, pos=np.array(voxels[-1]))
    G.add_edge(0, 1, voxels=voxels, assigned_diameter_um=2.0)

    reconstructed = redilate_skeleton_to_volume(G, SHAPE, unit)
    assert reconstructed[10, 10, 10]
    assert not reconstructed[10, 18, 18]
