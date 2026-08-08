"""Regression tests for (x, y, z) vs (z, y, x) voxel-size handling.

Image metadata reports voxel size as ``(x, y, z)`` while array axes are canonical
``(z, y, x)``. Feeding metadata order straight into code that scales array
indices swaps the z and x spacings, which is invisible for isotropic voxels and
wrong for every real microscopy stack. Every test here uses three *distinct*
spacings so the two orders cannot be confused.
"""
import numpy as np
import pytest
import tifffile

from haemolynx.graph import build_graph_from_skeleton
from haemolynx.graph.large_vessels import dilate_binary_mask_by_microns
from haemolynx.haemodynamics.automated import physical_points_to_continuous_indices
from haemolynx.io import load_3d_tif_with_voxel_size, voxel_size_zyx_from_xyz

# Deliberately anisotropic: coarse z, fine x — the usual confocal/2-photon case.
VOXEL_SIZE_XYZ = (0.4, 0.5, 2.0)
VOXEL_SIZE_ZYX = (2.0, 0.5, 0.4)


def test_voxel_size_zyx_from_xyz_puts_z_spacing_on_axis_zero():
    assert voxel_size_zyx_from_xyz(VOXEL_SIZE_XYZ) == VOXEL_SIZE_ZYX


def test_tiff_metadata_converts_to_array_axis_spacing(tmp_path):
    """The loader reports (x, y, z); axis-0 spacing must be the z spacing."""
    volume = np.zeros((6, 8, 10), dtype=np.uint8)
    volume[2, 3, 4] = 1
    path = tmp_path / "anisotropic.tif"
    tifffile.imwrite(
        str(path),
        volume,
        imagej=True,
        resolution=(1.0 / VOXEL_SIZE_XYZ[0], 1.0 / VOXEL_SIZE_XYZ[1]),
        metadata={"spacing": VOXEL_SIZE_XYZ[2], "unit": "um"},
    )

    _image, vx, vy, vz, status = load_3d_tif_with_voxel_size(str(path))
    assert status["status"] == "complete"
    assert (vx, vy, vz) == pytest.approx(VOXEL_SIZE_XYZ)

    spacing_zyx = voxel_size_zyx_from_xyz((vx, vy, vz))
    # Axis 0 is the stack axis, so it must carry the z spacing, not the x spacing.
    assert spacing_zyx[0] == pytest.approx(VOXEL_SIZE_XYZ[2])
    assert spacing_zyx[2] == pytest.approx(VOXEL_SIZE_XYZ[0])


def test_graph_edge_length_along_z_uses_z_spacing():
    """A vessel running along array axis 0 must be measured with the z spacing.

    With the metadata order passed through unconverted this edge measures
    7 * 0.4 = 2.8 um instead of 7 * 2.0 = 14.0 um.
    """
    pytest.importorskip("skan")
    skeleton = np.zeros((12, 7, 7), dtype=bool)
    skeleton[2:10, 3, 3] = True  # 8 voxels along z, 7 steps

    G = build_graph_from_skeleton(
        skeleton,
        voxel_size=VOXEL_SIZE_ZYX,
        min_stub_length=0.0,
        cluster_collapse_distance=0.0,
    )

    assert G.graph["voxel_size"] == pytest.approx(VOXEL_SIZE_ZYX)
    assert G.number_of_edges() == 1
    length = float(next(iter(G.edges(data=True)))[2]["length"])
    assert length == pytest.approx(7 * VOXEL_SIZE_XYZ[2], rel=1e-9)

    z_positions = sorted(float(data["pos"][0]) for _, data in G.nodes(data=True))
    assert z_positions == pytest.approx([2 * 2.0, 9 * 2.0])


def test_graph_edge_length_along_x_uses_x_spacing():
    """The mirror case: a vessel along array axis 2 is measured with the x spacing."""
    pytest.importorskip("skan")
    skeleton = np.zeros((7, 7, 12), dtype=bool)
    skeleton[3, 3, 2:10] = True

    G = build_graph_from_skeleton(
        skeleton,
        voxel_size=VOXEL_SIZE_ZYX,
        min_stub_length=0.0,
        cluster_collapse_distance=0.0,
    )

    assert G.number_of_edges() == 1
    length = float(next(iter(G.edges(data=True)))[2]["length"])
    assert length == pytest.approx(7 * VOXEL_SIZE_XYZ[0], rel=1e-9)


def test_mask_dilation_extends_further_along_fine_axes():
    """A 1 um dilation reaches 2 voxels in x/y (0.4-0.5 um) but not 1 in z (2 um)."""
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[3, 3, 3] = True

    dilated = dilate_binary_mask_by_microns(
        mask, dilation_microns=1.0, voxel_size_zyx=VOXEL_SIZE_ZYX
    )

    # z spacing is 2.0 um, so no neighbouring slice is within 1 um.
    assert not dilated[2, 3, 3]
    assert not dilated[4, 3, 3]
    # y spacing is 0.5 um: two voxels out is exactly 1.0 um, three is not.
    assert dilated[3, 5, 3]
    assert not dilated[3, 6, 3]
    # x spacing is 0.4 um: two voxels out is 0.8 um.
    assert dilated[3, 3, 5]


def test_physical_points_to_continuous_indices_uses_per_array_axis_spacing():
    point_phys = np.array([[4.0, 1.5, 0.8]])  # (z, y, x) microns
    indices = physical_points_to_continuous_indices(point_phys, VOXEL_SIZE_ZYX)
    assert indices[0] == pytest.approx([2.0, 3.0, 2.0])


def test_terminal_node_mask_lookup_uses_array_axis_spacing():
    """Node positions are physical (z, y, x); mask lookup must invert with zyx spacing."""
    pytest.importorskip("skan")
    import networkx as nx

    from haemolynx.graph import select_terminal_nodes_from_large_vessel_masks

    arteriole = np.zeros((8, 8, 8), dtype=bool)
    venule = np.zeros((8, 8, 8), dtype=bool)
    arteriole[1, 2, 3] = True
    venule[6, 2, 3] = True

    G = nx.MultiGraph()
    # Physical positions of voxels (1, 2, 3) and (6, 2, 3) under zyx spacing.
    G.add_node(0, pos=np.array([1 * 2.0, 2 * 0.5, 3 * 0.4]))
    G.add_node(1, pos=np.array([6 * 2.0, 2 * 0.5, 3 * 0.4]))
    G.add_edge(0, 1, voxels=[G.nodes[0]["pos"], G.nodes[1]["pos"]])

    start_nodes, output_nodes = select_terminal_nodes_from_large_vessel_masks(
        G,
        large_arteriole_mask=arteriole,
        large_venule_mask=venule,
        voxel_size_zyx=VOXEL_SIZE_ZYX,
        allow_overlap=False,
    )

    assert start_nodes == [0]
    assert output_nodes == [1]
