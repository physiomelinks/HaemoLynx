"""Unit tests for efficiency optimisations: ROI, Downsampling, Padded Slicing, Pruning."""
import numpy as np
from skan import csr
from ImageLynx import preprocessing, graph

def test_crop_roi():
    """Test ROI cropping logic and shape calculations."""
    # 100x100x100 ramp: each voxel holds its own index packed as z*10000 + y*100 + x, so a
    # crop's first and last values say where it was cut from.
    z, y, x = np.indices((100, 100, 100))
    image = z * 10000 + y * 100 + x

    def zyx(value):
        return [int(value) // 10000, int(value) // 100 % 100, int(value) % 100]
    
    # 1. 50% crop, no offset (should be centre 50x50x50)
    cropped = preprocessing.crop_roi(image, sub_volume_percentage=0.5)
    assert cropped.shape == (50, 50, 50)
    
    # 2. Offset by 0.25 in Z moves the crop's first z index by 25 voxels
    centred = preprocessing.crop_roi(image, sub_volume_percentage=0.2)
    cropped_offset = preprocessing.crop_roi(image, sub_volume_percentage=0.2, offset_z=0.25)
    assert cropped_offset.shape == (20, 20, 20)
    assert zyx(centred[0, 0, 0]) == [40, 40, 40]
    assert zyx(cropped_offset[0, 0, 0]) == [65, 40, 40]
    
    # 3. Boundary handling: an offset past the edge clamps the box against the last x face
    cropped_edge = preprocessing.crop_roi(image, sub_volume_percentage=0.2, offset_x=0.6)
    assert cropped_edge.shape == (20, 20, 20)
    assert zyx(cropped_edge[0, 0, 0]) == [40, 40, 80]
    assert zyx(cropped_edge[-1, -1, -1]) == [59, 59, 99]

def test_keep_largest_mask_components():
    """Test early mask component pruning."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    # Component 1 (Large: 3x3x3 = 27 voxels)
    mask[2:5, 2:5, 2:5] = True
    # Component 2 (Small: 1 voxel)
    mask[10, 10, 10] = True
    # Component 3 (Medium: 2 voxels)
    mask[15, 15, 15:17] = True
    
    # Keep only the largest (Component 1)
    pruned = preprocessing.keep_largest_mask_components(mask, n_components=1)
    assert pruned.sum() == 27
    assert pruned[2, 2, 2] == True
    assert pruned[10, 10, 10] == False
    
    # Keep top 2
    pruned_2 = preprocessing.keep_largest_mask_components(mask, n_components=2)
    assert pruned_2.sum() == 29 # 27 + 2

def test_padded_slicing_equivalence():
    """Verify that local padded slicing finds the same cycles as global detection."""
    # Create a synthetic volume with a loop (a square ring)
    # Component 1: 10x10 ring
    vol = np.zeros((30, 30, 30), dtype=bool)
    # Z=5 plane
    vol[5, 5:15, 5] = True
    vol[5, 5:15, 15] = True
    vol[5, 5, 5:15] = True
    vol[5, 15, 5:16] = True
    
    sk = csr.Skeleton(vol)
    
    # Run global (legacy)
    G_global, loops_global, _ = graph.build_graph_segment_skan_stitched_loops(
        sk, vol, use_padded_slicing=False
    )
    
    # Run padded (optimised)
    G_local, loops_padded, _ = graph.build_graph_segment_skan_stitched_loops(
        sk, vol, use_padded_slicing=True, padding=3
    )
    
    assert len(loops_global) == len(loops_padded)
    assert G_global.number_of_nodes() == G_local.number_of_nodes()
    # Both find the one ring, with the same 40 voxels (equal counts alone pass at zero).
    ring = set(zip(*np.nonzero(vol)))
    assert len(ring) == 40
    assert len(loops_global) == 1
    assert set(map(tuple, loops_global[0])) == ring
    assert set(map(tuple, loops_padded[0])) == ring

def test_rescale_and_skeletonize_3d():
    """Test downsampled skeletonization logic."""
    # Create a 100x100x100 volume with a hollow square frame (rods)
    vol = np.zeros((100, 100, 100), dtype=bool)
    # Rods of 10x10 thickness
    vol[20:80, 20:30, 20:30] = True
    vol[20:80, 70:80, 20:30] = True
    vol[20:30, 20:80, 20:30] = True
    vol[70:80, 20:80, 20:30] = True
    
    # Downsampled
    skel_down = preprocessing.rescale_and_skeletonize_3d(vol, downsample_factor=2.0)
    
    # 1. Verify it's not empty
    assert skel_down.any()
    
    # 2. Verify it's thinned (should be rods of 1 voxel thick)
    # Length of 4 rods ~ 240 voxels.
    assert 100 < skel_down.sum() < 500
