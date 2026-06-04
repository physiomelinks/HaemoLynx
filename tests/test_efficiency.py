"""Unit tests for efficiency optimizations: ROI, Downsampling, Padded Slicing, Pruning."""
import numpy as np
import pytest
import networkx as nx
from skan import csr
from ImageLynx import preprocessing, graph

def test_crop_roi():
    """Test ROI cropping logic and shape calculations."""
    # 100x100x100 volume
    image = np.zeros((100, 100, 100))
    
    # 1. 50% crop, no offset (should be center 50x50x50)
    cropped = preprocessing.crop_roi(image, sub_volume_percentage=0.5)
    assert cropped.shape == (50, 50, 50)
    
    # 2. Test offsets
    # Offset by 0.25 in Z (should shift the crop start)
    cropped_offset = preprocessing.crop_roi(image, sub_volume_percentage=0.2, offset_z=0.25)
    assert cropped_offset.shape == (20, 20, 20)
    
    # 3. Boundary handling (offset too large)
    cropped_edge = preprocessing.crop_roi(image, sub_volume_percentage=0.2, offset_x=0.6)
    assert cropped_edge.shape == (20, 20, 20)

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
    
    # Run padded (optimized)
    G_local, loops_padded, _ = graph.build_graph_segment_skan_stitched_loops(
        sk, vol, use_padded_slicing=True, padding=3
    )
    
    assert len(loops_global) == len(loops_padded)
    assert G_global.number_of_nodes() == G_local.number_of_nodes()

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
