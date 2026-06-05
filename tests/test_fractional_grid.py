import pytest
import numpy as np
from ImageLynx.graph.tiling import generate_evenly_distributed_bounding_boxes, calculate_evenly_distributed_grid

def test_strict_volume_coverage():
    """Verify that the collective grid exactly equals the loaded field dimensions."""
    shape = (117, 234, 489)
    fraction = 0.2
    
    mask = np.zeros(shape, dtype=bool)
    
    for bbox in generate_evenly_distributed_bounding_boxes(shape, fraction, margin=0):
        z1, z2, y1, y2, x1, x2 = bbox["core"]
        mask[z1:z2, y1:y2, x1:x2] = True
        
    assert np.all(mask), "Not all voxels were covered by the generated core bounding boxes."

def test_uniform_chunk_sizing_variance():
    """Verify chunks differ by no more than 1 voxel."""
    shape = (100, 100, 100)
    fraction = 0.3
    
    z_sizes = []
    for bbox in generate_evenly_distributed_bounding_boxes(shape, fraction, margin=0):
        if bbox["core"][2] == 0 and bbox["core"][4] == 0: # Only look at first row to avoid duplicates
            z_sizes.append(bbox["core"][1] - bbox["core"][0])
            
    assert max(z_sizes) - min(z_sizes) <= 1, f"Chunk sizes vary by more than 1 voxel: {z_sizes}"
    # 100 / 33.33 = 3 chunks. step = 33.33
    # 0 to 33 (33), 33 to 67 (34), 67 to 100 (33)
    assert set(z_sizes) == {33, 34}

def test_extreme_anisotropy_near_cubic_resolution():
    """Verify extreme anisotropy."""
    shape = (10, 15, 1000)
    fraction = 0.1
    
    # max is 1000. Target S = 100.
    # N_z = max(1, round(10/100)) = 1
    # N_y = max(1, round(15/100)) = 1
    # N_x = max(1, round(1000/100)) = 10
    
    N_z, N_y, N_x, sz, sy, sx = calculate_evenly_distributed_grid(shape, fraction)
    assert (N_z, N_y, N_x) == (1, 1, 10)
    
    bboxes = list(generate_evenly_distributed_bounding_boxes(shape, fraction, margin=0))
    assert len(bboxes) == 10
    
    # First chunk shape
    z1, z2, y1, y2, x1, x2 = bboxes[0]["core"]
    assert (z2-z1, y2-y1, x2-x1) == (10, 15, 100)

def test_margin_truncation_at_absolute_boundaries():
    """Verify margin truncation."""
    shape = (100, 100, 100)
    fraction = 0.5
    margin = 20
    
    bboxes = list(generate_evenly_distributed_bounding_boxes(shape, fraction, margin=margin))
    
    first_pad = bboxes[0]["padded"]
    assert first_pad[0] == 0 and first_pad[2] == 0 and first_pad[4] == 0
    
    last_pad = bboxes[-1]["padded"]
    assert last_pad[1] == 100 and last_pad[3] == 100 and last_pad[5] == 100
