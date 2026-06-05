import numpy as np
from typing import Callable
from joblib import Parallel, delayed

from ImageLynx.graph.tiling import generate_evenly_distributed_bounding_boxes

def map_reduce_pipeline(
    volume: np.ndarray,
    chunk_fraction: float,
    margin: int,
    worker_fn: Callable[[np.ndarray, dict], np.ndarray],
    n_jobs: int = -1
) -> np.ndarray:
    """
    Executes a tiling and stitching Map-Reduce pipeline on a 3D volume,
    returning a stitched binary mask.
    
    Args:
        volume: The global 3D array (lazy dask array or memory-mapped zarr).
        chunk_fraction: Fraction to divide the array by.
        margin: Overlap margin in voxels.
        worker_fn: A function that takes (chunk_array, bbox_dict) and returns a local_core_binary mask.
        n_jobs: Number of parallel workers (for joblib).
        
    Returns:
        A continuous stitched global numpy array.
    """
    Z, Y, X = volume.shape
    
    bboxes = list(generate_evenly_distributed_bounding_boxes((Z, Y, X), chunk_fraction, margin))
    
    def process_chunk(item):
        idx, bbox = item
        pz1, pz2, py1, py2, px1, px2 = bbox['padded']
        chunk = volume[pz1:pz2, py1:py2, px1:px2]
        
        # If the chunk is entirely empty background, skip processing
        if np.max(chunk) == 0:
            return None, bbox
            
        # Execute local pipeline (returns local_core_binary without margin)
        local_core_binary = worker_fn(chunk, bbox, idx, len(bboxes))
        
        return local_core_binary, bbox

    print(f"Executing {len(bboxes)} chunks across {n_jobs if n_jobs > 0 else 'all'} workers...")
    
    # Phase 2: Parallel Local Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(item) for item in enumerate(bboxes, 1)
    )
    
    # Phase 3: Binary Stitching
    print("Stitching local binary masks...")
    stitched_binary_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    
    for local_core_binary, bbox in results:
        if local_core_binary is not None:
            cz1, cz2, cy1, cy2, cx1, cx2 = bbox['core']
            stitched_binary_mask[cz1:cz2, cy1:cy2, cx1:cx2] = local_core_binary
            
    return stitched_binary_mask
