import math
import numpy as np

def calculate_evenly_distributed_grid(shape: tuple[int, int, int], chunk_fraction: float):
    Z, Y, X = shape
    
    if chunk_fraction >= 1.0 or chunk_fraction <= 0.0:
        return 1, 1, 1, float(Z), float(Y), float(X)
        
    # 1. Target Isotropic Size
    S = max(Z, Y, X) * chunk_fraction
    
    # 2. Subdivisions per axis
    N_z = max(1, round(Z / S))
    N_y = max(1, round(Y / S))
    N_x = max(1, round(X / S))
    
    # 3. Floating point step sizes
    step_z = Z / N_z
    step_y = Y / N_y
    step_x = X / N_x
    
    return N_z, N_y, N_x, step_z, step_y, step_x

def generate_evenly_distributed_bounding_boxes(shape: tuple[int, int, int], chunk_fraction: float, margin: int = 0):
    Z, Y, X = shape
    N_z, N_y, N_x, step_z, step_y, step_x = calculate_evenly_distributed_grid(shape, chunk_fraction)
    
    for i_z in range(N_z):
        for i_y in range(N_y):
            for i_x in range(N_x):
                core_z_start = int(round(i_z * step_z))
                core_z_end = int(round((i_z + 1) * step_z))
                
                core_y_start = int(round(i_y * step_y))
                core_y_end = int(round((i_y + 1) * step_y))
                
                core_x_start = int(round(i_x * step_x))
                core_x_end = int(round((i_x + 1) * step_x))
                
                pad_z_start = max(0, core_z_start - margin)
                pad_z_end = min(Z, core_z_end + margin)
                
                pad_y_start = max(0, core_y_start - margin)
                pad_y_end = min(Y, core_y_end + margin)
                
                pad_x_start = max(0, core_x_start - margin)
                pad_x_end = min(X, core_x_end + margin)
                
                yield {
                    "core": (core_z_start, core_z_end, core_y_start, core_y_end, core_x_start, core_x_end),
                    "padded": (pad_z_start, pad_z_end, pad_y_start, pad_y_end, pad_x_start, pad_x_end),
                    "offset": (pad_z_start, pad_y_start, pad_x_start)
                }
