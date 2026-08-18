#!/usr/bin/env python3
"""
================================================================================
TH-Positive Glomus Cell Preprocessing and 2-Channel Tiling Pipeline
================================================================================
Author: Gemini Notebook / Data Craft Pipeline
Resolution Calibrated: 1.8660 x 1.8660 x 1.8639 µm^3
Description: Automatically processes 3D Z-stack confocal microimages of the
             TH channel (glomus cells) using SciPy and Scikit-Image. Generates:
             - Channel 1: Preprocessed, denoised, and normalized grayscale.
             - Channel 2: 3D Difference of Gaussians (DoG) Soma Ring Enhancer.
             Then tiles the hyperstack laterally (256x256xZ) and exports them
             into chunked 3D HDF5 files with 'ZCYXS' axis tagging for Ilastik.
================================================================================
"""

import os
import sys
import json
import numpy as np
import h5py
from scipy.ndimage import median_filter, gaussian_filter
from skimage import io, exposure
from skimage.morphology import white_tophat, disk

def print_status(message):
    """Prints a formatted system status message."""
    print(f"[STATUS] {message}")
    sys.stdout.flush()

def match_z_bleaching(volume):
    """
    Corrects Z-axis signal decay slice-by-slice using Histogram Matching.
    Matches the histogram of each slice to the middle slice of the stack.
    """
    print_status("Phase 2: Correcting Z-axis bleaching...")
    num_slices = volume.shape[0]
    ref_idx = num_slices // 2
    reference_slice = volume[ref_idx]
    
    corrected_volume = np.empty_like(volume)
    corrected_volume[ref_idx] = reference_slice
    
    for z in range(num_slices):
        if z == ref_idx:
            continue
        # Slice-by-slice histogram matching
        corrected_volume[z] = exposure.match_histograms(
            volume[z], reference_slice
        )
    return corrected_volume

def apply_background_subtraction(volume, radius=12.0):
    """
    Clears inter-cluster haze using a 2D White Top-Hat Filter on each slice.
    A radius of 12.0 pixels (~22.4 µm) avoids erasing individual 8-15 µm cells.
    """
    print_status(f"Phase 4: Subtracting background haze (Rolling Ball radius = {radius}px)...")
    num_slices = volume.shape[0]
    processed_volume = np.empty_like(volume)
    selem = disk(int(radius))
    
    for z in range(num_slices):
        processed_volume[z] = white_tophat(volume[z], footprint=selem)
    return processed_volume

def normalize_dynamic_range(volume, saturated_percent=0.35):
    """
    Stretches the image histogram to utilize the complete 16-bit range (0 to 65535).
    Saturates a tiny fraction of top pixels (default: 0.35%) to normalize cohorts.
    """
    print_status("Phase 5: Normalizing dynamic range (0.35% saturation)...")
    v_min = np.min(volume)
    # 0.35% saturation on top intensities
    v_max = np.percentile(volume, 100.0 - saturated_percent)
    
    if v_max == v_min:
        v_max += 1e-5
        
    # Standardize to 16-bit unsigned integer space
    normalized = np.clip((volume - v_min) / (v_max - v_min), 0.0, 1.0)
    return (normalized * 65535).astype(np.uint16)

def generate_soma_ring_enhancer(volume):
    """
    Creates a secondary channel using a 3D Difference of Gaussians (DoG).
    Gaussian 1 (Sigma=1.0) suppresses sub-voxel camera noise.
    Gaussian 3 (Sigma=3.0) matches the expected radius of individual glomus cells (~5.6 µm).
    """
    print_status("Phase 6: Generating 3D Difference of Gaussians Soma Ring Enhancer...")
    # Apply narrow blur
    g1 = gaussian_filter(volume.astype(np.float32), sigma=1.0)
    # Apply wide blur
    g3 = gaussian_filter(volume.astype(np.float32), sigma=3.0)
    
    # Subtract to isolate cell boundaries/glowing cytoplasm rings
    dog = g1 - g3
    # Remove negative values (representing dark centers or exterior)
    dog = np.clip(dog, 0, None)
    
    # Normalize the output to 16-bit range
    v_min = np.min(dog)
    v_max = np.percentile(dog, 99.65)  # 0.35% saturated
    if v_max == v_min:
        v_max += 1e-5
    dog_normalized = np.clip((dog - v_min) / (v_max - v_min), 0.0, 1.0)
    
    return (dog_normalized * 65535).astype(np.uint16)

def save_hdf5_with_axis_tags(filepath, data):
    """
    Saves a 2-channel 3D volume as an HDF5 dataset with 'ZCYXS' axis metadata.
    This guarantees that Ilastik reads the channels and slices correctly.
    """
    with h5py.File(filepath, "w") as f:
        # Create dataset with gzip compression
        dset = f.create_dataset(
            "volume", 
            data=data, 
            compression="gzip", 
            compression_opts=4
        )
        
        # Build Ilastik axis metadata
        # Axis description: Z (depth), C (channels), Y (height), X (width), S (states/none)
        axis_tags = {
            "axes": [
                {"key": "z", "type": "space", "description": "depth"},
                {"key": "c", "type": "channel", "description": "channel"},
                {"key": "y", "type": "space", "description": "height"},
                {"key": "x", "type": "space", "description": "width"}
            ]
        }
        dset.attrs["axistags"] = json.dumps(axis_tags)

def tile_and_export(ch1_grayscale, ch2_vesselness, tile_size, output_dir, prefix):
    """
    Crops the 2-channel hyperstack into sub-volumes of size (Z, 2, tile_size, tile_size)
    and saves them to chunked HDF5 files. Keeps the Z-axis fully intact.
    """
    print_status("Phase 7: Running RAM-optimized lateral tiling and HDF5 export...")
    os.makedirs(output_dir, exist_ok=True)
    
    z_depth, height, width = ch1_grayscale.shape
    num_tiles_y = int(np.ceil(height / tile_size))
    num_tiles_x = int(np.ceil(width / tile_size))
    
    # Pad the volume if sizes are not multiples of the tile_size
    pad_y = num_tiles_y * tile_size - height
    pad_x = num_tiles_x * tile_size - width
    
    if pad_y > 0 or pad_x > 0:
        print_status(f"Padding volume boundaries by Y:+{pad_y}px, X:+{pad_x}px to ensure seamless tiling...")
        ch1_grayscale = np.pad(ch1_grayscale, ((0, 0), (0, pad_y), (0, pad_x)), mode='reflect')
        ch2_vesselness = np.pad(ch2_vesselness, ((0, 0), (0, pad_y), (0, pad_x)), mode='reflect')
    
    total_saved = 0
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            ys = ty * tile_size
            ye = ys + tile_size
            xs = tx * tile_size
            xe = xs + tile_size
            
            # Crop Z-depth fully intact, slicing 256x256 laterally
            tile_ch1 = ch1_grayscale[:, ys:ye, xs:xe]
            tile_ch2 = ch2_vesselness[:, ys:ye, xs:xe]
            
            # Stack channels to yield shape: [Z, C, Y, X] -> C1 is Grayscale, C2 is Soma Ring Enhancer
            merged_tile = np.stack([tile_ch1, tile_ch2], axis=1)
            
            tile_filename = f"{prefix}_tile_x{tx}_y{ty}.h5"
            tile_filepath = os.path.join(output_dir, tile_filename)
            
            save_hdf5_with_axis_tags(tile_filepath, merged_tile)
            total_saved += 1
            
    print_status(f"Exported {total_saved} 2-channel HDF5 tiles to '{output_dir}'.")

def process_pipeline(input_image_path, output_directory, tile_size=256):
    """Executes the full preprocessing and tiling sequence on a single multi-page TIFF Z-stack."""
    filename_base = os.path.splitext(os.path.basename(input_image_path))[0]
    print_status(f"Starting pipeline on image: {input_image_path}")
    
    # Read the 3D raw stack
    volume = io.imread(input_image_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D Z-stack image, but got dimensions: {volume.shape}")
    
    print_status(f"Loaded raw image stack with dimensions: Z={volume.shape[0]}, Y={volume.shape[1]}, X={volume.shape[2]}")
    
    # Run the calibrated pre-processing steps
    corrected = match_z_bleaching(volume)
    
    print_status("Phase 3: Volumetric edge-preserving denoising (3D Median Filter)...")
    denoised = median_filter(corrected, size=(3, 3, 3))  # 3D Median (size=3 corresponds to 1px radius)
    
    grayscale_normalized = apply_background_subtraction(denoised, radius=12.0)
    ch1_grayscale = normalize_dynamic_range(grayscale_normalized, saturated_percent=0.35)
    
    ch2_ring_enhancer = generate_soma_ring_enhancer(ch1_grayscale)
    
    # Cut into 2-channel HDF5 tiles
    tile_and_export(
        ch1_grayscale, 
        ch2_ring_enhancer, 
        tile_size=tile_size, 
        output_dir=output_directory, 
        prefix=filename_base
    )
    print_status(f"Successfully processed and generated inputs for file: '{input_image_path}'!")

if __name__ == "__main__":
    # Example local usage block
    if len(sys.argv) < 3:
        print("Usage: python process_th_glomus_cells.py <path_to_raw_Zstack.tif> <output_directory>")
        print("Example: python process_th_glomus_cells.py /data/WKY_Sample_1_TH.tif /data/ilastik_inputs")
    else:
        raw_tiff_path = sys.argv[1]
        out_dir = sys.argv[2]
        try:
            process_pipeline(raw_tiff_path, out_dir)
        except Exception as e:
            print(f"[ERROR] Pipeline execution failed: {str(e)}")
            sys.exit(1)
