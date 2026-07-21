"""Tests for new preprocessing functions."""
import pytest
import numpy as np
from ImageLynx.preprocessing.image import (
    smooth_probability_map, 
    hysteresis_threshold,
    joint_hysteresis_threshold,
    median_filter_image,
    morphological_opening,
    calculate_entropy_map,
    crop_roi
)
from ImageLynx.preprocessing.skeleton import fill_holes_3d

def test_smooth_probability_map():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[5, 5, 5] = 1.0
    smoothed = smooth_probability_map(image, sigma=1.0)
    assert smoothed.shape == image.shape
    assert smoothed.max() < 1.0  # Max should be reduced by smoothing
    assert smoothed.sum() == pytest.approx(image.sum(), rel=1e-5)

def test_hysteresis_threshold():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[5, 5, 5] = 0.8  # Seed
    image[5, 5, 6] = 0.4  # Connected above low
    image[5, 5, 7] = 0.2  # Below low
    
    # Hysteresis (low=0.3, high=0.7)
    binary = hysteresis_threshold(image, low=0.3, high=0.7)
    assert binary[5, 5, 5] == True
    assert binary[5, 5, 6] == True
    assert binary[5, 5, 7] == False
    
    # No seed case
    image2 = np.ones((10, 10, 10)) * 0.4
    binary2 = hysteresis_threshold(image2, low=0.3, high=0.7)
    assert not binary2.any()

def test_fill_holes_3d():
    # Hollow cube
    binary = np.ones((10, 10, 10), dtype=bool)
    binary[1:9, 1:9, 1:9] = False
    
    # Fill the hole
    filled = fill_holes_3d(binary)
    assert filled.all()

def test_median_filter_image():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[5, 5, 5] = 1.0  # Salt noise
    filtered = median_filter_image(image, size=3)
    # Median of 3x3x3 around (5,5,5) should be 0.0 because only one pixel is 1.0
    assert filtered[5, 5, 5] == 0.0
    assert filtered.shape == image.shape

def test_morphological_opening_binary():
    binary = np.zeros((10, 10, 10), dtype=bool)
    binary[5, 5, 5] = True  # Single voxel noise
    binary[2:5, 2:5, 2:5] = True  # Larger object (3x3x3)
    
    # Opening with radius 1 (ball) should remove single voxel
    opened = morphological_opening(binary, radius=1)
    assert opened[5, 5, 5] == False
    # Center of 3x3x3 block should remain (ball radius 1 fits in 3x3x3)
    assert opened[3, 3, 3] == True
    assert opened.shape == binary.shape

def test_morphological_opening_grayscale():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    image[5, 5, 5] = 1.0
    image[2:5, 2:5, 2:5] = 0.8
    
    opened = morphological_opening(image, radius=1)
    # Grayscale opening with ball radius 1 should remove isolated peak of size 1
    assert opened[5, 5, 5] == 0.0
    # Center of 0.8 block should remain
    assert opened[3, 3, 3] == pytest.approx(0.8)

def test_calculate_entropy_map():
    # 4 channels: C, Z, Y, X where C is the smallest dimension (4)
    # Uniform probabilities = max entropy (1.0)
    image_uniform = np.ones((4, 10, 10, 10), dtype=np.float32) * 0.25
    entropy_uniform = calculate_entropy_map(image_uniform)
    assert entropy_uniform.shape == (10, 10, 10)
    assert np.allclose(entropy_uniform, 1.0)

    # Absolute certainty = min entropy (0.0)
    image_certain = np.zeros((10, 10, 10, 4), dtype=np.float32)
    image_certain[..., 0] = 1.0  # Class 0 is 100% certain
    entropy_certain = calculate_entropy_map(image_certain)
    assert entropy_certain.shape == (10, 10, 10)
    assert np.allclose(entropy_certain, 0.0)

def test_crop_roi():
    image = np.ones((10, 20, 20), dtype=np.float32)
    # Crop to 50% volume
    cropped = crop_roi(image, sub_volume_percentage=0.5)
    assert cropped.shape == (5, 10, 10)

    # Crop with offset
    cropped_offset = crop_roi(image, sub_volume_percentage=0.5, offset_z=0.25, offset_y=0.25, offset_x=0.25)
    assert cropped_offset.shape == (5, 10, 10)

    # Crop 4D array (preserving channel dimension which is smallest)
    image_4d = np.ones((2, 10, 20, 20), dtype=np.float32)
    cropped_4d = crop_roi(image_4d, sub_volume_percentage=0.5)
    assert cropped_4d.shape == (2, 5, 10, 10)


def test_joint_hysteresis_logic():
    # 10x10x10 arrays
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    ent = np.ones((10, 10, 10), dtype=np.float32) * 0.99  # default high entropy
    
    # Core seed: High prob (0.8), Low entropy (0.5)
    prob[5, 5, 5] = 0.8
    ent[5, 5, 5] = 0.5
    
    # Attached uncertain vessel: High prob (0.8), High entropy (0.9)
    prob[5, 5, 6] = 0.8
    ent[5, 5, 6] = 0.9
    
    # Isolated uncertain vessel (noise): High prob (0.8), High entropy (0.9)
    prob[2, 2, 2] = 0.8
    ent[2, 2, 2] = 0.9
    
    # Connected certain background: Low prob (0.1), Low entropy (0.5)
    prob[5, 5, 4] = 0.1
    ent[5, 5, 4] = 0.5
    
    binary = joint_hysteresis_threshold(
        prob, ent, 
        low=0.2, high=0.4, 
        shannon_core=0.6, shannon_max=0.95
    )
    
    # Test Case 1: Spatial Connectivity Preservation
    # The core seed and the attached uncertain vessel should both be True
    assert binary[5, 5, 5] == True
    assert binary[5, 5, 6] == True
    
    # Test Case 2: Isolated Noise Rejection
    # The isolated uncertain vessel should be False (not connected to a seed)
    assert binary[2, 2, 2] == False
    
    # Test Case 3: Certain Background Nullification
    # Low probability region should be False even if connected to seed
    assert binary[5, 5, 4] == False

def test_joint_hysteresis_validation():
    prob = np.zeros((2, 2, 2))
    ent = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        joint_hysteresis_threshold(prob, ent, low=0.8, high=0.4)
    with pytest.raises(ValueError):
        joint_hysteresis_threshold(prob, ent, shannon_core=0.9, shannon_max=0.5)
