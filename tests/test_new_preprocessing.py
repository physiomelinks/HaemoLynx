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

def test_joint_hysteresis_rejects_two_class_entropy():
    """A 2-class entropy map carries no evidence independent of p, so the joint criteria must
    refuse it rather than silently producing a mask that is non-monotonic in p."""
    prob = np.zeros((4, 4, 4), dtype=np.float32)
    ent = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="at least 3 classes"):
        joint_hysteresis_threshold(prob, ent, n_classes=2)
    with pytest.raises(ValueError, match="at least 3 classes"):
        joint_hysteresis_threshold(prob, ent, n_classes=1)

def test_joint_hysteresis_accepts_three_or_more_classes():
    """The joint path stays fully available at >=3 classes, so it re-engages by itself once
    the classifier is retrained with a TH/glomus class."""
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    ent = np.ones((10, 10, 10), dtype=np.float32) * 0.99
    prob[5, 5, 5], ent[5, 5, 5] = 0.8, 0.5   # core seed
    prob[5, 5, 6], ent[5, 5, 6] = 0.8, 0.9   # attached uncertain vessel

    unchecked = joint_hysteresis_threshold(prob, ent)
    three_class = joint_hysteresis_threshold(prob, ent, n_classes=3)

    assert np.array_equal(unchecked, three_class)
    assert three_class[5, 5, 5] and three_class[5, 5, 6]

def test_two_class_entropy_criterion_is_non_monotonic_in_probability():
    """Characterisation test for the premise behind the class-count gate.

    This one passes with or without the gate - it does not guard the gate, it guards the
    mathematical fact the gate is justified by. If the entropy computation or its
    normalisation ever changes such that H is no longer folded about p = 0.5, this fails and
    the gate should be revisited.

    Built from a genuine 2-class softmax volume and the real entropy map, the candidate
    criterion of the joint threshold keeps a low-probability voxel while discarding one of
    markedly *higher* vessel probability. A mask that is non-monotonic in its own evidence
    cannot support morphometry, because every vessel becomes a high-confidence core plus a
    detached shell with the wall voxels evacuated.
    """
    p_values = np.array([0.30, 0.50, 0.75, 0.95], dtype=np.float32)
    volume = np.empty((2, 4, 4, 4), dtype=np.float32)   # (C, Z, Y, X), varying along X
    volume[0] = p_values
    volume[1] = 1.0 - p_values

    entropy = calculate_entropy_map(volume)
    assert entropy.shape == (4, 4, 4)
    ent_line = entropy[0, 0, :]

    # H is folded about p = 0.5, which is where it peaks.
    assert ent_line[1] == pytest.approx(1.0, abs=1e-6)

    candidate = (p_values >= 0.25) & (ent_line <= 0.95)
    assert candidate[0], "p = 0.30 should be retained by the candidate criterion"
    assert not candidate[1], "p = 0.50 should be discarded - this is the non-monotonicity"
    assert candidate[2] and candidate[3], "high-confidence voxels should be retained"

def test_entropy_map_not_computed_for_two_class_probability_field(tmp_path):
    """The pipeline-level gate, which is what actually protects the run.

    Leaving entropy_map as None routes _apply_preprocessing_filters down its existing plain
    hysteresis branch. There is a single origin for the entropy map, so this covers the
    monolithic and the map-reduce paths alike.
    """
    import sys
    from pathlib import Path
    import tifffile

    examples_path = Path(__file__).parent.parent / "examples"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import (
        _load_raw_probability_field, PreprocessingConfig, SkeletonConfig,
    )

    def write_probability_field(path, n_classes):
        vol = np.zeros((6, n_classes, 8, 8), dtype=np.float32)   # ZCYX
        vol[:, 0] = 0.7
        vol[:, 1:] = 0.3 / (n_classes - 1)
        tifffile.imwrite(str(path), vol)
        return path

    pre_config = PreprocessingConfig()
    assert pre_config.enable_shannon_entropy, "fixture assumes entropy is requested"
    skel_config = SkeletonConfig()
    skel_config.sub_volume_percentage = 1.0

    two_class = write_probability_field(tmp_path / "two_class.tif", 2)
    _, entropy_two = _load_raw_probability_field(str(two_class), "tif", pre_config, skel_config)
    assert entropy_two is None, "a 2-class field must not produce an entropy map"

    three_class = write_probability_field(tmp_path / "three_class.tif", 3)
    _, entropy_three = _load_raw_probability_field(str(three_class), "tif", pre_config, skel_config)
    assert entropy_three is not None, "a 3-class field must still produce an entropy map"
    assert entropy_three.shape == (6, 8, 8)

def test_evaluate_preprocessing_uncertainty():
    from ImageLynx.statistics.benchmarking import evaluate_preprocessing_uncertainty
    
    # 10x10x10 arrays
    ent = np.zeros((10, 10, 10), dtype=np.float32)
    binary = np.zeros((10, 10, 10), dtype=bool)
    
    # Empty mask should return 0
    res = evaluate_preprocessing_uncertainty(ent, binary)
    assert res["mean_uncertainty"] == 0.0
    assert res["high_uncertainty_fraction"] == 0.0
    
    # 10 voxels kept
    binary[0:10, 0, 0] = True
    
    # 5 voxels are highly uncertain (0.9), 5 are very certain (0.1)
    ent[0:5, 0, 0] = 0.9
    ent[5:10, 0, 0] = 0.1
    
    res2 = evaluate_preprocessing_uncertainty(ent, binary)
    assert res2["mean_uncertainty"] == pytest.approx(0.5)
    # 5 out of 10 voxels have entropy > 0.8
    assert res2["high_uncertainty_fraction"] == pytest.approx(0.5)
    
    # Let's test the threshold exact boundary
    ent[0:5, 0, 0] = 0.79  # Less than 0.8
    res3 = evaluate_preprocessing_uncertainty(ent, binary)
    assert res3["high_uncertainty_fraction"] == 0.0
