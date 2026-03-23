"""Load 3D images from TIFF or HDF5 and produce skeleton."""
import logging
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.util import img_as_bool
from skimage.morphology import remove_small_objects, skeletonize

from ..preprocessing.skeleton import bridge_gaps, close_binary_mask, rescale_and_skeletonize_3d

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


def resolve_image_path_with_optional_zip(image_path: str | Path) -> Path:
    """Return an existing image path, extracting from a nearby zip when needed."""
    image_path = Path(image_path)
    if image_path.exists():
        return image_path

    zip_candidates = [
        image_path.with_suffix(f"{image_path.suffix}.zip"),
        image_path.with_suffix(".zip"),
    ]
    checked: list[Path] = []
    for zip_path in zip_candidates:
        if zip_path in checked:
            continue
        checked.append(zip_path)
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            target_member = None

            # Prefer an exact filename match within the archive.
            for member in members:
                if Path(member).name == image_path.name:
                    target_member = member
                    break

            # Fallback: if archive has only one file, use it.
            if target_member is None and len(members) == 1:
                target_member = members[0]

            if target_member is None:
                raise FileNotFoundError(
                    f"Could not find '{image_path.name}' in archive '{zip_path}'. "
                    f"Archive members: {members}"
                )

            zf.extract(target_member, path=image_path.parent)
            extracted_path = image_path.parent / target_member
            if extracted_path.exists():
                logger.info("Extracted '%s' from '%s'.", target_member, zip_path)
                return extracted_path

    raise FileNotFoundError(
        f"Input image not found: {image_path}. "
        f"Checked zip candidates: {[str(p) for p in checked]}"
    )


def crop_tiff_volume_from_corners(
    input_path: str | Path,
    output_path: str | Path,
    corner_a: tuple[float, float, float],
    corner_b: tuple[float, float, float],
) -> dict:
    """Crop a 3D TIFF volume using two opposite corners and save to TIFF.

    Corners are interpreted in (z, y, x) index order and treated as inclusive.
    Corner order does not matter; bounds are normalized internally.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    volume = tifffile.imread(str(input_path))
    if volume.ndim != 3:
        volume = simplify_to_3d(np.asarray(volume))

    shape = np.asarray(volume.shape, dtype=int)
    a = np.asarray(corner_a, dtype=float)
    b = np.asarray(corner_b, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("corner_a and corner_b must each be 3D (z, y, x) coordinates.")

    lo = np.minimum(a, b).astype(int)
    hi = np.maximum(a, b).astype(int)
    lo = np.clip(lo, 0, shape - 1)
    hi = np.clip(hi, 0, shape - 1)
    if np.any(hi < lo):
        raise ValueError("Invalid crop bounds after clipping.")

    cropped = volume[
        lo[0] : hi[0] + 1,
        lo[1] : hi[1] + 1,
        lo[2] : hi[2] + 1,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), cropped)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_shape": tuple(int(v) for v in shape),
        "cropped_shape": tuple(int(v) for v in cropped.shape),
        "corner_a": tuple(int(v) for v in lo),
        "corner_b": tuple(int(v) for v in hi),
    }


def load_3d_tif(filepath: str | Path) -> np.ndarray:
    """Load a 3D TIFF volume."""
    return tifffile.imread(str(filepath))


def load_3d_h5(filepath: str | Path, dataset_name: str) -> np.ndarray:
    """Load a 3D HDF5 volume."""
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    with h5py.File(str(filepath), "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
        image = np.array(f[dataset_name][:])
    return simplify_to_3d(image)


def get_tif_spacing(filepath: str | Path) -> tuple[float, float, float]:
    """Attempt to extract (z, y, x) spacing from TIFF metadata.
    
    Returns (1.0, 1.0, 1.0) if metadata is missing or invalid.
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            # Default to isotropic
            x = y = z = 1.0
            
            # X and Y Resolution
            # resolution is usually stored as (numerator, denominator)
            page = tif.pages[0]
            if 'XResolution' in page.tags:
                val = page.tags['XResolution'].value
                x = val[1] / val[0] if isinstance(val, tuple) else 1.0 / val
            if 'YResolution' in page.tags:
                val = page.tags['YResolution'].value
                y = val[1] / val[0] if isinstance(val, tuple) else 1.0 / val
                
            # Z Spacing (often in ImageJ metadata)
            ij_meta = tif.imagej_metadata
            if ij_meta and 'spacing' in ij_meta:
                z = float(ij_meta['spacing'])
            
            return (z, y, x)
    except Exception:
        return (1.0, 1.0, 1.0)


def load_and_skeletonize_3d_tif(
    filepath: str,
    voxel_size: float = 1.0,
    closing_radius: int = 3,
    bridge_gap_size: int = 4,
    downsample_factor: float = 1.0,
):
    """Load a TIFF stack, threshold, fill holes, close gaps, and skeletonize.

    Parameters
    ----------
    filepath:
        Path to the TIFF file.
    voxel_size:
        Isotropic voxel size (unused in the skeleton but available for callers).
    closing_radius:
        Radius (in voxels) for the morphological closing step applied to the
        binary mask before skeletonization.  Closing seals concavities and
        bridges between nearby vessel blobs without permanently expanding
        boundaries.  Set to 0 to skip.
    bridge_gap_size:
        Maximum gap (in voxels) filled by the distance-transform dilation step
        after closing and hole-filling.
    downsample_factor:
        Factor to downsample by before skeletonization.
    """
    logger.debug("Loading and skeletonizing TIFF...")
    image = tifffile.imread(filepath)
    threshold = threshold_otsu(image)
    binary = image > threshold
    if closing_radius > 0:
        logger.debug("Applying morphological closing (radius=%d)…", closing_radius)
        binary = close_binary_mask(binary, radius=closing_radius)
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled, max_gap=bridge_gap_size)

    if downsample_factor > 1.0:
        skeleton = rescale_and_skeletonize_3d(bridged, downsample_factor=downsample_factor)
    else:
        skeleton = skeletonize(img_as_bool(bridged))
    return image, skeleton.astype(bool)


def load_and_skeletonize_3d_h5(
    filepath: str,
    dataset_name: str,
    voxel_size: float = 1.0,
    closing_radius: int = 3,
    bridge_gap_size: int = 4,
    downsample_factor: float = 1.0,
):
    """Load an HDF5 dataset, simplify to 3D, then skeletonize.

    Parameters
    ----------
    closing_radius:
        Radius for the morphological closing step.  Set to 0 to skip.
    bridge_gap_size:
        Maximum gap filled by the distance-transform dilation step.
    downsample_factor:
        Factor to downsample by before skeletonization.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    logger.debug("Loading and skeletonizing H5...")
    with h5py.File(filepath, "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {available}"
            )
        image = np.array(f[dataset_name][:])
    logger.debug("Original image shape: %s", image.shape)
    image = simplify_to_3d(image)
    logger.debug("Simplified image shape: %s", image.shape)
    threshold = threshold_otsu(image)
    binary = image > threshold
    if closing_radius > 0:
        logger.debug("Applying morphological closing (radius=%d)…", closing_radius)
        binary = close_binary_mask(binary, radius=closing_radius)
    filled = binary_fill_holes(binary)
    bridged = bridge_gaps(filled, max_gap=bridge_gap_size)

    if downsample_factor > 1.0:
        skeleton = rescale_and_skeletonize_3d(bridged, downsample_factor=downsample_factor)
    else:
        skeleton = skeletonize(img_as_bool(bridged))
    return image, skeleton


def simplify_to_3d(image: np.ndarray) -> np.ndarray:
    """Reduce image to 3D by taking first spatial/channel slice."""
    if image.ndim == 3:
        return image
    if image.ndim < 3:
        raise ValueError(f"Image has {image.ndim} dimensions. Need at least 3D.")
    logger.warning(
        "Image has %d dimensions. Taking first 3 spatial + first channel.",
        image.ndim,
    )
    if image.ndim == 6:
        return image[:, :, :, 0, 0, 0]
    return image[:, :, :, 0]
