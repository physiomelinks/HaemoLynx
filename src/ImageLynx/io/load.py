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


def load_3d_tif(filepath: str | Path, lazy: bool = False) -> np.ndarray:
    """Load a 3D TIFF volume.
    
    If lazy=True, returns a memory-mapped array or dask array if possible.
    """
    if lazy:
        try:
            import dask.array as da
            # Using dask.array.from_array with tifffile.memmap for lazy loading
            mmap = tifffile.memmap(str(filepath))
            return da.from_array(mmap, chunks="auto")
        except Exception as e:
            logger.warning("Lazy TIFF loading failed, falling back to in-memory: %s", e)
    
    return tifffile.imread(str(filepath))


def load_3d_tif_with_voxel_size(filepath, voxel_size=1.0, lazy: bool = False):
    return load_3d_tif(filepath, lazy=lazy), voxel_size, voxel_size, voxel_size


def load_3d_h5(filepath: str | Path, dataset_name: str, lazy: bool = False) -> np.ndarray:
    """Load a 3D HDF5 volume.
    
    If lazy=True, returns a dask array wrapper around the H5 dataset.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
    
    if lazy:
        try:
            import dask.array as da
            # We don't close the file because dask needs it open
            f = h5py.File(str(filepath), "r")
            if dataset_name not in f:
                available = list(f.keys())
                raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
            ds = f[dataset_name]
            return da.from_array(ds, chunks="auto")
        except Exception as e:
            logger.warning("Lazy H5 loading failed, falling back to in-memory: %s", e)

    with h5py.File(str(filepath), "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
        image = np.array(f[dataset_name][:])
    return simplify_to_3d(image)


def load_3d_h5_with_voxel_size(filepath, dataset_name, voxel_size=1.0):
    return load_3d_h5(filepath, dataset_name), voxel_size, voxel_size, voxel_size


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
    closing_radius: int = 0,
    bridge_gap_size: int = 0,
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
        
    # Attempt to get voxel size for the 5-value return signature expected by some pipelines
    z, y, x = get_tif_spacing(filepath)
    return image, skeleton.astype(bool), x, y, z


def load_and_skeletonize_3d_h5(
    filepath: str,
    dataset_name: str | None = None,
    voxel_size: float = 1.0,
    closing_radius: int = 0,
    bridge_gap_size: int = 0,
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
        available = list(f.keys())
        if dataset_name is None:
            path_stem = Path(filepath).stem
            candidates = [path_stem, "data", "image", "volume"]
            for candidate in candidates:
                if candidate in f:
                    dataset_name = candidate
                    break
            if dataset_name is None:
                if len(available) == 1:
                    dataset_name = available[0]
                else:
                    raise ValueError(f"Could not auto-detect dataset. Available: {available}")
                    
        if dataset_name not in f:
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
        
    # Default to isotropic 1.0 for H5 as we don't have a robust extractor here yet
    return image, skeleton.astype(bool), 1.0, 1.0, 1.0


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


def read_ilastik_probabilities(
    path,
    vessel_class_index: int | None = None,
    dataset: str | None = None,
    expected_shape_zyx: tuple | None = None,
    check_calibration: bool = True,
) -> np.ndarray:
    """Read one class channel from an Ilastik headless probability export.

    The export is 4D with one probability channel per class, in label order. Selecting the
    wrong channel is the one failure in this chain that yields a complete and plausible set
    of results rather than an error: the background probability is a perfectly well-formed
    volume, and everything downstream is then computed from the inverse segmentation. So the
    class index is never defaulted, and the returned channel is checked against the calibre
    of thing a vessel probability can be - a few percent of the volume, not most of it.

    ``vessel_class_index`` defaults to ``specimens.VESSEL_CLASS_INDEX``, which is unset until
    the trained project's label order is recorded, and raises while it is.

    ``expected_shape_zyx`` catches a probability map paired with the wrong specimen.

    Returns the selected channel as float32 in (z, y, x).
    """
    from ..specimens import PROBABILITIES_DATASET, resolve_vessel_class_index

    if h5py is None:
        raise ImportError("h5py is required to read Ilastik exports. pip install h5py")

    index = resolve_vessel_class_index(vessel_class_index)
    dataset = PROBABILITIES_DATASET if dataset is None else dataset
    path = Path(path)

    with h5py.File(path, "r") as handle:
        available = list(handle.keys())
        if dataset not in handle:
            raise KeyError(
                f"{path.name} has no dataset {dataset!r}. Available: {available}. Ilastik "
                f"writes {PROBABILITIES_DATASET!r} by default; --output_internal_path sets it."
            )
        volume = np.squeeze(np.asarray(handle[dataset]))

    if volume.ndim != 4:
        raise ValueError(
            f"{path.name}/{dataset} has shape {volume.shape}, which carries no class axis "
            f"after squeezing. A 3D export has already collapsed the classes, so which one "
            f"survived cannot be recovered - re-export with every class channel."
        )

    # The class axis is whichever is short; the other three are spatial. Ilastik writes it
    # last, but the raw acquisitions in this study are ZCYX, so position is not reliable.
    class_axis = int(np.argmin(volume.shape))
    n_classes = volume.shape[class_axis]
    if n_classes > 8:
        raise ValueError(
            f"{path.name}/{dataset} has shape {volume.shape} and no axis short enough to be "
            f"a class axis; the smallest is {n_classes}."
        )
    if index >= n_classes:
        raise ValueError(
            f"vessel class index {index} is out of range for {n_classes} classes in "
            f"{path.name}."
        )

    probabilities = np.ascontiguousarray(
        np.take(volume, index, axis=class_axis)
    ).astype(np.float32)

    if expected_shape_zyx is not None and probabilities.shape != tuple(expected_shape_zyx):
        raise ValueError(
            f"{path.name} has shape {probabilities.shape}, expected "
            f"{tuple(expected_shape_zyx)}. This is a probability map paired with the wrong "
            f"specimen, not a reshaping problem."
        )

    lo, hi = float(probabilities.min()), float(probabilities.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"{path.name} class {index} spans [{lo:.4f}, {hi:.4f}], which is not a "
            f"probability. Export with --export_dtype=float32 rather than 8-bit."
        )

    # Deliberately the whole-volume mean, not the mean of any sub-volume. A swapped class
    # is a property of the file, and the empty margin is what gives the test its headroom:
    # measured across the six CB specimens the whole-volume mean runs 0.179-0.353, while
    # inside the 160^3 ROI those same channels run 0.538-0.611 - above this limit in every
    # one of them, because roi_placement centres the box on the densest tissue in the stack.
    # Applying this check to a placed sub-volume would reject every correct channel.
    mean = float(probabilities.mean())
    if check_calibration and mean > 0.5:
        raise ValueError(
            f"{path.name} class {index} has mean probability {mean:.3f}. Over a whole stack, "
            f"most of which is empty margin, a vessel channel averages well below 0.5 "
            f"(0.18-0.35 on the CB specimens); a mean above 0.5 means this is the background "
            f"class, and using it would compute every downstream result from the inverse "
            f"segmentation without erroring anywhere. Check the trained project's label "
            f"order, or pass check_calibration=False if this really is intended."
        )
    logger.info("Read %s class %d: mean probability %.4f", path.name, index, mean)
    return probabilities
