"""Load 3D images from TIFF or HDF5 and produce skeleton."""
import logging
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from skimage.util import img_as_bool
from skimage.morphology import  skeletonize_3d
from scipy.ndimage import binary_fill_holes
try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


def _coerce_triplet(value) -> tuple[float, float, float] | None:
    """Convert HDF5 attr values to a 3-float tuple when possible."""
    if value is None:
        return None
    arr = np.asarray(value).astype(float).ravel()
    if arr.size == 1:
        v = float(arr[0])
        return (v, v, v)
    if arr.size >= 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    return None


def _extract_h5_voxel_size(dataset, h5_file) -> tuple[float, float, float]:
    """Extract (x, y, z) voxel size from common HDF5 attribute conventions."""
    attrs = {}
    for source in (h5_file.attrs, dataset.attrs):
        for key in source.keys():
            attrs[str(key).lower()] = source[key]

    # Most common in microscopy exports: element_size_um is usually stored as (z, y, x).
    if "element_size_um" in attrs:
        zyx = _coerce_triplet(attrs["element_size_um"])
        if zyx is not None:
            z, y, x = zyx
            return (x, y, z)

    # Common generic triplet keys; assume they are already ordered (x, y, z).
    for key in ("voxel_size", "voxel_size_um", "pixelsize", "pixel_size", "resolution"):
        if key in attrs:
            xyz = _coerce_triplet(attrs[key])
            if xyz is not None:
                return xyz

    # Axis-specific metadata keys.
    x = y = z = None
    x_keys = ("voxel_size_x", "x_voxel_size", "spacing_x", "x_spacing", "resolution_x")
    y_keys = ("voxel_size_y", "y_voxel_size", "spacing_y", "y_spacing", "resolution_y")
    z_keys = ("voxel_size_z", "z_voxel_size", "spacing_z", "z_spacing", "resolution_z", "spacing")
    for key in x_keys:
        if key in attrs:
            x = float(np.asarray(attrs[key]).astype(float).ravel()[0])
            break
    for key in y_keys:
        if key in attrs:
            y = float(np.asarray(attrs[key]).astype(float).ravel()[0])
            break
    for key in z_keys:
        if key in attrs:
            z = float(np.asarray(attrs[key]).astype(float).ravel()[0])
            break

    if x is not None and y is not None and z is not None:
        return (x, y, z)

    return (1.0, 1.0, 1.0)


def simplify_to_3d(image: np.ndarray) -> np.ndarray:
    """Convert image arrays to a 3D volume.

    - 3D inputs are returned unchanged.
    - 4D inputs are reduced to the first channel/volume along axis 3.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        return image
    if image.ndim == 4:
        return image[..., 0]
    raise ValueError(f"Expected 3D or 4D image, got shape {image.shape}")


def _to_binary_volume_for_skeletonization(image: np.ndarray) -> np.ndarray:
    """Convert loaded image volume to a boolean mask for skeletonization.

    - Preserve 0/1 integer masks by using ``> 0`` so value ``1`` is foreground.
    - Keep prior skimage conversion behavior for full-range integer images.
    - For normalized floating masks (0..1), threshold at 0.5.
    - For other floating data, use ``> 0`` as a conservative fallback.
    """
    arr = np.asarray(image)
    if arr.dtype == bool:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        if arr_min >= 0 and arr_max <= 1:
            return arr > 0
        return img_as_bool(arr)

    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=bool)
        if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            return arr > 0.5
        return arr > 0.0

    return arr.astype(bool)


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


def load_3d_tif_with_voxel_size(filepath: str) -> tuple[np.ndarray, float, float, float]:
    """Load 3D TIFF image and return image with voxel sizes (x, y, z)."""
    with tifffile.TiffFile(filepath) as tif:
        image = tif.asarray()
        meta = tif.imagej_metadata or {}
        tags = tif.pages[0].tags

        x_res_tag = tags.get("XResolution")
        y_res_tag = tags.get("YResolution")

        if x_res_tag:
            x_res = x_res_tag.value[0] / x_res_tag.value[1]
        else:
            print("No x resolution tag found; defaulting to 1.0")
            x_res = 1.0

        if y_res_tag:
            y_res = y_res_tag.value[0] / y_res_tag.value[1]
        else:
            print("No y resolution tag found; defaulting to 1.0")
            y_res = 1.0

        if "spacing" in meta:
            z_res = float(meta.get("spacing"))
        else:
            print("No z resolution (spacing) found; defaulting to 1.0")
            z_res = 1.0

        voxel_size_x = 1.0 / x_res if x_res else 1.0
        voxel_size_y = 1.0 / y_res if y_res else 1.0
        voxel_size_z = z_res

    return image, voxel_size_x, voxel_size_y, voxel_size_z


def load_3d_h5_with_voxel_size(
    filepath: str,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Load 3D H5 image and return image with voxel sizes (x, y, z)."""
    if h5py is None:
        raise ImportError("h5py is required to load .h5 files. Install with `pip install h5py`.")
    if dataset_name is None:
        path = Path(filepath)
        if path.suffix != ".h5":
            raise ValueError(f"Expected a .h5 file, got: {filepath}")
        dataset_name = path.stem
        logger.debug("Auto-parsed dataset name: %s", dataset_name)

    with h5py.File(filepath, "r") as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {filepath}. "
                f"Available datasets: {available}"
            )
        dataset = f[dataset_name]
        image = np.array(dataset)
        voxel_size_x, voxel_size_y, voxel_size_z = _extract_h5_voxel_size(dataset, f)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    return image, voxel_size_x, voxel_size_y, voxel_size_z


def load_and_skeletonize_3d_tif(filepath: str):
    print("Loading and skeletonizing TIFF...")
    image, voxel_size_x, voxel_size_y, voxel_size_z = load_3d_tif_with_voxel_size(filepath)

    print("Voxel size — x: %s, y: %s, z: %s", voxel_size_x, voxel_size_y, voxel_size_z)
    binary = _to_binary_volume_for_skeletonization(image)
    skeleton = skeletonize_3d(binary)
    skeleton = binary_fill_holes(skeleton)
    return image, skeleton.astype(bool), voxel_size_x, voxel_size_y, voxel_size_z


def load_and_skeletonize_3d_h5(
    filepath: str,
    dataset_name: str | None = None,
):
    logger.debug("Loading and skeletonizing H5...")
    image, voxel_size_x, voxel_size_y, voxel_size_z = load_3d_h5_with_voxel_size(
        filepath,
        dataset_name=dataset_name,
    )

    logger.debug("Original image shape: %s", image.shape)
    logger.debug("Simplified image shape: %s", image.shape)

    # Ensure image is (X, Y, Z)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    binary = _to_binary_volume_for_skeletonization(image)
    skeleton = skeletonize_3d(binary)
    return image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z

