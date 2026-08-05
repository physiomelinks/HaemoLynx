"""Load 3D images from TIFF or HDF5 and produce skeleton."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from skimage.util import img_as_bool
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes
try:
    import h5py
except ImportError:
    h5py = None

from .axis_order import CANONICAL_AXIS_ORDER, apply_axis_order

logger = logging.getLogger(__name__)


def _is_valid_voxel_size_triplet(voxel_size_xyz: tuple[float, float, float]) -> bool:
    """Return True when voxel-size tuple is finite and strictly positive."""
    arr = np.asarray(voxel_size_xyz, dtype=float).ravel()
    return bool(arr.size == 3 and np.all(np.isfinite(arr)) and np.all(arr > 0))


def _default_voxel_meta_status(source: str, status: str, **extra) -> dict[str, object]:
    """Create a normalized voxel metadata status payload."""
    payload: dict[str, object] = {
        "source": source,
        "status": status,
    }
    payload.update(extra)
    return payload


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


def _extract_h5_voxel_size(dataset, h5_file) -> tuple[tuple[float, float, float], dict[str, object]]:
    """Extract (x, y, z) voxel size and metadata status from HDF5 attrs."""
    attrs = {}
    for source in (h5_file.attrs, dataset.attrs):
        for key in source.keys():
            attrs[str(key).lower()] = source[key]

    # Most common in microscopy exports: element_size_um is usually stored as (z, y, x).
    if "element_size_um" in attrs:
        zyx = _coerce_triplet(attrs["element_size_um"])
        if zyx is not None:
            z, y, x = zyx
            voxel_size_xyz = (x, y, z)
            if _is_valid_voxel_size_triplet(voxel_size_xyz):
                return voxel_size_xyz, _default_voxel_meta_status(
                    source="h5_attributes",
                    status="complete",
                    key_used="element_size_um",
                    axis_order="zyx_to_xyz",
                )
            return (1.0, 1.0, 1.0), _default_voxel_meta_status(
                source="h5_attributes",
                status="invalid",
                key_used="element_size_um",
                fallback_applied=True,
            )

    # Common generic triplet keys; assume they are already ordered (x, y, z).
    for key in ("voxel_size", "voxel_size_um", "pixelsize", "pixel_size", "resolution"):
        if key in attrs:
            xyz = _coerce_triplet(attrs[key])
            if xyz is not None:
                if _is_valid_voxel_size_triplet(xyz):
                    return xyz, _default_voxel_meta_status(
                        source="h5_attributes",
                        status="complete",
                        key_used=key,
                    )
                return (1.0, 1.0, 1.0), _default_voxel_meta_status(
                    source="h5_attributes",
                    status="invalid",
                    key_used=key,
                    fallback_applied=True,
                )

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

    available_axes: list[str] = []
    if x is not None:
        available_axes.append("x")
    if y is not None:
        available_axes.append("y")
    if z is not None:
        available_axes.append("z")

    if x is not None and y is not None and z is not None:
        voxel_size_xyz = (x, y, z)
        if _is_valid_voxel_size_triplet(voxel_size_xyz):
            return voxel_size_xyz, _default_voxel_meta_status(
                source="h5_attributes",
                status="complete",
                key_used="axis_specific",
                available_axes=available_axes,
            )
        return (1.0, 1.0, 1.0), _default_voxel_meta_status(
            source="h5_attributes",
            status="invalid",
            key_used="axis_specific",
            available_axes=available_axes,
            fallback_applied=True,
        )

    status = "missing" if not available_axes else "partial"
    return (1.0, 1.0, 1.0), _default_voxel_meta_status(
        source="h5_attributes",
        status=status,
        key_used="axis_specific",
        available_axes=available_axes,
        fallback_applied=True,
    )


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

    - Preserve low-cardinality integer label masks (e.g., 0/1, 0/255, 1/2).
    - Keep prior skimage conversion behavior for grayscale integer images.
    - For normalized floating masks (0..1), threshold at 0.5.
    - For other floating data, use ``> 0`` as a conservative fallback.
    """
    arr = np.asarray(image)
    if arr.dtype == bool:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        values, counts = np.unique(arr, return_counts=True)
        if values.size == 1:
            return arr > 0
        # Common binary-mask conventions (e.g., 0/1 or 0/255).
        if values.size == 2 and 0 in values:
            fg_value = values[values != 0][0]
            return arr == fg_value
        # Two non-zero labels often mean background/foreground without 0.
        # Use the minority class as foreground (e.g. 1/2 encoded masks).
        if values.size == 2:
            fg_value = values[int(np.argmin(counts))]
            return arr == fg_value
        # For very small integer label sets, pick the least frequent non-zero
        # class as foreground and treat zero as background when present.
        if values.size <= 4:
            nonzero_values = values[values != 0]
            if nonzero_values.size > 0:
                nonzero_counts = np.array(
                    [counts[np.where(values == v)[0][0]] for v in nonzero_values]
                )
                fg_value = nonzero_values[int(np.argmin(nonzero_counts))]
                return arr == fg_value
        arr_min = int(values.min())
        arr_max = int(values.max())
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


def load_3d_tif_with_voxel_size(
    filepath: str,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Load 3D TIFF image and return image + voxel size (x, y, z) + metadata status.

    *axis_order* names what the file's array axes mean (default ``"zyx"``); the
    volume is transposed into the canonical ``(z, y, x)`` order on load. The
    returned voxel size is always physical ``(x, y, z)`` — use
    :func:`ImageLynx.io.voxel_size_zyx_from_xyz` before scaling array indices.
    """
    with tifffile.TiffFile(filepath) as tif:
        image = apply_axis_order(tif.asarray(), axis_order)
        meta = tif.imagej_metadata or {}
        tags = tif.pages[0].tags

        x_res_tag = tags.get("XResolution")
        y_res_tag = tags.get("YResolution")
        missing_axes: list[str] = []
        invalid_axes: list[str] = []

        if x_res_tag:
            x_res = x_res_tag.value[0] / x_res_tag.value[1]
        else:
            print("No x resolution tag found; defaulting to 1.0")
            x_res = 1.0
            missing_axes.append("x")

        if y_res_tag:
            y_res = y_res_tag.value[0] / y_res_tag.value[1]
        else:
            print("No y resolution tag found; defaulting to 1.0")
            y_res = 1.0
            missing_axes.append("y")

        if "spacing" in meta:
            z_res = float(meta.get("spacing"))
        else:
            print("No z resolution (spacing) found; defaulting to 1.0")
            z_res = 1.0
            missing_axes.append("z")

        voxel_size_x = 1.0 / x_res if x_res else 1.0
        voxel_size_y = 1.0 / y_res if y_res else 1.0
        voxel_size_z = z_res

    if x_res <= 0:
        invalid_axes.append("x")
    if y_res <= 0:
        invalid_axes.append("y")
    if z_res <= 0:
        invalid_axes.append("z")
    if not np.isfinite(voxel_size_x):
        invalid_axes.append("x")
    if not np.isfinite(voxel_size_y):
        invalid_axes.append("y")
    if not np.isfinite(voxel_size_z):
        invalid_axes.append("z")
    invalid_axes = sorted(set(invalid_axes))

    if invalid_axes and len(invalid_axes) == 3:
        status = "invalid"
    elif invalid_axes:
        status = "partial"
    elif missing_axes and len(missing_axes) == 3:
        status = "missing"
    elif missing_axes:
        status = "partial"
    else:
        status = "complete"

    voxel_meta_status = _default_voxel_meta_status(
        source="tiff_metadata",
        status=status,
        missing_axes=sorted(set(missing_axes)),
        invalid_axes=invalid_axes,
        fallback_applied=bool(missing_axes or invalid_axes),
    )
    return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_3d_h5_with_voxel_size(
    filepath: str,
    dataset_name: str | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Load 3D H5 image and return image + voxel size (x, y, z) + metadata status.

    *axis_order* names what the dataset's array axes mean (default ``"zyx"``);
    the volume is transposed into the canonical ``(z, y, x)`` order on load.
    """
    if h5py is None:
        raise ImportError("h5py is required to load .h5 files. Install with `pip install h5py`.")
    path = Path(filepath)
    if dataset_name is None and path.suffix not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected a .h5/.hdf5 file, got: {filepath}")

    with h5py.File(filepath, "r") as f:
        available = list(f.keys())
        selected_name = dataset_name

        if selected_name is None:
            # Robust autodetection for common H5 layouts:
            # 1) stem-matching dataset, 2) common canonical names,
            # 3) single top-level dataset fallback.
            candidates = [path.stem, "data", "image", "volume"]
            for candidate in candidates:
                if candidate in f:
                    selected_name = candidate
                    break
            if selected_name is None and len(available) == 1:
                selected_name = available[0]
            if selected_name is None:
                raise KeyError(
                    f"Could not auto-select dataset in {filepath}. "
                    f"Available datasets: {available}. "
                    "Please pass dataset_name explicitly."
                )
            logger.debug("Auto-selected H5 dataset name: %s", selected_name)

        if selected_name not in f:
            raise KeyError(
                f"Dataset '{selected_name}' not found in {filepath}. "
                f"Available datasets: {available}"
            )
        dataset = f[selected_name]
        image = np.array(dataset)
        (
            (voxel_size_x, voxel_size_y, voxel_size_z),
            voxel_meta_status,
        ) = _extract_h5_voxel_size(dataset, f)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    image = apply_axis_order(image, axis_order)

    return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_and_skeletonize_3d_tif(filepath: str, *, axis_order: str = CANONICAL_AXIS_ORDER):
    """Load a TIFF in canonical ``(z, y, x)`` order and skeletonize it."""
    print("Loading and skeletonizing TIFF...")
    (
        image,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_meta_status,
    ) = load_3d_tif_with_voxel_size(filepath, axis_order=axis_order)

    print("Voxel size — x: %s, y: %s, z: %s", voxel_size_x, voxel_size_y, voxel_size_z)
    binary = _to_binary_volume_for_skeletonization(image)
    skeleton = skeletonize(binary.astype(bool), method="lee")
    skeleton = binary_fill_holes(skeleton)
    return image, skeleton.astype(bool), voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_and_skeletonize_3d_h5(
    filepath: str,
    dataset_name: str | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
):
    """Load an H5 volume in canonical ``(z, y, x)`` order and skeletonize it."""
    logger.debug("Loading and skeletonizing H5...")
    (
        image,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_meta_status,
    ) = load_3d_h5_with_voxel_size(
        filepath,
        dataset_name=dataset_name,
        axis_order=axis_order,
    )

    logger.debug("Original image shape: %s", image.shape)
    logger.debug("Simplified image shape: %s", image.shape)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    binary = _to_binary_volume_for_skeletonization(image)
    skeleton = skeletonize(binary.astype(bool), method="lee")
    return image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status

