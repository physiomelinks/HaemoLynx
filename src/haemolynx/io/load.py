"""Load 3D images from TIFF or HDF5 and produce skeleton."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from skimage.util import img_as_bool
from ..preprocessing.skeleton import fill_binary_holes, skeletonize_by_component
from ..preprocessing.memmap_support import (
    new_memmap_array,
    release_memmap_array,
    slab_step,
    release_superseded,
)
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


def _unique_with_counts(arr: np.ndarray, *, use_memmap: bool) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(arr, return_counts=True)`` without flattening *arr* whole.

    ``np.unique`` sorts a flattened copy of its input to find distinct
    values -- for a ``use_memmap_loading`` volume, that copy is a fresh,
    plain-RAM, full-volume buffer regardless of how the source pixels are
    stored, exactly what ``use_memmap_loading`` exists to avoid (and large
    enough on its own to be the allocation that fails, before skeletonize
    ever runs). With *use_memmap* False, this is the plain call, unchanged.
    With it True, tallies are accumulated one slab of axis-0 slices at a
    time instead, each slab's own count bounded to that slab's size rather
    than the whole volume's -- by ``np.bincount`` for 8- and 16-bit integers
    (linear, no sort), by ``np.unique`` otherwise.
    """
    if not use_memmap:
        return np.unique(arr, return_counts=True)
    tally: dict[object, int] = {}
    small_int = np.issubdtype(arr.dtype, np.integer) and arr.dtype.itemsize <= 2
    offset = int(np.iinfo(arr.dtype).min) if small_int else 0
    step = slab_step(arr.shape)
    for start in range(0, arr.shape[0], step):
        slab = np.asarray(arr[start:start + step])
        if small_int:
            counts = np.bincount((slab.astype(np.int32) - offset).ravel())
            present = np.flatnonzero(counts)
            slab_values, slab_counts = present + offset, counts[present]
        else:
            slab_values, slab_counts = np.unique(slab, return_counts=True)
        for value, count in zip(slab_values.tolist(), slab_counts.tolist()):
            tally[value] = tally.get(value, 0) + count
    ordered = sorted(tally)
    values = np.array(ordered, dtype=arr.dtype)
    counts = np.array([tally[value] for value in ordered], dtype=np.int64)
    return values, counts


def _finite_range(arr: np.ndarray, *, use_memmap: bool) -> tuple[float, float] | None:
    """``(min, max)`` of *arr*'s finite values, or None if it has none.

    With *use_memmap*, accumulated one axis-0 slice at a time: the plain
    ``arr[np.isfinite(arr)]`` builds a full-volume mask and then a copy of
    every finite value -- two volume-sized plain-RAM arrays.
    """
    if not use_memmap:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())
    low, high = np.inf, -np.inf
    step = slab_step(arr.shape)
    for start in range(0, arr.shape[0], step):
        slab = np.asarray(arr[start:start + step])
        finite = slab[np.isfinite(slab)]
        if finite.size:
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))
    return None if low > high else (low, high)


def _binarization_rule(arr: np.ndarray, *, use_memmap: bool):
    """The elementwise rule that turns *arr* into a foreground mask.

    Decided from the whole volume (its distinct values, or its finite
    range), but applied elementwise -- so the same rule gives the same
    answer whether it is applied to the whole array at once or to one
    slice at a time.
    """
    if np.issubdtype(arr.dtype, np.integer):
        values, counts = _unique_with_counts(arr, use_memmap=use_memmap)
        if values.size == 1:
            return lambda a: a > 0
        # Common binary-mask conventions (e.g., 0/1 or 0/255).
        if values.size == 2 and 0 in values:
            fg_value = values[values != 0][0]
            return lambda a: a == fg_value
        # Two non-zero labels often mean background/foreground without 0.
        # Use the minority class as foreground (e.g. 1/2 encoded masks).
        if values.size == 2:
            fg_value = values[int(np.argmin(counts))]
            return lambda a: a == fg_value
        # For very small integer label sets, pick the least frequent non-zero
        # class as foreground and treat zero as background when present.
        if values.size <= 4:
            nonzero_values = values[values != 0]
            if nonzero_values.size > 0:
                nonzero_counts = np.array(
                    [counts[np.where(values == v)[0][0]] for v in nonzero_values]
                )
                fg_value = nonzero_values[int(np.argmin(nonzero_counts))]
                return lambda a: a == fg_value
        arr_min = int(values.min())
        arr_max = int(values.max())
        if arr_min >= 0 and arr_max <= 1:
            return lambda a: a > 0
        # A threshold at half the dtype's range -- elementwise and
        # independent of the data, so it is the same per slice.
        return img_as_bool

    if np.issubdtype(arr.dtype, np.floating):
        finite_range = _finite_range(arr, use_memmap=use_memmap)
        if finite_range is None:
            return lambda a: np.zeros(a.shape, dtype=bool)
        if finite_range[0] >= 0.0 and finite_range[1] <= 1.0:
            return lambda a: a > 0.5
        return lambda a: a > 0.0

    return lambda a: a.astype(bool)


def _to_binary_volume_for_skeletonization(
    image: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """Convert loaded image volume to a boolean mask for skeletonization.

    - Preserve low-cardinality integer label masks (e.g., 0/1, 0/255, 1/2).
    - Keep prior skimage conversion behavior for grayscale integer images.
    - For normalized floating masks (0..1), threshold at 0.5.
    - For other floating data, use ``> 0`` as a conservative fallback.

    Uses ``asanyarray``, not ``asarray``, for the already-bool pass-through
    case specifically: an ``np.memmap`` loaded with ``use_memmap_loading``
    would otherwise come back demoted to a plain in-RAM-looking ``ndarray``
    (same underlying disk-backed buffer, since this is still a view rather
    than a copy, but the wrong type for anything downstream that checks
    ``isinstance(x, np.memmap)``) -- the same reasoning as
    ``io.axis_order.apply_axis_order``.

    *use_memmap*, when True, finds the volume's distinct values (see
    :func:`_unique_with_counts`) or finite range (:func:`_finite_range`)
    one slice at a time, and writes the mask itself one slice at a time
    into a new memmap in *memmap_directory* -- otherwise the final
    comparison alone is a fresh, plain-RAM boolean array the size of the
    whole volume. The already-bool pass-through is unaffected either way.
    """
    arr = np.asanyarray(image)
    if arr.dtype == bool:
        return arr
    rule = _binarization_rule(arr, use_memmap=use_memmap)
    if not use_memmap:
        return rule(arr)
    result = new_memmap_array(arr.shape, bool, directory=memmap_directory)
    step = slab_step(arr.shape)
    for start in range(0, arr.shape[0], step):
        result[start:start + step] = rule(np.asarray(arr[start:start + step]))
    return result


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


def _voxel_size_xyz_from_tiff(tif) -> tuple[float, float, float, dict[str, object]]:
    """Voxel size (x, y, z) and its provenance, from an open TIFF's tags.

    Split out from :func:`load_3d_tif_with_voxel_size` so the size can be read
    without the pixels: the tags are in the first page's header, and a caller
    that only wants to know how big a voxel is should not have to read a
    300 MB stack to find out. See :func:`read_voxel_size_xyz`.
    """
    meta = tif.imagej_metadata or {}
    tags = tif.pages[0].tags

    x_res_tag = tags.get("XResolution")
    y_res_tag = tags.get("YResolution")
    missing_axes: list[str] = []
    invalid_axes: list[str] = []

    if x_res_tag:
        x_res = x_res_tag.value[0] / x_res_tag.value[1]
    else:
        logger.warning("No x resolution tag found; defaulting to 1.0")
        x_res = 1.0
        missing_axes.append("x")

    if y_res_tag:
        y_res = y_res_tag.value[0] / y_res_tag.value[1]
    else:
        logger.warning("No y resolution tag found; defaulting to 1.0")
        y_res = 1.0
        missing_axes.append("y")

    if "spacing" in meta:
        z_res = float(meta.get("spacing"))
    else:
        logger.warning("No z resolution (spacing) found; defaulting to 1.0")
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
    return voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def read_voxel_size_xyz(
    filepath: str | Path,
) -> tuple[tuple[float, float, float], dict[str, object]] | None:
    """Physical voxel size ``(x, y, z)`` of a TIFF, without reading the pixels.

    Returns ``None`` for a file that is not a readable TIFF, and for one whose
    tags say nothing -- a size of all ones is the absence of an answer, not an
    answer. The panel uses this to give an opened layer the scale its own file
    describes, which is a header read rather than a second copy of the stack.
    """
    path = Path(filepath)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            x, y, z, status = _voxel_size_xyz_from_tiff(tif)
    except Exception:  # noqa: BLE001 - an unreadable file simply has no size
        logger.debug("Could not read a voxel size from %s", path, exc_info=True)
        return None
    if (x, y, z) == (1.0, 1.0, 1.0):
        return None
    return (x, y, z), status


def load_3d_tif_with_voxel_size(
    filepath: str,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    allow_2d: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    memmap_path: str | Path | None = None,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Load 3D TIFF image and return image + voxel size (x, y, z) + metadata status.

    *axis_order* names what the file's array axes mean (default ``"zyx"``); the
    volume is transposed into the canonical ``(z, y, x)`` order on load. The
    returned voxel size is always physical ``(x, y, z)`` — use
    :func:`haemolynx.io.voxel_size_zyx_from_xyz` before scaling array indices.

    *allow_2d*, when True, returns a genuinely 2D image as-is instead of
    letting :func:`apply_axis_order` raise for a non-default *axis_order* --
    a 2-axis array has no ``(z, y, x)`` to reorder, so it is returned
    untransposed and the caller is responsible for turning it into a volume.
    Mirrors :func:`load_3d_h5_with_voxel_size`'s own *allow_2d*;
    :mod:`haemolynx.io.load_2d` is what actually sets this. Every other
    caller leaves it False, unchanged from before this parameter existed.

    *use_memmap*, when True, decompresses the pixel data into a disk-backed
    ``numpy.memmap`` (a temporary file, cleaned up by tifffile itself once
    the returned array is garbage-collected) instead of a plain in-RAM
    array -- for a volume too large to hold in memory outright. Compression
    on the file does not change what this needs: every voxel still has to
    be materialised somewhere to be indexed at all, this only chooses disk
    over RAM for where. *memmap_directory*, when given, is where that
    temporary file is created (tifffile's own ``"memmap:<dir>"`` syntax)
    instead of the OS default temp directory. *memmap_path*, when given,
    takes precedence over *memmap_directory* and writes to that exact file
    instead of a randomly-named one -- a persistent file at a caller-chosen
    path, e.g. for :mod:`haemolynx.io.raw_volume_cache` to reopen on a later
    run, rather than a temp file tifffile deletes once nothing references it.
    Both are ignored when *use_memmap* is False.
    """
    with tifffile.TiffFile(filepath) as tif:
        if use_memmap:
            if memmap_path:
                Path(memmap_path).parent.mkdir(parents=True, exist_ok=True)
                out = str(memmap_path)
            elif memmap_directory:
                Path(memmap_directory).mkdir(parents=True, exist_ok=True)
                out = f"memmap:{memmap_directory}"
            else:
                out = "memmap"
        else:
            out = None
        raw = tif.asarray(out=out)
        if allow_2d and raw.ndim == 2:
            image = raw
        else:
            image = apply_axis_order(
                raw, axis_order, use_memmap=use_memmap, memmap_directory=memmap_directory
            )
        voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status = (
            _voxel_size_xyz_from_tiff(tif)
        )
    return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_3d_h5_with_voxel_size(
    filepath: str,
    dataset_name: str | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    allow_2d: bool = False,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    memmap_path: str | Path | None = None,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Load 3D H5 image and return image + voxel size (x, y, z) + metadata status.

    *axis_order* names what the dataset's array axes mean (default ``"zyx"``);
    the volume is transposed into the canonical ``(z, y, x)`` order on load.

    *allow_2d*, when True, returns a genuinely 2D dataset as-is instead of
    raising -- *axis_order* is not applied to it, since a 2-axis array has
    no ``(z, y, x)`` to reorder; the caller is responsible for turning it
    into a volume. :mod:`haemolynx.io.load_2d` is what actually sets this
    and does that promotion. Every other caller leaves it False, unchanged
    from before this parameter existed.

    *use_memmap*, when True, copies the dataset into a disk-backed
    ``numpy.memmap`` one slice (along axis 0) at a time, instead of
    ``h5py``'s own ``[...]`` reading the whole thing into a fresh in-RAM
    array in one call -- unlike tifffile's own memmap support, h5py has no
    built-in equivalent, so this bounds the *copy's* own peak memory too,
    not just where the final array ends up. The caller owns the returned
    array's backing file (see :func:`haemolynx.preprocessing.new_memmap_array`)
    the same way it would own a plain array. *memmap_directory* is forwarded
    to :func:`haemolynx.preprocessing.new_memmap_array`. *memmap_path*, when
    given, takes precedence and copies the dataset into that exact
    persistent file instead of a randomly-named one -- see
    :func:`load_3d_tif_with_voxel_size`'s own *memmap_path*. Both are
    ignored when *use_memmap* is False.
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
        if use_memmap:
            if memmap_path:
                Path(memmap_path).parent.mkdir(parents=True, exist_ok=True)
                image = np.memmap(
                    memmap_path, dtype=dataset.dtype, mode="w+", shape=dataset.shape
                )
            else:
                image = new_memmap_array(
                    dataset.shape, dataset.dtype, directory=memmap_directory
                )
            # Slabs of whole HDF5 chunks along axis 0: read a slice at a
            # time, a chunk spanning 16 slices is decompressed 16 times.
            chunk_depth = int(dataset.chunks[0]) if dataset.chunks else 1
            step = slab_step(dataset.shape)
            step = max(chunk_depth, step - step % chunk_depth)
            for start in range(0, dataset.shape[0], step):
                image[start:start + step] = dataset[start:start + step]
        else:
            image = np.array(dataset)
        (
            (voxel_size_x, voxel_size_y, voxel_size_z),
            voxel_meta_status,
        ) = _extract_h5_voxel_size(dataset, f)

    if image.ndim == 2 and allow_2d:
        return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    reordered = apply_axis_order(
        image, axis_order, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    if use_memmap and not memmap_path and reordered is not image:
        # The untransposed copy was this function's own temp file, and
        # nothing else will ever hold it now that the reordered copy exists.
        release_memmap_array(image)

    return reordered, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_volume_and_voxel_size(
    volume_path: str | Path,
    *,
    h5_dataset_name: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    description: str = "mask",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a TIFF/H5 volume in canonical ``(z, y, x)`` order with its voxel size.

    The returned voxel size is image-metadata order ``(x, y, z)``; convert with
    :func:`voxel_size_zyx_from_xyz` before scaling array indices. ``description``
    names the volume in the unsupported-format error, so callers can say what the
    user actually passed ("pericyte mask", "cell mask", ...). ``use_memmap`` and
    ``memmap_directory`` are forwarded to whichever of
    :func:`load_3d_tif_with_voxel_size` / :func:`load_3d_h5_with_voxel_size`
    actually reads the file.
    """
    path = Path(volume_path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_tif_with_voxel_size(
            str(path),
            axis_order=axis_order,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    elif suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_h5_with_voxel_size(
            str(path),
            dataset_name=h5_dataset_name,
            axis_order=axis_order,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )
    else:
        raise ValueError(
            f"Unsupported {description} format '{suffix}'. "
            "Expected .tif, .tiff, or .h5."
        )
    return image, (float(voxel_x), float(voxel_y), float(voxel_z))


def load_binary_mask_and_voxel_size(
    mask_path: str | Path,
    *,
    h5_dataset_name: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    description: str = "mask",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a 3D binary mask and return ``(mask_bool, voxel_size_xyz)``.

    Uses the same binarisation as skeletonisation inputs: ``0/1``, ``0/255``,
    and two-label encodings without a zero background (e.g. ``1/2``) all become
    a sparse boolean foreground. A raw ``> 0`` cast is *not* used -- on a
    ``1/2`` mask it would mark every voxel True and poison every downstream
    consumer (dilation, volume filters, terminal assignment, napari display).

    The path may point at a file inside a sibling zip archive; see
    :func:`resolve_image_path_with_optional_zip`. ``use_memmap`` reads the
    file disk-backed and, via :func:`_to_binary_volume_for_skeletonization`,
    writes the boolean mask a slice at a time into a new disk-backed array
    in *memmap_directory*, which the caller releases; the mask is the same
    either way.
    """
    path = resolve_image_path_with_optional_zip(Path(mask_path))
    image, voxel_size_xyz = load_volume_and_voxel_size(
        path,
        h5_dataset_name=h5_dataset_name,
        axis_order=axis_order,
        description=description,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D {description}, got shape {image.shape}.")
    return (
        _to_binary_volume_for_skeletonization(
            image, use_memmap=use_memmap, memmap_directory=memmap_directory
        ),
        voxel_size_xyz,
    )


def _skeletonize_loaded_volume(
    image: np.ndarray,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
    tile_large_components: bool = False,
    tile_max_voxels: int = 200_000_000,
    tile_halo_voxels: int = 0,
) -> np.ndarray:
    """Binarize, skeletonize and fill holes: what every loader must do.

    One function because the TIFF and H5 loaders drifted apart -- the TIFF one
    filled holes and the H5 one did not, so the same volume produced a
    different skeleton depending on the file it was saved in. The hole fill was
    added deliberately to the TIFF path (issue #4) and the graph builder has
    been tuned against skeletons that have it, so the H5 path was the one that
    was wrong. It is a no-op on the normal case anyway: a thin 3D curve
    encloses no background, which is what
    ``test_fill_binary_holes_on_a_sparse_skeleton_changes_nothing`` pins.

    *use_memmap* is forwarded to :func:`skeletonize_by_component` (whose
    per-component splitting, when it actually shrinks what skimage's own
    Lee thinning is called on, is the widest lever this path has for a
    volume too large to skeletonize as one piece) and to
    :func:`fill_binary_holes`, whose own connected-component labelling is a
    full-volume, wider-than-boolean buffer this path can also redirect to
    disk. *tile_large_components*, *tile_max_voxels* and
    *tile_halo_voxels* are also forwarded to
    :func:`skeletonize_by_component`, for the residual case where a single
    connected component is still too large after the per-component split
    -- see its own docstring.

    Neither ``_to_binary_volume_for_skeletonization``'s result nor
    ``fill_binary_holes``'s is ever re-cast with ``.astype(bool)`` here --
    both already guarantee a boolean return, and ``.astype()`` defaults to
    ``copy=True`` even when the dtype already matches, which would silently
    materialise a second full-volume plain-RAM copy of whichever one, memmap
    or not, use_memmap_loading just avoided making one for.
    """
    binary = _to_binary_volume_for_skeletonization(
        image, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    skeleton = skeletonize_by_component(
        binary,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
        tile_large_components=tile_large_components,
        tile_max_voxels=tile_max_voxels,
        tile_halo_voxels=tile_halo_voxels,
    )
    # Each of these is a volume-sized temp file under use_memmap; neither is
    # needed once the step after it has run. `image` itself is the caller's.
    release_superseded(binary, skeleton, keep=image)
    filled = fill_binary_holes(skeleton, use_memmap=use_memmap, memmap_directory=memmap_directory)
    release_superseded(skeleton, filled, keep=image)
    return filled


def load_and_skeletonize_3d_tif(
    filepath: str,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
):
    """Load a TIFF in canonical ``(z, y, x)`` order and skeletonize it."""
    logger.info("Loading and skeletonizing TIFF...")
    (
        image,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_meta_status,
    ) = load_3d_tif_with_voxel_size(
        filepath,
        axis_order=axis_order,
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )

    logger.info("Voxel size — x: %s, y: %s, z: %s", voxel_size_x, voxel_size_y, voxel_size_z)
    skeleton = _skeletonize_loaded_volume(
        image, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    return image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status


def load_and_skeletonize_3d_h5(
    filepath: str,
    dataset_name: str | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
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
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )

    logger.debug("Original image shape: %s", image.shape)
    logger.debug("Simplified image shape: %s", image.shape)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after simplification, got shape: {image.shape}")

    skeleton = _skeletonize_loaded_volume(
        image, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    return image, skeleton, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status

