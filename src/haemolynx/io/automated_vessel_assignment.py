"""Helpers for optional automated large-vessel mask loading."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .axis_order import CANONICAL_AXIS_ORDER, voxel_size_zyx_from_xyz
from .crop import CropBox, crop_volume
from .downsample import Downsample, downsample_volume
from .ilastik import run_ilastik_headless_segmentation
from .load import (
    _to_binary_volume_for_skeletonization,
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    resolve_image_path_with_optional_zip,
)
from .raw_volume_cache import _modified_too_recently
from .voxel_validation import resolve_voxel_size_xyz

logger = logging.getLogger(__name__)


def _load_mask_image(
    mask_path: Path,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, object]]:
    """Load a vessel mask as a boolean volume in canonical ``(z, y, x)`` order.

    Must binarise with the shared mask loader, not a raw intensity read:
    ``astype(bool)`` / ``> 0`` on a ``1/2``-encoded mask fills the whole
    volume, and opposite-attached volume cleanup then treats every venule
    voxel as attached to that solid arteriole and can wipe the venule mask.

    Returns the mask, metadata voxel size ``(x, y, z)``, and the metadata
    status dict so callers can apply the same override policy as the main
    image. *use_memmap* (the low-RAM option) reads it, and binarises it, onto
    disk as the main image is read (``memmap_directory``), not into RAM.
    """
    path = Path(mask_path)
    suffix = path.suffix.lower()
    low_ram = dict(use_memmap=True, memmap_directory=memmap_directory) if use_memmap else {}
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z, voxel_meta_status = load_3d_tif_with_voxel_size(
            str(path),
            axis_order=axis_order,
            **low_ram,
        )
    elif suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z, voxel_meta_status = load_3d_h5_with_voxel_size(
            str(path),
            axis_order=axis_order,
            **low_ram,
        )
    else:
        raise ValueError(
            f"Unsupported mask format '{suffix}'. Expected .tif, .tiff, or .h5."
        )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {image.shape}.")
    mask = _to_binary_volume_for_skeletonization(image, **low_ram)
    return mask, (float(voxel_x), float(voxel_y), float(voxel_z)), voxel_meta_status


def _resolve_mask_voxel_size_xyz(
    metadata_voxel_size_xyz: tuple[float, float, float],
    metadata_status: dict[str, object],
    *,
    voxel_size_override_xyz,
    voxel_size_policy: str,
    fallback_voxel_size_xyz: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Apply the run's voxel-size override policy to one mask's metadata.

    *fallback_voxel_size_xyz* is the main image's spacing: under ``auto``, a
    mask with no spacing of its own and no override takes it (an ilastik
    export carries none), rather than running on 1 um and failing the
    alignment check against the image it was segmented from.
    """
    status = str((metadata_status or {}).get("status", "missing")).lower()
    if (
        fallback_voxel_size_xyz is not None
        and str(voxel_size_policy).strip().lower() == "auto"
        and status != "complete"
        and voxel_size_override_xyz is None
    ):
        logger.info(
            "Mask carries no voxel spacing (metadata %s); taking the main image's "
            "(x, y, z) = %s.",
            status,
            tuple(float(v) for v in fallback_voxel_size_xyz),
        )
        return tuple(float(v) for v in fallback_voxel_size_xyz)
    resolved, _source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size_xyz,
        metadata_status=metadata_status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
    )
    return resolved


def _source_voxel_size_xyz(
    main_voxel_size_xyz, downsample: Downsample | None
) -> tuple[float, float, float]:
    """The main image's voxel size as read from disk -- what a mask with no
    spacing of its own takes: the run's, or before *downsample* scaled it."""
    if downsample is None or downsample.is_identity:
        return tuple(float(v) for v in main_voxel_size_xyz)  # type: ignore[return-value]
    return downsample.source_voxel_size_xyz


def load_large_vessel_masks(
    enabled: bool,
    large_arteriole_mask_path: str | Path | None = None,
    large_venule_mask_path: str | Path | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
    fallback_voxel_size_xyz: tuple[float, float, float] | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Load optional large arteriole/venule masks.

    When enabled is False, no mask paths are allowed and (None, None) is returned.
    When enabled is True, the arteriole mask is required. With a venule mask
    too, both are loaded; without one, the arteriole mask is a *global* mask
    -- every vessel of its size, arteriole or venule -- and is returned with
    ``None`` for the venule mask and its voxel size (see
    :func:`vessel_masks_are_global`).

    Voxel sizes are resolved with the same ``voxel_size_override_xyz`` /
    ``voxel_size_policy`` rules as the main image, so a manual unit override
    applies to the masks instead of leaving them on missing-metadata ``(1,1,1)``;
    with neither metadata nor an override, a mask takes
    *fallback_voxel_size_xyz* (the main image's spacing) when given.
    """
    has_arteriole = bool(large_arteriole_mask_path)
    has_venule = bool(large_venule_mask_path)

    if not enabled:
        if has_arteriole or has_venule:
            raise ValueError(
                "Large-vessel masks are disabled. Set enabled=True to provide "
                "large_arteriole_mask_path and large_venule_mask_path."
            )
        return None, None, None, None

    if not has_arteriole:
        raise ValueError(
            "Vessel masks are enabled, but the arteriole/global mask path is "
            "missing. Enter the arteriole mask (with the venule mask), or one "
            "global vessel mask on its own, in the arteriole/global row."
        )

    def _load(path) -> tuple[np.ndarray, tuple[float, float, float]]:
        mask, metadata_xyz, status = _load_mask_image(
            resolve_image_path_with_optional_zip(Path(path)), axis_order=axis_order,
            use_memmap=use_memmap, memmap_directory=memmap_directory,
        )
        voxel_size = _resolve_mask_voxel_size_xyz(
            metadata_xyz,
            status,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
            fallback_voxel_size_xyz=fallback_voxel_size_xyz,
        )
        return mask, voxel_size

    large_arteriole_mask, large_arteriole_voxel_size = _load(large_arteriole_mask_path)
    if not has_venule:
        return large_arteriole_mask, None, large_arteriole_voxel_size, None
    large_venule_mask, large_venule_voxel_size = _load(large_venule_mask_path)
    return (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_voxel_size,
        large_venule_voxel_size,
    )


#: What the log says when a role's masks are off.
_DISABLED_MESSAGES = {
    "large": "Large-vessel masks disabled; skipping arteriole/venule mask loading.",
    "small": (
        "Small-vessel-mask boundary assignment disabled; "
        "manual arteriole/venule boundary-node selection remains available."
    ),
}


def _vessel_mask_config(mask_role: Literal["large", "small"]) -> dict[str, str]:
    """The setting names a role's messages quote, read off
    :data:`VESSEL_MASK_SETTINGS` so the two cannot drift apart."""
    names = VESSEL_MASK_SETTINGS[mask_role]
    return {
        "enabled_flag": names["enabled"],
        "ilastik_flag": names["use_ilastik"],
        "ilastik_arteriole_image_param": names["ilastik_unsegmented_arteriole_path"],
        "ilastik_venule_image_param": names["ilastik_unsegmented_venule_path"],
        "ilastik_arteriole_classifier_param": names["ilastik_arteriole_classifier_path"],
        "ilastik_venule_classifier_param": names["ilastik_venule_classifier_path"],
        "arteriole_mask_attr": f"{mask_role}_arteriole_mask",
        "venule_mask_attr": f"{mask_role}_venule_mask",
        "disabled_message": _DISABLED_MESSAGES[mask_role],
    }


#: Relative tolerance for comparing voxel sizes loaded independently from
#: separate files (main image, arteriole mask, venule mask). Physically
#: identical spacing can still differ by a float ULP or two after passing
#: through a different loader's own division of a TIFF resolution rational
#: tag; a genuine mismatch (wrong axis order, a different acquisition) is
#: orders of magnitude larger than this, so it is not swallowed by it.
_VOXEL_SIZE_RELATIVE_TOLERANCE = 1e-6


def _assert_voxel_sizes_match_main_image(
    *,
    main_voxel_size_xyz: tuple[float, float, float],
    arteriole_voxel_size_xyz: tuple[float, float, float],
    venule_voxel_size_xyz: tuple[float, float, float] | None,
    mask_role: Literal["large", "small"],
) -> None:
    """*venule_voxel_size_xyz* None is a global mask, checked alone."""
    arteriole_label = "arteriole" if mask_role == "large" else "small_arteriole"
    venule_label = "venule" if mask_role == "large" else "small_venule"
    vessel_phrase = "large-vessel masks" if mask_role == "large" else "small-vessel masks"
    if venule_voxel_size_xyz is None:
        arteriole_label = f"{mask_role}_global"
        venule_voxel_size_xyz = arteriole_voxel_size_xyz
    if (
        np.allclose(
            main_voxel_size_xyz,
            arteriole_voxel_size_xyz,
            rtol=_VOXEL_SIZE_RELATIVE_TOLERANCE,
            atol=0.0,
        )
        and np.allclose(
            main_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=_VOXEL_SIZE_RELATIVE_TOLERANCE,
            atol=0.0,
        )
        and np.allclose(
            arteriole_voxel_size_xyz,
            venule_voxel_size_xyz,
            rtol=_VOXEL_SIZE_RELATIVE_TOLERANCE,
            atol=0.0,
        )
    ):
        logger.info(
            f"Voxel-size check passed. {vessel_phrase.capitalize()} are aligned "
            "to the same physical voxel units as the main image."
        )
        return

    error_message = (
        f"Voxel-size mismatch detected across input image and {vessel_phrase}. "
        f"main={main_voxel_size_xyz}, "
        f"{arteriole_label}={arteriole_voxel_size_xyz}, "
        f"{venule_label}={venule_voxel_size_xyz}. "
        "All must match in x, y, and z (within floating-point round-off)."
    )
    logger.error(error_message)
    raise ValueError(error_message)


#: Config setting names per mask role, so a caller with the vessel-mask section
#: of the config can pass it whole instead of naming eleven settings. The small
#: role does not simply prefix the large one, which is why this is written out.
VESSEL_MASK_SETTINGS: dict[str, dict[str, str]] = {
    "large": {
        "enabled": "use_large_vessel_masks",
        "use_ilastik": "use_ilastik_large_vessel_segmentation",
        "arteriole_mask_path": "large_arteriole_mask_path",
        "venule_mask_path": "large_venule_mask_path",
        "ilastik_unsegmented_arteriole_path": "ilastik_unsegmented_arteriole_image_path",
        "ilastik_unsegmented_venule_path": "ilastik_unsegmented_venule_image_path",
        "ilastik_arteriole_classifier_path": "ilastik_arteriole_classifier_path",
        "ilastik_venule_classifier_path": "ilastik_venule_classifier_path",
        "dilation_microns": "large_vessel_mask_dilation_microns",
        "min_component_volume_um3": "large_vessel_min_component_volume_um3",
        "swap_minority_components": "large_vessel_swap_minority_components",
        "swap_max_size_ratio": "large_vessel_swap_max_size_ratio",
        "swap_min_contact_fraction": "large_vessel_swap_min_contact_fraction",
        "remove_small_opposite_attached_components": (
            "large_vessel_remove_small_opposite_attached_components"
        ),
        "opposite_attached_max_component_volume_um3": (
            "large_vessel_opposite_attached_max_component_volume_um3"
        ),
        "opposite_attached_max_distance_microns": (
            "large_vessel_opposite_attached_max_distance_microns"
        ),
        "exclude_smaller_overlapping_volumes": "exclude_smaller_overlapping_volumes",
    },
    "small": {
        "enabled": "use_small_vessel_masks_for_boundary_assignment",
        "use_ilastik": "use_ilastik_small_vessel_segmentation",
        "arteriole_mask_path": "small_arteriole_mask_path",
        "venule_mask_path": "small_venule_mask_path",
        "ilastik_unsegmented_arteriole_path": "ilastik_unsegmented_small_arteriole_image_path",
        "ilastik_unsegmented_venule_path": "ilastik_unsegmented_small_venule_image_path",
        "ilastik_arteriole_classifier_path": "ilastik_small_arteriole_classifier_path",
        "ilastik_venule_classifier_path": "ilastik_small_venule_classifier_path",
        "min_component_volume_um3": "small_vessel_min_component_volume_um3",
        "swap_minority_components": "small_vessel_swap_minority_components",
        "swap_max_size_ratio": "small_vessel_swap_max_size_ratio",
        "swap_min_contact_fraction": "small_vessel_swap_min_contact_fraction",
    },
}

#: Swapped components named one per line in the run log; the rest are counted.
_MAX_LOGGED_SWAPS = 20

#: Settings both roles share, named identically in the config.
_SHARED_VESSEL_MASK_SETTINGS = (
    "ilastik_output_dir",
    "ilastik_output_suffix",
    "ilastik_executable",
    "ilastik_timeout_seconds",
    "ilastik_reuse_existing_output",
    "voxel_size_override_xyz",
    "voxel_size_policy",
)


def vessel_mask_arguments(
    settings: Mapping[str, object], mask_role: Literal["large", "small"]
) -> dict[str, object]:
    """Arguments for :func:`load_and_validate_vessel_masks` for one mask role.

    Takes the vessel-mask and segmentation settings as loaded from the config
    and picks out the ones that role uses, so the caller states the role once
    rather than eleven setting names twice.
    """
    if mask_role not in VESSEL_MASK_SETTINGS:
        raise ValueError(
            f"mask_role must be 'large' or 'small', got {mask_role!r}."
        )
    arguments: dict[str, object] = {"mask_role": mask_role}
    for parameter, setting_name in VESSEL_MASK_SETTINGS[mask_role].items():
        if setting_name in settings:
            arguments[parameter] = settings[setting_name]
    for setting_name in _SHARED_VESSEL_MASK_SETTINGS:
        if setting_name in settings:
            arguments[setting_name] = settings[setting_name]
    if "image_axis_order" in settings:
        arguments["axis_order"] = settings["image_axis_order"]
    return arguments


def load_and_validate_vessel_masks(
    *,
    mask_role: Literal["large", "small"],
    enabled: bool,
    use_ilastik: bool,
    arteriole_mask_path: str | Path | None,
    venule_mask_path: str | Path | None,
    image_shape: tuple[int, ...],
    main_voxel_size_xyz: tuple[float, float, float],
    ilastik_unsegmented_arteriole_path: str | Path | None = None,
    ilastik_unsegmented_venule_path: str | Path | None = None,
    ilastik_arteriole_classifier_path: str | Path | None = None,
    ilastik_venule_classifier_path: str | Path | None = None,
    ilastik_output_dir: str | Path | None = None,
    ilastik_output_suffix: str = ".tif",
    ilastik_executable: str | None = None,
    ilastik_timeout_seconds: float | None = None,
    ilastik_reuse_existing_output: bool = False,
    dilation_microns: float = 0.0,
    min_component_volume_um3: float = 0.0,
    swap_minority_components: bool = False,
    swap_max_size_ratio: float = 1.0,
    swap_min_contact_fraction: float = 0.3,
    remove_small_opposite_attached_components: bool = False,
    opposite_attached_max_component_volume_um3: float = 250.0,
    opposite_attached_max_distance_microns: float = 3.0,
    exclude_smaller_overlapping_volumes: bool = False,
    loaded_message_suffix: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
    crop: CropBox | None = None,
    downsample: Downsample | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Resolve, optionally segment with ilastik, load, and validate arteriole/venule masks.

    With no venule mask (under ilastik, no venule classifier) the arteriole
    one is a global mask of every vessel of the role's size, returned as
    ``(mask, None, voxel size, None)`` (:func:`_load_global_vessel_mask`).

    *crop* is the ``crop_on_import`` box the main image was cut to (see
    :mod:`haemolynx.io.crop`): each mask must have the whole image's shape, and
    is cut to the same box before anything else is done to it. *downsample* is
    how the main image was then reduced (``segmented_image_downsample``,
    :mod:`haemolynx.io.downsample`): each mask is reduced by the same blocks,
    and its voxel size read against the image's as read, then scaled -- so
    *image_shape* and *main_voxel_size_xyz* are the reduced image's.

    *use_memmap* (the low-RAM option) reads the masks onto disk
    (*memmap_directory*), cleans them with their components labelled a slab
    of planes at a time, and hands them, and the ones remembered from an
    earlier load, out on disk. The same masks either way.
    """
    config = _vessel_mask_config(mask_role)
    low_ram = dict(low_memory=bool(use_memmap), memmap_directory=memmap_directory)
    scale_label = mask_role

    if use_ilastik and not enabled:
        raise ValueError(
            f"{config['ilastik_flag']}=True requires {config['enabled_flag']}=True."
        )

    effective_arteriole_mask_path = arteriole_mask_path or None
    effective_venule_mask_path = venule_mask_path or None
    if not enabled:
        effective_arteriole_mask_path = None
        effective_venule_mask_path = None

    if enabled and use_ilastik:
        # With no venule classifier the arteriole one segments a global mask:
        # every vessel of the role's size, which no classifier told apart.
        global_mask = not ilastik_venule_classifier_path
        required = [
            (config["ilastik_arteriole_image_param"], ilastik_unsegmented_arteriole_path),
            (config["ilastik_arteriole_classifier_param"], ilastik_arteriole_classifier_path),
        ]
        if not global_mask:
            required.append(
                (config["ilastik_venule_image_param"], ilastik_unsegmented_venule_path)
            )
        for param_name, value in required:
            if value is None:
                raise ValueError(
                    f"{param_name} must be set when {config['ilastik_flag']}=True."
                )

        output_dir = Path(ilastik_output_dir)
        unsegmented_arteriole_image_path = resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_arteriole_path)
        )
        # Named by role as well as by image: the arteriole and venule
        # classifiers are usually run on the one raw stack, and both outputs
        # called `{stem}_segmented` meant the venule run overwrote the
        # arteriole's (or, reusing existing output, read it back as its own).
        arteriole_class = "global" if global_mask else "arteriole"
        segmented_arteriole_path = output_dir / (
            f"{unsegmented_arteriole_image_path.stem}_{mask_role}_{arteriole_class}_segmented"
            f"{ilastik_output_suffix}"
        )

        logger.info(
            f"Running ilastik segmentation for {scale_label} {arteriole_class} image: "
            f"{unsegmented_arteriole_image_path}"
        )
        effective_arteriole_mask_path = run_ilastik_headless_segmentation(
            input_image_path=unsegmented_arteriole_image_path,
            classifier_path=Path(ilastik_arteriole_classifier_path),
            output_path=segmented_arteriole_path,
            ilastik_executable=ilastik_executable,
            timeout=ilastik_timeout_seconds,
            reuse_existing=bool(ilastik_reuse_existing_output),
        )
        if global_mask:
            effective_venule_mask_path = None
            logger.info(
                f"Using the ilastik-segmented {scale_label}-vessel global mask: "
                f"{effective_arteriole_mask_path}"
            )
        else:
            unsegmented_venule_image_path = resolve_image_path_with_optional_zip(
                Path(ilastik_unsegmented_venule_path)
            )
            segmented_venule_path = output_dir / (
                f"{unsegmented_venule_image_path.stem}_{mask_role}_venule_segmented"
                f"{ilastik_output_suffix}"
            )
            logger.info(
                f"Running ilastik segmentation for {scale_label} venule image: "
                f"{unsegmented_venule_image_path}"
            )
            effective_venule_mask_path = run_ilastik_headless_segmentation(
                input_image_path=unsegmented_venule_image_path,
                classifier_path=Path(ilastik_venule_classifier_path),
                output_path=segmented_venule_path,
                ilastik_executable=ilastik_executable,
                timeout=ilastik_timeout_seconds,
                reuse_existing=bool(ilastik_reuse_existing_output),
            )
            logger.info(
                f"Using ilastik-segmented {scale_label}-vessel masks: "
                f"arteriole={effective_arteriole_mask_path}, "
                f"venule={effective_venule_mask_path}"
            )

    if enabled and effective_arteriole_mask_path is not None and effective_venule_mask_path is None:
        return _load_global_vessel_mask(
            mask_role,
            effective_arteriole_mask_path,
            image_shape=image_shape,
            main_voxel_size_xyz=main_voxel_size_xyz,
            dilation_microns=dilation_microns,
            min_component_volume_um3=min_component_volume_um3,
            axis_order=axis_order,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
            crop=crop,
            downsample=downsample,
            use_memmap=use_memmap,
            memmap_directory=memmap_directory,
        )

    cache_key = None
    if enabled and effective_arteriole_mask_path is not None and effective_venule_mask_path is not None:
        cache_key = _mask_cache_key(
            mask_role,
            effective_arteriole_mask_path,
            effective_venule_mask_path,
            axis_order=axis_order,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
            image_shape=image_shape,
            main_voxel_size_xyz=main_voxel_size_xyz,
            dilation_microns=dilation_microns,
            min_component_volume_um3=min_component_volume_um3,
            swap_minority_components=swap_minority_components,
            swap_max_size_ratio=swap_max_size_ratio,
            swap_min_contact_fraction=swap_min_contact_fraction,
            remove_small_opposite_attached_components=remove_small_opposite_attached_components,
            opposite_attached_max_component_volume_um3=opposite_attached_max_component_volume_um3,
            opposite_attached_max_distance_microns=opposite_attached_max_distance_microns,
            exclude_smaller_overlapping_volumes=exclude_smaller_overlapping_volumes,
            crop=crop,
            downsample=downsample,
        )
        cached = _LOADED_MASKS.get(mask_role)
        if cached is not None and cached[0] == cache_key:
            logger.info(
                f"Reusing the {scale_label}-vessel masks loaded earlier in this "
                "process: the same files, cleaned with the same settings."
            )
            return _unpack_masks(cached[1], use_memmap=use_memmap, memmap_directory=memmap_directory)

    (
        arteriole_mask,
        venule_mask,
        arteriole_mask_voxel_size,
        venule_mask_voxel_size,
    ) = load_large_vessel_masks(
        enabled=enabled,
        large_arteriole_mask_path=effective_arteriole_mask_path,
        large_venule_mask_path=effective_venule_mask_path,
        axis_order=axis_order,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
        fallback_voxel_size_xyz=_source_voxel_size_xyz(main_voxel_size_xyz, downsample),
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )

    if arteriole_mask is None or venule_mask is None:
        logger.info(config["disabled_message"])
        return None, None, None, None

    if crop is not None and not crop.is_whole_volume:
        for name, mask in (
            (config["arteriole_mask_attr"], arteriole_mask),
            (config["venule_mask_attr"], venule_mask),
        ):
            if mask.shape != tuple(crop.source_shape_zyx):
                raise ValueError(
                    f"{name} shape does not match input image shape before "
                    f"crop_on_import cut it: {mask.shape} != {tuple(crop.source_shape_zyx)}"
                )
        logger.info(f"Crop on import, {scale_label}-vessel masks: {crop.describe()}")
        arteriole_mask = crop_volume(arteriole_mask, crop)
        venule_mask = crop_volume(venule_mask, crop)

    if downsample is not None and not downsample.is_identity:
        for name, mask in (
            (config["arteriole_mask_attr"], arteriole_mask),
            (config["venule_mask_attr"], venule_mask),
        ):
            if mask.shape != tuple(downsample.source_shape_zyx):
                raise ValueError(
                    f"{name} shape does not match input image shape before "
                    f"segmented_image_downsample reduced it: {mask.shape} != "
                    f"{tuple(downsample.source_shape_zyx)}"
                )
        arteriole_mask = downsample_volume(
            arteriole_mask, downsample, reduce="majority", description=f"{scale_label}-vessel arteriole mask"
        )
        venule_mask = downsample_volume(
            venule_mask, downsample, reduce="majority", description=f"{scale_label}-vessel venule mask"
        )
        arteriole_mask_voxel_size = downsample.scale_voxel_size_xyz(arteriole_mask_voxel_size)
        venule_mask_voxel_size = downsample.scale_voxel_size_xyz(venule_mask_voxel_size)

    if arteriole_mask.shape != image_shape:
        raise ValueError(
            f"{config['arteriole_mask_attr']} shape does not match input image shape: "
            f"{arteriole_mask.shape} != {image_shape}"
        )
    if venule_mask.shape != image_shape:
        raise ValueError(
            f"{config['venule_mask_attr']} shape does not match input image shape: "
            f"{venule_mask.shape} != {image_shape}"
        )

    arteriole_voxel_size_xyz = tuple(float(v) for v in arteriole_mask_voxel_size)
    venule_voxel_size_xyz = tuple(float(v) for v in venule_mask_voxel_size)
    logger.info(
        f"Loaded {scale_label}-vessel masks: "
        f"arteriole={arteriole_mask.shape}, venule={venule_mask.shape}"
    )
    logger.info(
        f"{scale_label.capitalize()}-vessel mask voxel sizes (x, y, z): "
        f"arteriole={arteriole_mask_voxel_size}, venule={venule_mask_voxel_size}"
    )
    _assert_voxel_sizes_match_main_image(
        main_voxel_size_xyz=main_voxel_size_xyz,
        arteriole_voxel_size_xyz=arteriole_voxel_size_xyz,
        venule_voxel_size_xyz=venule_voxel_size_xyz,
        mask_role=mask_role,
    )

    if dilation_microns > 0:
        from haemolynx.graph.large_vessels import dilate_large_vessel_masks_by_microns

        arteriole_mask, venule_mask = dilate_large_vessel_masks_by_microns(
            large_arteriole_mask=arteriole_mask,
            large_venule_mask=venule_mask,
            dilation_microns=dilation_microns,
            voxel_size_zyx=voxel_size_zyx_from_xyz(main_voxel_size_xyz),
            **low_ram,
        )
        logger.info(
            f"Dilated {scale_label}-vessel masks by {float(dilation_microns):.3f} microns."
        )

    if float(min_component_volume_um3) > 0:
        from haemolynx.graph.mask_component_volume import (
            remove_small_vessel_components_by_volume,
        )

        arteriole_mask, venule_mask, volume_stats = (
            remove_small_vessel_components_by_volume(
                arteriole_mask,
                venule_mask,
                voxel_size_xyz=tuple(float(v) for v in main_voxel_size_xyz),
                min_component_volume_um3=float(min_component_volume_um3),
                **low_ram,
            )
        )
        arteriole_stats = volume_stats.get("arteriole") or {}
        venule_stats = volume_stats.get("venule") or {}
        logger.info(
            f"{scale_label.capitalize()}-vessel component-volume filtering: "
            f"threshold={float(min_component_volume_um3):.3f} um^3, "
            f"removed_components(arteriole="
            f"{int(arteriole_stats.get('removed_component_count', 0))}, "
            f"venule={int(venule_stats.get('removed_component_count', 0))}), "
            f"removed_volume_um3(arteriole="
            f"{float(arteriole_stats.get('removed_volume_um3', 0.0)):.3f}, "
            f"venule={float(venule_stats.get('removed_volume_um3', 0.0)):.3f})."
        )

    # Before the opposite-attached removal, so a small mislabelled piece of a
    # vessel is given back to it rather than deleted and left as a gap.
    if bool(swap_minority_components):
        from haemolynx.graph.large_vessels import (
            swap_minority_touching_vessel_components,
        )

        arteriole_mask, venule_mask, swap_stats = (
            swap_minority_touching_vessel_components(
                arteriole_mask,
                venule_mask,
                max_size_ratio=float(swap_max_size_ratio),
                min_contact_fraction=float(swap_min_contact_fraction),
                voxel_size_zyx=voxel_size_zyx_from_xyz(main_voxel_size_xyz),
                **low_ram,
            )
        )
        # Largest first, so a mask with thousands of mislabelled specks still
        # names the vessels that matter without burying the rest of the log.
        swapped_components = [c for c in swap_stats["components"] if c["swapped"]]
        for component in swapped_components[:_MAX_LOGGED_SWAPS]:
            logger.info(
                f"{scale_label.capitalize()}-vessel minority swap: "
                f"{component['from']} component of {component['voxel_count']} "
                f"voxels -> {component['to']} (touching {component['to']} "
                f"component of {component['largest_partner_voxel_count']} "
                f"voxels, contact fraction {component['contact_fraction']:.3f})."
            )
        if len(swapped_components) > _MAX_LOGGED_SWAPS:
            logger.info(
                f"{scale_label.capitalize()}-vessel minority swap: and "
                f"{len(swapped_components) - _MAX_LOGGED_SWAPS} smaller component(s)."
            )
        logger.info(
            f"{scale_label.capitalize()}-vessel minority swap: "
            f"max_size_ratio={float(swap_max_size_ratio):.3f}, "
            f"min_contact_fraction={float(swap_min_contact_fraction):.3f}; "
            f"swapped {swap_stats['swapped_to_venule_component_count']} arteriole "
            f"component(s) to venule ({swap_stats['swapped_to_venule_voxel_count']} "
            f"voxels) and {swap_stats['swapped_to_arteriole_component_count']} "
            f"venule component(s) to arteriole "
            f"({swap_stats['swapped_to_arteriole_voxel_count']} voxels); kept "
            f"{swap_stats['kept_touching_component_count']} touching smaller "
            f"component(s) below the contact fraction (highest "
            f"{float(swap_stats['max_kept_contact_fraction']):.3f})."
        )

    if mask_role == "large" and bool(remove_small_opposite_attached_components):
        from haemolynx.graph.large_vessels import (
            remove_small_opposite_attached_large_vessel_components,
        )

        (
            arteriole_mask,
            venule_mask,
            opposite_attached_cleanup_stats,
        ) = remove_small_opposite_attached_large_vessel_components(
            arteriole_mask,
            venule_mask,
            voxel_size_xyz=tuple(float(v) for v in main_voxel_size_xyz),
            max_component_volume_um3=float(
                opposite_attached_max_component_volume_um3
            ),
            max_attach_distance_microns=float(
                opposite_attached_max_distance_microns
            ),
            **low_ram,
        )
        oa_art = opposite_attached_cleanup_stats.get("arteriole") or {}
        oa_ven = opposite_attached_cleanup_stats.get("venule") or {}
        logger.info(
            "Large-vessel opposite-attached tiny-component cleanup: "
            f"max_component_volume_um3="
            f"{float(opposite_attached_max_component_volume_um3):.3f}, "
            f"max_attach_distance_microns="
            f"{float(opposite_attached_max_distance_microns):.3f}, "
            f"removed_components(arteriole="
            f"{int(oa_art.get('removed_component_count', 0))}, "
            f"venule={int(oa_ven.get('removed_component_count', 0))})."
        )

    if bool(exclude_smaller_overlapping_volumes):
        from haemolynx.graph.large_vessels import (
            exclude_smaller_overlapping_large_vessel_components,
            exclude_smaller_overlapping_small_vessel_components,
        )

        if mask_role == "large":
            arteriole_mask, venule_mask = (
                exclude_smaller_overlapping_large_vessel_components(
                    arteriole_mask,
                    venule_mask,
                    **low_ram,
                )
            )
        else:
            arteriole_mask, venule_mask = (
                exclude_smaller_overlapping_small_vessel_components(
                    arteriole_mask,
                    venule_mask,
                    **low_ram,
                )
            )

    if mask_role == "small":
        overlap_info = (
            f", {loaded_message_suffix}" if loaded_message_suffix else ""
        )
        logger.info(
            "Loaded small-vessel masks for boundary assignment: "
            f"arteriole={arteriole_mask.shape}, venule={venule_mask.shape}"
            f"{overlap_info}"
        )

    result = (
        arteriole_mask,
        venule_mask,
        arteriole_mask_voxel_size,
        venule_mask_voxel_size,
    )
    if cache_key is not None:
        _LOADED_MASKS[mask_role] = (cache_key, _pack_masks(result))
    return result


def _load_global_vessel_mask(
    mask_role: Literal["large", "small"],
    mask_path: str | Path,
    *,
    image_shape: tuple[int, ...],
    main_voxel_size_xyz: tuple[float, float, float],
    dilation_microns: float,
    min_component_volume_um3: float,
    axis_order: str,
    voxel_size_override_xyz,
    voxel_size_policy: str,
    crop: CropBox | None,
    downsample: Downsample | None = None,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> tuple[np.ndarray, None, tuple[float, float, float], None]:
    """One role's global vessel mask: the arteriole/global row's file with no
    venule mask beside it, every vessel of the role's size, arteriole or
    venule. Cut, reduced, checked and cleaned as a pair is, bar the steps that
    need two classes (the minority swap, the opposite-attached and the overlap
    cleanups); ``(mask, None, voxel size, None)``."""
    cache_key = _mask_cache_key(
        mask_role,
        mask_path,
        None,
        axis_order=axis_order,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
        image_shape=image_shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
        dilation_microns=dilation_microns,
        min_component_volume_um3=min_component_volume_um3,
        crop=crop,
        downsample=downsample,
    )
    cached = _LOADED_MASKS.get(mask_role)
    if cache_key is not None and cached is not None and cached[0] == cache_key:
        logger.info(
            f"Reusing the {mask_role}-vessel global mask loaded earlier in this "
            "process: the same file, cleaned with the same settings."
        )
        return _unpack_masks(cached[1], use_memmap=use_memmap, memmap_directory=memmap_directory)

    mask, _venule, voxel_size, _venule_voxel_size = load_large_vessel_masks(
        enabled=True,
        large_arteriole_mask_path=mask_path,
        large_venule_mask_path=None,
        axis_order=axis_order,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
        fallback_voxel_size_xyz=_source_voxel_size_xyz(main_voxel_size_xyz, downsample),
        use_memmap=use_memmap,
        memmap_directory=memmap_directory,
    )
    name = f"{mask_role}_global_mask"
    if crop is not None and not crop.is_whole_volume:
        if mask.shape != tuple(crop.source_shape_zyx):
            raise ValueError(
                f"{name} shape does not match input image shape before "
                f"crop_on_import cut it: {mask.shape} != {tuple(crop.source_shape_zyx)}"
            )
        logger.info(f"Crop on import, {mask_role}-vessel global mask: {crop.describe()}")
        mask = crop_volume(mask, crop)
    if downsample is not None and not downsample.is_identity:
        if mask.shape != tuple(downsample.source_shape_zyx):
            raise ValueError(
                f"{name} shape does not match input image shape before "
                f"segmented_image_downsample reduced it: {mask.shape} != "
                f"{tuple(downsample.source_shape_zyx)}"
            )
        mask = downsample_volume(
            mask, downsample, reduce="majority", description=f"{mask_role}-vessel global mask"
        )
        voxel_size = downsample.scale_voxel_size_xyz(voxel_size)
    if mask.shape != image_shape:
        raise ValueError(
            f"{name} shape does not match input image shape: {mask.shape} != {image_shape}"
        )
    logger.info(
        f"Loaded the {mask_role}-vessel global mask {mask.shape} (no venule mask "
        f"entered: every vessel in it is of unknown class), voxel size (x, y, z) {voxel_size}."
    )
    _assert_voxel_sizes_match_main_image(
        main_voxel_size_xyz=main_voxel_size_xyz,
        arteriole_voxel_size_xyz=tuple(float(v) for v in voxel_size),
        venule_voxel_size_xyz=None,
        mask_role=mask_role,
    )
    if dilation_microns > 0:
        from haemolynx.graph.large_vessels import dilate_binary_mask_by_microns

        mask = dilate_binary_mask_by_microns(
            mask,
            dilation_microns=dilation_microns,
            voxel_size_zyx=voxel_size_zyx_from_xyz(main_voxel_size_xyz),
            low_memory=bool(use_memmap),
            memmap_directory=memmap_directory,
        )
        logger.info(
            f"Dilated the {mask_role}-vessel global mask by {float(dilation_microns):.3f} microns."
        )
    if float(min_component_volume_um3) > 0:
        from haemolynx.graph.mask_component_volume import (
            remove_small_mask_components_by_volume,
        )

        mask, stats = remove_small_mask_components_by_volume(
            mask,
            voxel_size_xyz=tuple(float(v) for v in main_voxel_size_xyz),
            min_component_volume_um3=float(min_component_volume_um3),
            low_memory=bool(use_memmap),
            memmap_directory=memmap_directory,
        )
        logger.info(
            f"{mask_role.capitalize()}-vessel global mask component-volume filtering: "
            f"threshold={float(min_component_volume_um3):.3f} um^3, removed "
            f"{int(stats['removed_component_count'])} component(s), "
            f"{float(stats['removed_volume_um3']):.3f} um^3."
        )
    result = (np.asanyarray(mask, dtype=bool), None, voxel_size, None)
    if cache_key is not None:
        _LOADED_MASKS[mask_role] = (cache_key, _pack_masks(result))
    return result


def vessel_mask_role_is_global(
    settings: Mapping[str, object], mask_role: Literal["large", "small"]
) -> bool:
    """Whether *mask_role*'s masks are in use and are one global vessel mask:
    the arteriole/global row filled with no venule mask entered (under
    ilastik, no venule classifier), so the mask holds every vessel of that
    size, arteriole or venule, without saying which. In use is
    ``automated_vessel_assignment`` and the role's own switch both on.

    The venule setting must be there and empty: a run's settings name every
    setting, while a mapping a caller wrote by hand without it describes the
    masks as they were before a global one could be entered, a pair."""
    from haemolynx.graph.boundaries import vessel_masks_in_use

    names = VESSEL_MASK_SETTINGS[mask_role]
    if not vessel_masks_in_use(settings, mask_role):
        return False
    venule = (
        names["ilastik_venule_classifier_path"]
        if settings.get(names["use_ilastik"])
        else names["venule_mask_path"]
    )
    return venule in settings and not settings[venule]


def global_vessel_mask_roles(settings: Mapping[str, object]) -> tuple[str, ...]:
    """The mask roles (``"large"``, ``"small"``) given one global mask each."""
    return tuple(
        role for role in VESSEL_MASK_SETTINGS if vessel_mask_role_is_global(settings, role)
    )


def vessel_masks_are_global(settings: Mapping[str, object]) -> bool:
    """Whether this run classifies its vessels without arterioles and venules.

    True once any mask role is a global mask (:func:`vessel_mask_role_is_global`):
    the vessels are then of unknown class -- large in the large mask, small in
    the small one, capillary elsewhere -- rather than given branch orders,
    the masks cut nothing and choose no inlets or outlets, and a role still
    given an arteriole/venule pair is read as the one mask they make together
    (:func:`as_global_vessel_mask`). The one place this is decided: the
    loading, the Boundaries and Diameters stages and preflight all ask here.
    """
    return bool(global_vessel_mask_roles(settings))


def as_global_vessel_mask(
    arteriole_mask: np.ndarray | None,
    venule_mask: np.ndarray | None,
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray | None:
    """A role's masks as one global mask: the arteriole/global mask, with the
    venule mask added to it when there is one. *use_memmap* (the low-RAM
    option) writes the union on disk a slab at a time, not into RAM."""
    if arteriole_mask is None:
        return None if venule_mask is None else np.asanyarray(venule_mask, dtype=bool)
    if venule_mask is None:
        return np.asanyarray(arteriole_mask, dtype=bool)
    if use_memmap:
        from haemolynx.preprocessing.memmap_support import combine_by_slab

        return combine_by_slab(
            lambda a, b: a.astype(bool) | b.astype(bool), arteriole_mask, venule_mask, directory=memmap_directory
        )
    return np.asarray(arteriole_mask, dtype=bool) | np.asarray(venule_mask, dtype=bool)


#: The masks loaded last, one entry per role, so a run that reads them twice
#: -- skeletonise's thick-vessel restriction, then build_network -- loads and
#: cleans them once, and a run repeated from the panel skips it too. Keyed on
#: the files' identity and every setting the cleanup reads; held packed, one
#: bit a voxel, and handed out as fresh arrays, so nothing a caller does to
#: its copy reaches the next. Nothing is remembered or reused while either
#: file was modified within the raw-volume cache's racy window: until then a
#: same-size rewrite in the same file-clock tick (ilastik overwriting its own
#: output) keeps the mtime and size the key holds.
_LOADED_MASKS: dict[str, tuple[tuple, tuple]] = {}


def clear_vessel_mask_cache() -> None:
    """Forget the masks :func:`load_and_validate_vessel_masks` remembered."""
    _LOADED_MASKS.clear()


def _file_identity(path: str | Path) -> tuple[str, int, int] | None:
    """The file's (path, mtime, size); None -- do not cache -- when it is
    missing, or so recently modified that a rewrite might keep both."""
    try:
        resolved = Path(path).resolve()
        status = resolved.stat()
    except OSError:
        return None
    if _modified_too_recently(status):
        return None
    return (str(resolved), int(status.st_mtime_ns), int(status.st_size))


def _mask_cache_key(mask_role: str, arteriole_path, venule_path, **settings) -> tuple | None:
    """*venule_path* None is a global mask: the arteriole/global file alone."""
    arteriole = _file_identity(arteriole_path)
    venule = None if venule_path is None else _file_identity(venule_path)
    if arteriole is None or (venule_path is not None and venule is None):
        return None
    frozen = []
    for name in sorted(settings):
        value = settings[name]
        if isinstance(value, (list, tuple, np.ndarray)):
            value = tuple(float(v) for v in value)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            value = value
        else:
            value = repr(value)
        frozen.append((name, value))
    return (mask_role, arteriole, venule, tuple(frozen))


def _pack_masks(result: tuple) -> tuple:
    arteriole, venule, arteriole_voxel_size, venule_voxel_size = result
    return (
        np.packbits(np.asarray(arteriole, dtype=bool).ravel()),
        None if venule is None else np.packbits(np.asarray(venule, dtype=bool).ravel()),
        tuple(arteriole.shape),
        arteriole_voxel_size,
        venule_voxel_size,
    )


def _unpack_masks(packed: tuple, *, use_memmap: bool = False, memmap_directory=None) -> tuple:
    """The masks :func:`_pack_masks` packed, fresh arrays: on disk, unpacked
    a slab of planes at a time, under the low-RAM option (*use_memmap*)."""
    arteriole_bits, venule_bits, shape, arteriole_voxel_size, venule_voxel_size = packed
    count = int(np.prod(shape))

    def unpack(bits):
        if bits is None:
            return None
        if use_memmap:
            from haemolynx.preprocessing.memmap_support import unpack_bits_by_slab

            return unpack_bits_by_slab(bits, shape, directory=memmap_directory)
        return np.unpackbits(bits)[:count].astype(bool).reshape(shape)

    return (unpack(arteriole_bits), unpack(venule_bits), arteriole_voxel_size, venule_voxel_size)
