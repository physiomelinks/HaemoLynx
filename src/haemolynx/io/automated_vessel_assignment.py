"""Helpers for optional automated large-vessel mask loading."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .axis_order import CANONICAL_AXIS_ORDER, voxel_size_zyx_from_xyz
from .ilastik import run_ilastik_headless_segmentation
from .load import (
    _to_binary_volume_for_skeletonization,
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    resolve_image_path_with_optional_zip,
)
from .voxel_validation import resolve_voxel_size_xyz

logger = logging.getLogger(__name__)


def _load_mask_image(
    mask_path: Path,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, object]]:
    """Load a vessel mask as a boolean volume in canonical ``(z, y, x)`` order.

    Must binarise with the shared mask loader, not a raw intensity read:
    ``astype(bool)`` / ``> 0`` on a ``1/2``-encoded mask fills the whole
    volume, and opposite-attached volume cleanup then treats every venule
    voxel as attached to that solid arteriole and can wipe the venule mask.

    Returns the mask, metadata voxel size ``(x, y, z)``, and the metadata
    status dict so callers can apply the same override policy as the main
    image.
    """
    path = Path(mask_path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z, voxel_meta_status = load_3d_tif_with_voxel_size(
            str(path),
            axis_order=axis_order,
        )
    elif suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z, voxel_meta_status = load_3d_h5_with_voxel_size(
            str(path),
            axis_order=axis_order,
        )
    else:
        raise ValueError(
            f"Unsupported mask format '{suffix}'. Expected .tif, .tiff, or .h5."
        )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {image.shape}.")
    mask = _to_binary_volume_for_skeletonization(image)
    return mask, (float(voxel_x), float(voxel_y), float(voxel_z)), voxel_meta_status


def _resolve_mask_voxel_size_xyz(
    metadata_voxel_size_xyz: tuple[float, float, float],
    metadata_status: dict[str, object],
    *,
    voxel_size_override_xyz,
    voxel_size_policy: str,
) -> tuple[float, float, float]:
    """Apply the run's voxel-size override policy to one mask's metadata."""
    resolved, _source = resolve_voxel_size_xyz(
        metadata_voxel_size_xyz=metadata_voxel_size_xyz,
        metadata_status=metadata_status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
    )
    return resolved


def load_large_vessel_masks(
    enabled: bool,
    large_arteriole_mask_path: str | Path | None = None,
    large_venule_mask_path: str | Path | None = None,
    *,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Load optional large arteriole/venule masks with strict pairing rules.

    When enabled is False, no mask paths are allowed and (None, None) is returned.
    When enabled is True, both mask paths are required and both are loaded.

    Voxel sizes are resolved with the same ``voxel_size_override_xyz`` /
    ``voxel_size_policy`` rules as the main image, so a manual unit override
    applies to the masks instead of leaving them on missing-metadata ``(1,1,1)``.
    """
    has_arteriole = large_arteriole_mask_path is not None
    has_venule = large_venule_mask_path is not None

    if not enabled:
        if has_arteriole or has_venule:
            raise ValueError(
                "Large-vessel masks are disabled. Set enabled=True to provide "
                "large_arteriole_mask_path and large_venule_mask_path."
            )
        return None, None, None, None

    if has_arteriole != has_venule:
        raise ValueError(
            "Either provide both large_arteriole_mask_path and "
            "large_venule_mask_path, or provide neither."
        )
    if not has_arteriole:
        raise ValueError(
            "Large-vessel masks are enabled, but mask paths are missing. "
            "Provide both large_arteriole_mask_path and large_venule_mask_path."
        )

    arteriole_path = resolve_image_path_with_optional_zip(Path(large_arteriole_mask_path))
    venule_path = resolve_image_path_with_optional_zip(Path(large_venule_mask_path))
    large_arteriole_mask, arteriole_metadata_xyz, arteriole_status = _load_mask_image(
        arteriole_path, axis_order=axis_order
    )
    large_venule_mask, venule_metadata_xyz, venule_status = _load_mask_image(
        venule_path, axis_order=axis_order
    )
    large_arteriole_voxel_size = _resolve_mask_voxel_size_xyz(
        arteriole_metadata_xyz,
        arteriole_status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
    )
    large_venule_voxel_size = _resolve_mask_voxel_size_xyz(
        venule_metadata_xyz,
        venule_status,
        voxel_size_override_xyz=voxel_size_override_xyz,
        voxel_size_policy=voxel_size_policy,
    )
    return (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_voxel_size,
        large_venule_voxel_size,
    )


def _vessel_mask_config(mask_role: Literal["large", "small"]) -> dict[str, str]:
    if mask_role == "large":
        return {
            "enabled_flag": "use_large_vessel_masks",
            "ilastik_flag": "use_ilastik_large_vessel_segmentation",
            "ilastik_arteriole_image_param": "ilastik_unsegmented_arteriole_image_path",
            "ilastik_venule_image_param": "ilastik_unsegmented_venule_image_path",
            "ilastik_arteriole_classifier_param": "ilastik_arteriole_classifier_path",
            "ilastik_venule_classifier_param": "ilastik_venule_classifier_path",
            "arteriole_mask_attr": "large_arteriole_mask",
            "venule_mask_attr": "large_venule_mask",
            "disabled_message": (
                "Large-vessel masks disabled; skipping arteriole/venule mask loading."
            ),
            "boundary_disabled_message": None,
        }
    return {
        "enabled_flag": "use_small_vessel_masks_for_boundary_assignment",
        "ilastik_flag": "use_ilastik_small_vessel_segmentation",
        "ilastik_arteriole_image_param": "ilastik_unsegmented_small_arteriole_image_path",
        "ilastik_venule_image_param": "ilastik_unsegmented_small_venule_image_path",
        "ilastik_arteriole_classifier_param": "ilastik_small_arteriole_classifier_path",
        "ilastik_venule_classifier_param": "ilastik_small_venule_classifier_path",
        "arteriole_mask_attr": "small_arteriole_mask",
        "venule_mask_attr": "small_venule_mask",
        "disabled_message": None,
        "boundary_disabled_message": (
            "Small-vessel-mask boundary assignment disabled; "
            "manual arteriole/venule boundary-node selection remains available."
        ),
    }


def _assert_voxel_sizes_match_main_image(
    *,
    main_voxel_size_xyz: tuple[float, float, float],
    arteriole_voxel_size_xyz: tuple[float, float, float],
    venule_voxel_size_xyz: tuple[float, float, float],
    mask_role: Literal["large", "small"],
) -> None:
    arteriole_label = "arteriole" if mask_role == "large" else "small_arteriole"
    venule_label = "venule" if mask_role == "large" else "small_venule"
    vessel_phrase = "large-vessel masks" if mask_role == "large" else "small-vessel masks"
    if (
        np.allclose(main_voxel_size_xyz, arteriole_voxel_size_xyz, rtol=0.0, atol=0.0)
        and np.allclose(main_voxel_size_xyz, venule_voxel_size_xyz, rtol=0.0, atol=0.0)
        and np.allclose(
            arteriole_voxel_size_xyz, venule_voxel_size_xyz, rtol=0.0, atol=0.0
        )
    ):
        if mask_role == "large":
            logger.info(
                "Voxel-size check passed. Arteriole and venule masks are aligned "
                "to the same physical voxel units as the main image."
            )
        return

    error_message = (
        f"Voxel-size mismatch detected across input image and {vessel_phrase}. "
        f"main={main_voxel_size_xyz}, "
        f"{arteriole_label}={arteriole_voxel_size_xyz}, "
        f"{venule_label}={venule_voxel_size_xyz}. "
        "All must match exactly in x, y, and z."
    )
    if mask_role == "large":
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
    },
}

#: Settings both roles share, named identically in the config.
_SHARED_VESSEL_MASK_SETTINGS = (
    "ilastik_output_dir",
    "ilastik_output_suffix",
    "ilastik_executable",
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
    dilation_microns: float = 0.0,
    min_component_volume_um3: float = 0.0,
    remove_small_opposite_attached_components: bool = False,
    opposite_attached_max_component_volume_um3: float = 250.0,
    opposite_attached_max_distance_microns: float = 3.0,
    exclude_smaller_overlapping_volumes: bool = False,
    loaded_message_suffix: str | None = None,
    axis_order: str = CANONICAL_AXIS_ORDER,
    voxel_size_override_xyz=None,
    voxel_size_policy: str = "auto",
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Resolve, optionally segment with ilastik, load, and validate arteriole/venule masks."""
    config = _vessel_mask_config(mask_role)
    scale_label = mask_role

    if use_ilastik and not enabled:
        raise ValueError(
            f"{config['ilastik_flag']}=True requires {config['enabled_flag']}=True."
        )

    effective_arteriole_mask_path = arteriole_mask_path
    effective_venule_mask_path = venule_mask_path
    if not enabled:
        effective_arteriole_mask_path = None
        effective_venule_mask_path = None

    if enabled and use_ilastik:
        for param_name, value in (
            (config["ilastik_arteriole_image_param"], ilastik_unsegmented_arteriole_path),
            (config["ilastik_venule_image_param"], ilastik_unsegmented_venule_path),
            (config["ilastik_arteriole_classifier_param"], ilastik_arteriole_classifier_path),
            (config["ilastik_venule_classifier_param"], ilastik_venule_classifier_path),
        ):
            if value is None:
                raise ValueError(
                    f"{param_name} must be set when {config['ilastik_flag']}=True."
                )

        output_dir = Path(ilastik_output_dir)
        unsegmented_arteriole_image_path = resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_arteriole_path)
        )
        unsegmented_venule_image_path = resolve_image_path_with_optional_zip(
            Path(ilastik_unsegmented_venule_path)
        )
        segmented_arteriole_path = output_dir / (
            f"{unsegmented_arteriole_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        segmented_venule_path = output_dir / (
            f"{unsegmented_venule_image_path.stem}_segmented{ilastik_output_suffix}"
        )

        logger.info(
            f"Running ilastik segmentation for {scale_label} arteriole image: "
            f"{unsegmented_arteriole_image_path}"
        )
        effective_arteriole_mask_path = run_ilastik_headless_segmentation(
            input_image_path=unsegmented_arteriole_image_path,
            classifier_path=Path(ilastik_arteriole_classifier_path),
            output_path=segmented_arteriole_path,
            ilastik_executable=ilastik_executable,
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
        )
        logger.info(
            f"Using ilastik-segmented {scale_label}-vessel masks: "
            f"arteriole={effective_arteriole_mask_path}, "
            f"venule={effective_venule_mask_path}"
        )

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
    )

    if arteriole_mask is None or venule_mask is None:
        disabled_message = config["boundary_disabled_message"] or config["disabled_message"]
        logger.info(disabled_message)
        return None, None, None, None

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
    if mask_role == "large":
        logger.info(
            f"Loaded {scale_label}-vessel masks: "
            f"arteriole={arteriole_mask.shape}, venule={venule_mask.shape}"
        )
        logger.info(
            "Large-vessel mask voxel sizes (x, y, z): "
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
                )
            )
        else:
            arteriole_mask, venule_mask = (
                exclude_smaller_overlapping_small_vessel_components(
                    arteriole_mask,
                    venule_mask,
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

    return (
        arteriole_mask,
        venule_mask,
        arteriole_mask_voxel_size,
        venule_mask_voxel_size,
    )
