"""Helpers for optional automated large-vessel mask loading."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .ilastik import run_ilastik_headless_segmentation
from .load import (
    load_3d_h5_with_voxel_size,
    load_3d_tif_with_voxel_size,
    resolve_image_path_with_optional_zip,
)


def _load_mask_image(mask_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a mask image and return (image, voxel_size_xyz)."""
    suffix = mask_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_tif_with_voxel_size(
            str(mask_path)
        )
        return image, (float(voxel_x), float(voxel_y), float(voxel_z))
    if suffix == ".h5":
        image, voxel_x, voxel_y, voxel_z, _voxel_meta_status = load_3d_h5_with_voxel_size(
            str(mask_path)
        )
        return image, (float(voxel_x), float(voxel_y), float(voxel_z))
    raise ValueError(
        f"Unsupported mask format '{suffix}' for {mask_path}. "
        "Supported formats: .tif, .tiff, .h5"
    )


def load_large_vessel_masks(
    enabled: bool,
    large_arteriole_mask_path: str | Path | None = None,
    large_venule_mask_path: str | Path | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    """Load optional large arteriole/venule masks with strict pairing rules.

    When enabled is False, no mask paths are allowed and (None, None) is returned.
    When enabled is True, both mask paths are required and both are loaded.
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
    large_arteriole_mask, large_arteriole_voxel_size = _load_mask_image(arteriole_path)
    large_venule_mask, large_venule_voxel_size = _load_mask_image(venule_path)
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
            print(
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
        print(error_message)
    raise ValueError(error_message)


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
    loaded_message_suffix: str | None = None,
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

        print(
            f"Running ilastik segmentation for {scale_label} arteriole image: "
            f"{unsegmented_arteriole_image_path}"
        )
        effective_arteriole_mask_path = run_ilastik_headless_segmentation(
            input_image_path=unsegmented_arteriole_image_path,
            classifier_path=Path(ilastik_arteriole_classifier_path),
            output_path=segmented_arteriole_path,
            ilastik_executable=ilastik_executable,
        )
        print(
            f"Running ilastik segmentation for {scale_label} venule image: "
            f"{unsegmented_venule_image_path}"
        )
        effective_venule_mask_path = run_ilastik_headless_segmentation(
            input_image_path=unsegmented_venule_image_path,
            classifier_path=Path(ilastik_venule_classifier_path),
            output_path=segmented_venule_path,
            ilastik_executable=ilastik_executable,
        )
        print(
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
    )

    if arteriole_mask is None or venule_mask is None:
        disabled_message = config["boundary_disabled_message"] or config["disabled_message"]
        print(disabled_message)
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
        print(
            f"Loaded {scale_label}-vessel masks: "
            f"arteriole={arteriole_mask.shape}, venule={venule_mask.shape}"
        )
        print(
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
        from ImageLynx.graph.large_vessels import dilate_large_vessel_masks_by_microns

        arteriole_mask, venule_mask = dilate_large_vessel_masks_by_microns(
            large_arteriole_mask=arteriole_mask,
            large_venule_mask=venule_mask,
            dilation_microns=dilation_microns,
            voxel_size_xyz=main_voxel_size_xyz,
        )
        print(
            f"Dilated {scale_label}-vessel masks by {float(dilation_microns):.3f} microns."
        )

    if mask_role == "small":
        overlap_info = (
            f", {loaded_message_suffix}" if loaded_message_suffix else ""
        )
        print(
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
