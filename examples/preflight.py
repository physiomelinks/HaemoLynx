#!/usr/bin/env python3
"""Preflight validation for resistance pipeline runs."""
from pathlib import Path
import shutil


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _path_exists_or_zip(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return False, "not provided"
    candidates = [
        path,
        path.with_suffix(f"{path.suffix}.zip"),
        path.with_suffix(".zip"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return True, str(candidate)
    checked = ", ".join(str(p) for p in candidates)
    return False, f"checked: {checked}"


def run_preflight_checklist(pipeline_kwargs: dict[str, object]) -> dict[str, object]:
    """Validate run inputs/toggles and print a checklist before execution."""
    checks: list[tuple[str, str, str]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def ok(label: str, detail: str) -> None:
        checks.append(("OK", label, detail))

    def warn(label: str, detail: str) -> None:
        checks.append(("WARN", label, detail))
        warnings.append(f"{label}: {detail}")

    def fail(label: str, detail: str, fix: str) -> None:
        checks.append(("ERROR", label, detail))
        errors.append(f"{label}: {detail}. Fix: {fix}")

    # Short aliases
    use_ilastik_segmentation = bool(pipeline_kwargs.get("use_ilastik_segmentation"))
    use_large_vessel_masks = bool(pipeline_kwargs.get("use_large_vessel_masks"))
    use_ilastik_large_vessel_segmentation = bool(
        pipeline_kwargs.get("use_ilastik_large_vessel_segmentation")
    )
    use_small_vessel_masks_for_boundary_assignment = bool(
        pipeline_kwargs.get("use_small_vessel_masks_for_boundary_assignment")
    )
    use_ilastik_small_vessel_segmentation = bool(
        pipeline_kwargs.get("use_ilastik_small_vessel_segmentation")
    )
    run_haemodynamics = bool(pipeline_kwargs.get("run_haemodynamics", True))
    do_skeletonize = bool(pipeline_kwargs.get("do_skeletonize", True))
    do_graph_building = bool(pipeline_kwargs.get("do_graph_building", True))
    do_equiv_resistance_calculation = bool(
        pipeline_kwargs.get("do_equiv_resistance_calculation")
    )
    use_fwhm_edge_diameters = bool(pipeline_kwargs.get("use_fwhm_edge_diameters"))
    do_pericyte_constriction = bool(pipeline_kwargs.get("do_pericyte_constriction"))
    use_pericyte_mask_constriction = bool(
        pipeline_kwargs.get("use_pericyte_mask_constriction")
    )
    measurement_3d_to_cell_mask = bool(
        pipeline_kwargs.get("measurement_3d_to_cell_mask")
    )
    strict_branch_order_assignment = bool(
        pipeline_kwargs.get("strict_branch_order_assignment")
    )
    automated_vessel_assignment = bool(
        pipeline_kwargs.get("automated_vessel_assignment")
    )

    # Main image / ilastik source checks
    image_path = _as_path(pipeline_kwargs.get("image_path"))
    unsegmented_image_path = _as_path(
        pipeline_kwargs.get("ilastik_unsegmented_image_path")
    )
    ilastik_classifier_path = _as_path(pipeline_kwargs.get("ilastik_classifier_path"))
    if use_ilastik_segmentation:
        exists, detail = _path_exists_or_zip(unsegmented_image_path)
        if exists:
            ok("Main image (ilastik input)", detail)
        else:
            fail(
                "Main image (ilastik input)",
                detail,
                "Set ILASTIK_UNSEGMENTED_IMAGE_PATH / --ilastik-unsegmented-image-path to an existing file.",
            )
        if ilastik_classifier_path and ilastik_classifier_path.exists():
            ok("Main ilastik classifier", str(ilastik_classifier_path))
        else:
            fail(
                "Main ilastik classifier",
                str(ilastik_classifier_path),
                "Set ILASTIK_CLASSIFIER_PATH / --ilastik-classifier-path to an existing .ilp file.",
            )
    else:
        exists, detail = _path_exists_or_zip(image_path)
        if exists:
            ok("Main input image", detail)
        else:
            fail(
                "Main input image",
                detail,
                "Set INPUT_PATH / --image-path to an existing segmented image.",
            )

    # Input axis order must be a permutation of xyz; it decides which axis is z.
    axis_order = pipeline_kwargs.get("axis_order", "zyx")
    normalized_axis_order = str(axis_order).strip().lower()
    if sorted(normalized_axis_order) == ["x", "y", "z"]:
        if normalized_axis_order == "zyx":
            ok("Input axis order", "zyx (canonical; no transpose)")
        else:
            ok(
                "Input axis order",
                f"{normalized_axis_order} (transposed to canonical zyx on load)",
            )
    else:
        fail(
            "Input axis order",
            str(axis_order),
            "Set IMAGE_AXIS_ORDER / --axis-order to a permutation of 'xyz', e.g. 'zyx'.",
        )

    # ilastik executable check when any ilastik mode is enabled
    ilastik_required = bool(
        use_ilastik_segmentation
        or use_ilastik_large_vessel_segmentation
        or use_ilastik_small_vessel_segmentation
    )
    ilastik_executable = pipeline_kwargs.get("ilastik_executable")
    if ilastik_required:
        exec_text = "" if ilastik_executable is None else str(ilastik_executable).strip()
        exec_path = _as_path(exec_text) if exec_text else None
        exists_as_path = bool(exec_path and exec_path.exists())
        found_on_path = bool(exec_text and shutil.which(exec_text))
        if exists_as_path or found_on_path:
            source = str(exec_path) if exists_as_path else str(shutil.which(exec_text))
            ok("Ilastik executable", source)
        else:
            fail(
                "Ilastik executable",
                str(ilastik_executable),
                "Set ILASTIK_EXECUTABLE to a valid executable path or command available on PATH.",
            )

    # Large-vessel mask checks
    if use_ilastik_large_vessel_segmentation and not use_large_vessel_masks:
        fail(
            "Large-vessel ilastik toggle",
            "use_ilastik_large_vessel_segmentation=True while use_large_vessel_masks=False",
            "Enable use_large_vessel_masks or disable ilastik large-vessel segmentation.",
        )
    if use_large_vessel_masks:
        if use_ilastik_large_vessel_segmentation:
            for label, key in (
                ("Large arteriole ilastik input", "ilastik_unsegmented_arteriole_image_path"),
                ("Large venule ilastik input", "ilastik_unsegmented_venule_image_path"),
                ("Large arteriole ilastik classifier", "ilastik_arteriole_classifier_path"),
                ("Large venule ilastik classifier", "ilastik_venule_classifier_path"),
            ):
                value = _as_path(pipeline_kwargs.get(key))
                if key.endswith("_classifier_path"):
                    if value and value.exists():
                        ok(label, str(value))
                    else:
                        fail(label, str(value), f"Provide a valid path for {key}.")
                else:
                    exists, detail = _path_exists_or_zip(value)
                    if exists:
                        ok(label, detail)
                    else:
                        fail(label, detail, f"Provide a valid path for {key}.")
        else:
            for label, key in (
                ("Large arteriole mask", "large_arteriole_mask_path"),
                ("Large venule mask", "large_venule_mask_path"),
            ):
                exists, detail = _path_exists_or_zip(_as_path(pipeline_kwargs.get(key)))
                if exists:
                    ok(label, detail)
                else:
                    fail(label, detail, f"Provide a valid path for {key}.")

    # Small-vessel mask checks
    if use_ilastik_small_vessel_segmentation and not use_small_vessel_masks_for_boundary_assignment:
        fail(
            "Small-vessel ilastik toggle",
            "use_ilastik_small_vessel_segmentation=True while use_small_vessel_masks_for_boundary_assignment=False",
            "Enable small-vessel boundary assignment or disable ilastik small-vessel segmentation.",
        )
    if use_small_vessel_masks_for_boundary_assignment:
        if use_ilastik_small_vessel_segmentation:
            for label, key in (
                ("Small arteriole ilastik input", "ilastik_unsegmented_small_arteriole_image_path"),
                ("Small venule ilastik input", "ilastik_unsegmented_small_venule_image_path"),
                ("Small arteriole ilastik classifier", "ilastik_small_arteriole_classifier_path"),
                ("Small venule ilastik classifier", "ilastik_small_venule_classifier_path"),
            ):
                value = _as_path(pipeline_kwargs.get(key))
                if key.endswith("_classifier_path"):
                    if value and value.exists():
                        ok(label, str(value))
                    else:
                        fail(label, str(value), f"Provide a valid path for {key}.")
                else:
                    exists, detail = _path_exists_or_zip(value)
                    if exists:
                        ok(label, detail)
                    else:
                        fail(label, detail, f"Provide a valid path for {key}.")
        else:
            for label, key in (
                ("Small arteriole mask", "small_arteriole_mask_path"),
                ("Small venule mask", "small_venule_mask_path"),
            ):
                exists, detail = _path_exists_or_zip(_as_path(pipeline_kwargs.get(key)))
                if exists:
                    ok(label, detail)
                else:
                    fail(label, detail, f"Provide a valid path for {key}.")

    # Key dependent inputs
    if use_fwhm_edge_diameters and run_haemodynamics:
        exists, detail = _path_exists_or_zip(_as_path(pipeline_kwargs.get("fwhm_raw_tiff_path")))
        if exists:
            ok("FWHM raw TIFF", detail)
        else:
            fail(
                "FWHM raw TIFF",
                detail,
                "Set FWHM_RAW_TIFF_PATH / --fwhm-raw-tiff to an existing raw image.",
            )
    if use_fwhm_edge_diameters and not run_haemodynamics:
        warn(
            "FWHM with haemodynamics disabled",
            "USE_FWHM_EDGE_DIAMETERS is enabled but run_haemodynamics=False; FWHM stage will be skipped.",
        )

    if use_pericyte_mask_constriction:
        if not do_pericyte_constriction:
            warn(
                "Pericyte mask mode inactive",
                "use_pericyte_mask_constriction=True but do_pericyte_constriction=False; pericyte mask will not be used.",
            )
        else:
            exists, detail = _path_exists_or_zip(_as_path(pipeline_kwargs.get("pericyte_mask_path")))
            if exists:
                ok("Pericyte mask", detail)
            else:
                fail(
                    "Pericyte mask",
                    detail,
                    "Set PERICYTE_MASK_PATH / --pericyte-mask-path to an existing mask.",
                )

    if measurement_3d_to_cell_mask:
        exists, detail = _path_exists_or_zip(_as_path(pipeline_kwargs.get("cell_mask_path")))
        if exists:
            ok("3D distance cell mask", detail)
        else:
            fail(
                "3D distance cell mask",
                detail,
                "Set CELL_MASK_PATH / --cell-mask-path to an existing mask.",
            )

    if do_equiv_resistance_calculation and not run_haemodynamics:
        fail(
            "Equivalent resistance toggle",
            "do_equiv_resistance_calculation=True while run_haemodynamics=False",
            "Disable equivalent resistance or enable haemodynamics.",
        )

    if strict_branch_order_assignment and not (
        automated_vessel_assignment or use_small_vessel_masks_for_boundary_assignment
    ):
        fail(
            "Strict branch-order assignment",
            "strict_branch_order_assignment=True without automated/mask boundary assignment; this may fail if boundary nodes are missing.",
            "Enable AUTOMATED_VESSEL_ASSIGNMENT or USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT, or disable strict mode.",
        )

    # Enumerated mode checks
    final_render_mode = str(pipeline_kwargs.get("final_render_mode"))
    if final_render_mode in {"2d", "3d"}:
        ok("Final render mode", final_render_mode)
    else:
        fail(
            "Final render mode",
            final_render_mode,
            "Use final_render_mode='2d' or '3d'.",
        )

    ide_plot_mode = str(pipeline_kwargs.get("ide_plot_mode"))
    if ide_plot_mode in {"all", "final_only", "none"}:
        ok("IDE plot mode", ide_plot_mode)
    else:
        fail(
            "IDE plot mode",
            ide_plot_mode,
            "Use ide_plot_mode='all', 'final_only', or 'none'.",
        )

    statistics_mode = str(pipeline_kwargs.get("statistics_mode"))
    if statistics_mode in {"fast", "full"}:
        ok("Statistics mode", statistics_mode)
    else:
        fail(
            "Statistics mode",
            statistics_mode,
            "Use statistics_mode='fast' or 'full'.",
        )

    # Stage reuse artifact checks when stages are disabled.
    vtk_output_prefix = _as_path(pipeline_kwargs.get("vtk_output_prefix"))
    if vtk_output_prefix is not None:
        output_dir = vtk_output_prefix.parent
        if use_ilastik_segmentation:
            ilastik_output_dir = _as_path(pipeline_kwargs.get("ilastik_output_dir"))
            unseg = _as_path(pipeline_kwargs.get("ilastik_unsegmented_image_path"))
            suffix = str(pipeline_kwargs.get("ilastik_output_suffix", ".tif"))
            if ilastik_output_dir is not None and unseg is not None:
                image_stem = f"{unseg.stem}_segmented"
            else:
                image_stem = image_path.stem if image_path is not None else "input"
        else:
            image_stem = image_path.stem if image_path is not None else "input"
        expected_skeleton = output_dir / f"{image_stem}_skeleton.npy"
        expected_graph = output_dir / f"{image_stem}_graph.pkl"
        if not do_skeletonize:
            if expected_skeleton.exists():
                ok("Cached skeleton artifact", str(expected_skeleton))
            else:
                fail(
                    "Cached skeleton artifact",
                    str(expected_skeleton),
                    "Enable do_skeletonize or provide the cached skeleton file.",
                )
        if not do_graph_building:
            if expected_graph.exists():
                ok("Cached graph artifact", str(expected_graph))
            else:
                fail(
                    "Cached graph artifact",
                    str(expected_graph),
                    "Enable do_graph_building or provide the cached graph file.",
                )

    # Print checklist
    print("\n=== Preflight Checklist ===")
    for status, label, detail in checks:
        print(f"[{status}] {label}: {detail}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
    if errors:
        print(f"Errors: {len(errors)}")
        print("Preflight failed. Fix the following issues before running:")
        for idx, message in enumerate(errors, start=1):
            print(f"  {idx}. {message}")
    else:
        print("Preflight passed.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
