#!/usr/bin/env python3
"""Interactive setup wizard for resistance pipeline runs."""
from pathlib import Path


def _prompt_yes_no(question: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _prompt_optional_text(question: str) -> str | None:
    raw = input(f"{question} (leave empty to keep current): ").strip()
    return raw or None


def run_interactive_setup_wizard(
    default_preset: str,
    available_presets: list[str],
) -> dict[str, object]:
    """Collect optional run overrides from interactive CLI prompts."""
    print("\n=== Interactive Pipeline Setup Wizard ===")
    print("Press Enter to accept defaults shown in prompts.\n")

    print("Available presets:")
    for idx, name in enumerate(available_presets, start=1):
        marker = " (default)" if name == default_preset else ""
        print(f"  {idx}. {name}{marker}")
    chosen = input(f"Select preset by name or number [{default_preset}]: ").strip()
    preset_name = default_preset
    if chosen:
        if chosen.isdigit():
            selection = int(chosen)
            if 1 <= selection <= len(available_presets):
                preset_name = available_presets[selection - 1]
            else:
                print("Invalid preset number; keeping default preset.")
        elif chosen in available_presets:
            preset_name = chosen
        else:
            print("Unknown preset name; keeping default preset.")

    pipeline_overrides: dict[str, object] = {}
    settings_overrides: dict[str, object] = {}

    image_path = _prompt_optional_text("Input image path")
    if image_path is not None:
        pipeline_overrides["image_path"] = Path(image_path)

    use_large_masks = _prompt_yes_no("Use large vessel masks?", default=False)
    settings_overrides["USE_LARGE_VESSEL_MASKS"] = use_large_masks
    if use_large_masks:
        use_ilastik_large = _prompt_yes_no(
            "Use ilastik for large vessel masks?",
            default=False,
        )
        settings_overrides["USE_ILASTIK_LARGE_VESSEL_SEGMENTATION"] = use_ilastik_large

    use_small_masks = _prompt_yes_no(
        "Use small vessel masks for boundary assignment?",
        default=False,
    )
    settings_overrides["USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT"] = use_small_masks
    if use_small_masks:
        use_ilastik_small = _prompt_yes_no(
            "Use ilastik for small vessel masks?",
            default=False,
        )
        settings_overrides["USE_ILASTIK_SMALL_VESSEL_SEGMENTATION"] = use_ilastik_small

    automated_assignment = _prompt_yes_no(
        "Use automated vessel assignment?",
        default=use_large_masks,
    )
    settings_overrides["AUTOMATED_VESSEL_ASSIGNMENT"] = automated_assignment

    run_haemodynamics = _prompt_yes_no("Run haemodynamics?", default=True)
    settings_overrides["RUN_HAEMODYNAMICS"] = run_haemodynamics

    if run_haemodynamics:
        use_fwhm = _prompt_yes_no(
            "Use automated FWHM edge diameters?",
            default=False,
        )
        settings_overrides["USE_FWHM_EDGE_DIAMETERS"] = use_fwhm
        do_equiv_resistance = _prompt_yes_no(
            "Compute equivalent resistance (two-point)?",
            default=False,
        )
        settings_overrides["DO_EQUIV_RESISTANCE_CALCULATION"] = do_equiv_resistance

    run_statistics = _prompt_yes_no("Run vessel statistics?", default=True)
    settings_overrides["STATISTICS"] = run_statistics

    run_distance_3d = _prompt_yes_no(
        "Run 3D distance-to-cell-mask measurement?",
        default=False,
    )
    settings_overrides["MEASUREMENT_3D_TO_CELL_MASK"] = run_distance_3d
    if run_distance_3d:
        cell_mask_path = _prompt_optional_text("Cell mask path")
        if cell_mask_path is not None:
            settings_overrides["CELL_MASK_PATH"] = Path(cell_mask_path)

    print("Wizard configuration captured.\n")
    return {
        "preset_name": preset_name,
        "settings_overrides": settings_overrides,
        "pipeline_overrides": pipeline_overrides,
    }
