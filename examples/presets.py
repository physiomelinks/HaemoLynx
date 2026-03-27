#!/usr/bin/env python3
"""Preset definitions."""
import ast
import copy
import importlib.util
from pathlib import Path


def _get_builtin_preset_definitions(root_dir: Path) -> dict[str, dict[str, object]]:
    """Return built-in named preset profiles and their setting overrides."""
    return {
        "default": {
            "description": "Current baseline behavior from this settings file.",
            "overrides": {},
        },
        "quick_debug": {
            "description": (
                "Fast iteration mode: disables heavy outputs/analysis and keeps logs concise."
            ),
            "overrides": {
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": False,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "STATISTICS": False,
                "MEASUREMENT_3D_TO_CELL_MASK": False,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_FWHM_EDGE_DIAMETERS": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": False,
            },
        },
        "publication": {
            "description": (
                "High-detail reporting mode: writes full stats/artifacts with non-interactive rendering."
            ),
            "overrides": {
                "VERBOSE_LOGGING": False,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
            },
        },
        "full_fwhm": {
            "description": (
                "Enable full FWHM diameter workflow with detailed outputs for raw-image-aligned analysis."
            ),
            "overrides": {
                "USE_FWHM_EDGE_DIAMETERS": True,
                "FWHM_RAW_TIFF_PATH": root_dir
                / "examples"
                / "images"
                / "Nerve_capillaries.tif",
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
            },
        },
        "all_automated": {
            "description": (
                "Fully automated profile: large/small vessel masks, pericyte-mask constriction, and FWHM diameters."
            ),
            "overrides": {
                # Automated boundary assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Automated diameter measurement
                "USE_FWHM_EDGE_DIAMETERS": True,
                "FWHM_RAW_TIFF_PATH": root_dir
                / "examples"
                / "images"
                / "Nerve_capillaries.tif",
                # Pericyte mask-driven constriction
                "DO_PERICYTE_CONSTRUCTION": True,
                "USE_PERICYTE_MASK_CONSTRICTION": True,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                # Keep this as None by default; users should provide with --set PERICYTE_MASK_PATH=...
                # if they are not using the default image naming/location workflow.
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "automated_assignment": {
            "description": (
                "Automated vessel assignment profile using pre-segmented large and small vessel masks (no ilastik)."
            ),
            "overrides": {
                # Automated input/output node assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Keep automated diameter/pericyte workflows disabled
                "USE_FWHM_EDGE_DIAMETERS": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "automated_assignment_ilastik": {
            "description": (
                "Automated vessel assignment profile using ilastik segmentation for large and small vessel masks."
            ),
            "overrides": {
                # Automated input/output node assignment from large-vessel masks
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": True,
                # Automated arteriole/venule boundary inference from small-vessel masks
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": True,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                # Keep automated diameter/pericyte workflows disabled
                "USE_FWHM_EDGE_DIAMETERS": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "PERICYTE_MASK_PATH": None,
                # Output/reporting defaults
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": True,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "statistics_only": {
            "description": (
                "Skeletonize/build graph + assign vessels + compute statistics with haemodynamics enabled but no equivalent-resistance solve."
            ),
            "overrides": {
                "DO_SKELETONIZE": True,
                "DO_GRAPH_BUILDING": True,
                "RUN_HAEMODYNAMICS": True,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "DO_PERICYTE_CONSTRUCTION": False,
                "USE_PERICYTE_MASK_CONSTRICTION": False,
                "USE_PROBABILISTIC_PERICYTE_CONSTRICTION": False,
                "RUN_PERICYTE_RESISTANCE_COMPARISON": False,
                "USE_FWHM_EDGE_DIAMETERS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "STATISTICS": True,
                "STATISTICS_MODE": "full",
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "FINAL_RENDER_MODE": "3d",
            },
        },
        "distance_to_mask_3d": {
            "description": (
                "Run 3D distance-to-mask measurement workflow with vessel assignment (no statistics)."
            ),
            "overrides": {
                "DO_SKELETONIZE": True,
                "DO_GRAPH_BUILDING": True,
                # Ensure branch-order vessel assignment is performed via mask-driven automation.
                "AUTOMATED_VESSEL_ASSIGNMENT": True,
                "USE_LARGE_VESSEL_MASKS": True,
                "USE_ILASTIK_LARGE_VESSEL_SEGMENTATION": False,
                "USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT": True,
                "USE_ILASTIK_SMALL_VESSEL_SEGMENTATION": False,
                "WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML": True,
                "STRICT_BRANCH_ORDER_ASSIGNMENT": True,
                "RUN_HAEMODYNAMICS": False,
                "DO_EQUIV_RESISTANCE_CALCULATION": False,
                "STATISTICS": False,
                "MEASUREMENT_3D_TO_CELL_MASK": True,
                # Keep unset by default; provide per run:
                # --set CELL_MASK_PATH='C:/path/to/cell_mask.tif'
                "CELL_MASK_PATH": None,
                "CELL_MASK_H5_DATASET_NAME": None,
                "MEASUREMENT_3D_VESSEL_MASK_PATH": None,
                "MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME": None,
                "MEASUREMENT_3D_REFERENCE_IMAGE_PATH": None,
                "MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME": None,
                "VERBOSE_LOGGING": True,
                "VISUALIZE_RESULTS": True,
                "SHOW_PLOTS_IN_IDE": False,
                "HOLD_IDE_PLOTS_OPEN": False,
                "INTERACTIVE_PLOTS": False,
                "VTK_export": False,
                "VISUALIZE_VTK": False,
                "FINAL_RENDER_MODE": "3d",
            },
        },
    }


def _load_local_preset_definitions(root_dir: Path) -> dict[str, dict[str, object]]:
    """Load optional user-defined presets from examples/local_presets.py."""
    local_presets_path = root_dir / "examples" / "local_presets.py"
    if not local_presets_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location(
        "imagelynx_local_presets",
        local_presets_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load local presets module from {local_presets_path}."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    local_defs = getattr(module, "LOCAL_PRESET_DEFINITIONS", {})
    if local_defs is None:
        return {}
    if not isinstance(local_defs, dict):
        raise ValueError(
            f"LOCAL_PRESET_DEFINITIONS in {local_presets_path} must be a dict."
        )
    return local_defs


def resolve_preset_inheritance(
    preset_definitions: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Resolve preset inheritance (`extends`) into flattened override dictionaries."""
    resolved: dict[str, dict[str, object]] = {}
    visiting: set[str] = set()

    def _coerce_payload(name: str, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError(f"Preset '{name}' must be a mapping.")
        return payload

    def _normalize_parents(raw_parents: object) -> list[str]:
        if raw_parents is None:
            return []
        if isinstance(raw_parents, str):
            return [raw_parents]
        if isinstance(raw_parents, list) and all(isinstance(x, str) for x in raw_parents):
            return list(raw_parents)
        raise ValueError("Preset 'extends' must be a string or list of strings.")

    def _resolve(name: str) -> dict[str, object]:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"Cyclic preset inheritance detected at '{name}'.")
        if name not in preset_definitions:
            options = ", ".join(sorted(preset_definitions.keys()))
            raise ValueError(
                f"Preset '{name}' is not defined. Available presets: {options}"
            )

        visiting.add(name)
        payload = _coerce_payload(name, preset_definitions[name])
        parents = _normalize_parents(payload.get("extends"))
        merged_overrides: dict[str, object] = {}
        merged_description = str(payload.get("description", "")).strip()
        for parent_name in parents:
            parent = _resolve(parent_name)
            merged_overrides.update(parent.get("overrides", {}))
            if not merged_description:
                merged_description = str(parent.get("description", "")).strip()

        own_overrides = payload.get("overrides", {})
        if own_overrides is None:
            own_overrides = {}
        if not isinstance(own_overrides, dict):
            raise ValueError(f"Preset '{name}' key 'overrides' must be a mapping.")
        merged_overrides.update(own_overrides)

        flattened = {
            "description": merged_description or f"Custom preset '{name}'",
            "overrides": merged_overrides,
        }
        visiting.remove(name)
        resolved[name] = flattened
        return flattened

    for preset_name in preset_definitions:
        _resolve(preset_name)
    return resolved


def get_preset_definitions(root_dir: Path) -> dict[str, dict[str, object]]:
    """Return built-in + local presets with inheritance resolved."""
    preset_definitions = _get_builtin_preset_definitions(root_dir)
    local_preset_definitions = _load_local_preset_definitions(root_dir)
    if local_preset_definitions:
        preset_definitions.update(local_preset_definitions)
    return resolve_preset_inheritance(preset_definitions)


def collect_setting_names(
    namespace: dict,
    preset_definitions: dict[str, dict[str, object]],
) -> set[str]:
    """Collect configurable setting names from a module namespace."""
    return {
        name
        for name in namespace
        if (
            (name.isupper() or name == "custom_edges")
            and not name.startswith("_")
            and name not in {"PRESET_DEFINITIONS"}
            and name not in set(preset_definitions.keys())
        )
    }


def collect_base_settings(
    namespace: dict,
    valid_setting_names: set[str],
) -> dict[str, object]:
    """Copy current setting values into a reusable baseline mapping."""
    base: dict[str, object] = {}
    for name in valid_setting_names:
        base[name] = copy.deepcopy(namespace[name])
    return base


def recompute_derived_settings(
    settings: dict[str, object],
    haemodynamics_module,
) -> None:
    """Rebuild derived diameter/constriction tables after overrides are applied."""
    settings["DIAMETER_BY_BRANCH_ORDER"] = haemodynamics_module.build_diameter_by_branch_order(
        all_diams_const=bool(settings["ALL_DIAMS_CONST"]),
        max_branch_order=int(settings["MAX_BRANCH_ORDER"]),
        default_diameter=float(settings["DEFAULT_DIAMETER"]),
        manual_capillary_diameter_by_branch_order=settings[
            "MANUAL_CAPILLARY_DIAMETER_BY_BRANCH_ORDER"
        ],
        manual_arteriole_diameter_by_branch_order=settings[
            "MANUAL_ARTERIOLE_DIAMETER_BY_BRANCH_ORDER"
        ],
        manual_venule_diameter_by_branch_order=settings[
            "MANUAL_VENULE_DIAMETER_BY_BRANCH_ORDER"
        ],
    )
    max_branch_order = int(settings["MAX_BRANCH_ORDER"])
    constriction: dict[str, float] = {"B01": 1.0}
    for i in range(2, max_branch_order + 1):
        constriction[f"B{i:02d}"] = 0.8
    constriction["Art1"] = 1.0
    constriction["Ven1"] = 1.0
    for i in range(2, max_branch_order + 1):
        constriction[f"Art{i}"] = 0.8
        constriction[f"Ven{i}"] = 0.8
    settings["CONSTRICTION_BY_BRANCH_ORDER"] = constriction


def list_presets(preset_definitions: dict[str, dict[str, object]]) -> dict[str, str]:
    """Return preset names mapped to human-readable descriptions."""
    return {
        name: str(payload["description"])
        for name, payload in preset_definitions.items()
    }


def _parse_cli_value(text: str) -> object:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _coerce_path_like_value(key: str, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        if key.endswith("_PATH") or key.endswith("_DIR") or key.endswith("_PREFIX"):
            return Path(value)
    return value


def parse_cli_override(
    override_text: str,
    valid_setting_names: set[str],
) -> tuple[str, object]:
    """Parse one KEY=VALUE override string into a typed setting pair."""
    if "=" not in override_text:
        raise ValueError(
            f"Invalid override '{override_text}'. Use KEY=VALUE format."
        )
    key_text, raw_value = override_text.split("=", 1)
    key = key_text.strip()
    if not key:
        raise ValueError(f"Invalid override '{override_text}': empty KEY.")
    key = key if key == "custom_edges" else key.upper()
    if key not in valid_setting_names:
        available = ", ".join(sorted(valid_setting_names))
        raise ValueError(
            f"Unknown setting '{key}' in override '{override_text}'. "
            f"Valid settings: {available}"
        )
    value = _parse_cli_value(raw_value.strip())
    value = _coerce_path_like_value(key, value)
    return key, value


def build_settings_for_preset(
    preset_name: str,
    preset_definitions: dict[str, dict[str, object]],
    valid_setting_names: set[str],
    base_settings_template: dict[str, object],
    haemodynamics_module,
    manual_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build final setting map from preset + manual overrides."""
    if preset_name not in preset_definitions:
        available = ", ".join(sorted(preset_definitions))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    settings = copy.deepcopy(base_settings_template)
    preset_overrides = dict(preset_definitions[preset_name]["overrides"])
    settings.update(preset_overrides)
    if manual_overrides:
        unknown = [k for k in manual_overrides if k not in valid_setting_names]
        if unknown:
            available = ", ".join(sorted(valid_setting_names))
            raise ValueError(
                f"Unknown manual override settings: {unknown}. "
                f"Valid settings: {available}"
            )
        for key, value in manual_overrides.items():
            settings[key] = _coerce_path_like_value(key, value)
    recompute_derived_settings(settings, haemodynamics_module)
    return settings


def apply_settings_to_namespace(
    settings: dict[str, object],
    namespace: dict,
    valid_setting_names: set[str],
) -> None:
    """Write resolved setting values back to a module namespace."""
    for key, value in settings.items():
        if key in valid_setting_names:
            namespace[key] = value


def _coerce_pipeline_path_like_value(param_name: str, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        if (
            param_name.endswith("_path")
            or param_name.endswith("_dir")
            or param_name.endswith("_prefix")
            or param_name == "plot_dir"
        ):
            return Path(value)
    return value


def _normalize_setting_key(raw_key: str, valid_setting_names: set[str]) -> str | None:
    if raw_key in valid_setting_names:
        return raw_key
    maybe_upper = raw_key.upper()
    if maybe_upper in valid_setting_names:
        return maybe_upper
    return None


def _serialize_for_yaml(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _serialize_for_yaml(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_yaml(v) for v in value]
    return value


def load_config_yaml(
    config_path: str | Path,
    valid_setting_names: set[str],
    available_preset_names: set[str],
    pipeline_param_names: set[str],
) -> dict[str, object]:
    """Load config YAML and split it into preset/settings/pipeline overrides."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config support. Install with `pip install pyyaml`."
        ) from exc

    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file '{config_path}' must contain a YAML mapping at top level."
        )

    preset_name = data.get("preset")
    if preset_name is not None:
        if not isinstance(preset_name, str):
            raise ValueError(
                f"Config key 'preset' must be a string in '{config_path}'."
            )
        if preset_name not in available_preset_names:
            options = ", ".join(sorted(available_preset_names))
            raise ValueError(
                f"Unknown preset '{preset_name}' in '{config_path}'. "
                f"Available presets: {options}"
            )

    settings_overrides: dict[str, object] = {}
    pipeline_overrides: dict[str, object] = {}

    explicit_settings = data.get("settings", {})
    if explicit_settings is None:
        explicit_settings = {}
    if not isinstance(explicit_settings, dict):
        raise ValueError(
            f"Config key 'settings' must be a mapping in '{config_path}'."
        )
    for raw_key, value in explicit_settings.items():
        key = _normalize_setting_key(str(raw_key), valid_setting_names)
        if key is None:
            options = ", ".join(sorted(valid_setting_names))
            raise ValueError(
                f"Unknown setting '{raw_key}' in config.settings of '{config_path}'. "
                f"Valid settings: {options}"
            )
        settings_overrides[key] = _coerce_path_like_value(key, value)

    explicit_pipeline = data.get("pipeline", {})
    if explicit_pipeline is None:
        explicit_pipeline = {}
    if not isinstance(explicit_pipeline, dict):
        raise ValueError(
            f"Config key 'pipeline' must be a mapping in '{config_path}'."
        )
    for raw_key, value in explicit_pipeline.items():
        key = str(raw_key)
        if key not in pipeline_param_names:
            options = ", ".join(sorted(pipeline_param_names))
            raise ValueError(
                f"Unknown pipeline parameter '{raw_key}' in config.pipeline of "
                f"'{config_path}'. Valid parameters: {options}"
            )
        pipeline_overrides[key] = _coerce_pipeline_path_like_value(key, value)

    reserved = {"preset", "settings", "pipeline"}
    for raw_key, value in data.items():
        if raw_key in reserved:
            continue
        raw_key_str = str(raw_key)
        setting_key = _normalize_setting_key(raw_key_str, valid_setting_names)
        if setting_key is not None:
            settings_overrides[setting_key] = _coerce_path_like_value(
                setting_key,
                value,
            )
            continue
        if raw_key_str in pipeline_param_names:
            pipeline_overrides[raw_key_str] = _coerce_pipeline_path_like_value(
                raw_key_str,
                value,
            )
            continue
        options = ", ".join(sorted(valid_setting_names | pipeline_param_names | reserved))
        raise ValueError(
            f"Unknown config key '{raw_key}' in '{config_path}'. "
            f"Allowed keys: {options}"
        )

    return {
        "preset_name": preset_name,
        "settings_overrides": settings_overrides,
        "pipeline_overrides": pipeline_overrides,
    }


def save_effective_config_yaml(
    output_path: str | Path,
    preset_name: str,
    settings: dict[str, object],
    pipeline_kwargs: dict[str, object],
) -> Path:
    """Write effective run configuration to a YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --save-config support. Install with `pip install pyyaml`."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": preset_name,
        "settings": _serialize_for_yaml(settings),
        "pipeline": _serialize_for_yaml(pipeline_kwargs),
    }
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    return output_path
