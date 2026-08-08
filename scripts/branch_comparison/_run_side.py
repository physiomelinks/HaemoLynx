#!/usr/bin/env python3
"""Run one side of a branch comparison, inside that side's checkout.

This runs as its own process with ``PYTHONPATH`` pointed at the checkout under
test, so it must stay standalone: standard library plus whatever that checkout
provides. It never imports the comparison package, and it takes every decision
from a JSON spec written by the orchestrator.

Two jobs, chosen by ``mode``:

``resolve``
    Import the entry point, work out which settings it accepts, and dump the
    full resolved settings dict. The orchestrator uses that dict to pin both
    sides to the same configuration.

``run``
    Apply the settings and run the pipeline, timing it and capturing the final
    graph.

The settings the entry point cannot accept are reported by name; the run is
refused rather than quietly using that branch's own default.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

ENTRY_POINT_RELATIVE = "examples/resistance_network_pipeline.py"
ENTRY_POINT_FUNCTION = "image_to_model_pipeline"


def _jsonable(value):
    """A JSON-safe copy, keeping numbers as numbers."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for attr in ("tolist", "item"):
        if hasattr(value, attr):
            try:
                return _jsonable(getattr(value, attr)())
            except Exception:  # noqa: BLE001 - fall through to the string form
                pass
    return str(value)


def _load_entry_point(checkout: Path):
    """Import that checkout's pipeline example as a module."""
    path = checkout / ENTRY_POINT_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"No pipeline entry point at {path}")
    spec = importlib.util.spec_from_file_location(
        "branch_comparison_pipeline_under_test", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load a module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



#: The import package was renamed ImageLynx -> haemolynx. This module runs
#: inside the reference checkout, which may predate that, so both are tried.
PACKAGE_NAMES = ("haemolynx", "ImageLynx")


def _import_library(submodule: str | None = None):
    """The library of the checkout being run, under whichever name it has."""
    import importlib

    errors = []
    for name in PACKAGE_NAMES:
        target = f"{name}.{submodule}" if submodule else name
        try:
            return importlib.import_module(target)
        except ImportError as error:  # noqa: PERF203 - two candidates, at most
            errors.append(f"{target}: {error}")
    raise ImportError("; ".join(errors))


def _assert_checkout_is_in_use(checkout: Path) -> None:
    """Fail loudly if the library resolved somewhere other than this checkout.

    The development install points at one working tree, so without this the
    reference run would silently exercise the current branch's library and the
    comparison would report "no differences" for every change.
    """
    library = _import_library()

    resolved = Path(library.__file__).resolve()
    expected = (checkout / "src").resolve()
    if expected not in resolved.parents:
        raise RuntimeError(
            f"{library.__name__} was imported from {resolved}, which is not "
            f"inside {expected}. The comparison would not be measuring this "
            "checkout."
        )


def _schema_accepts(schema, name: str) -> bool:
    if schema is None:
        return False
    try:
        return name in schema
    except TypeError:
        return False


def _apply_settings(module, spec) -> dict:
    """Map canonical setting names onto whatever this branch accepts.

    Three routes, in order: a named parameter of the entry point, a schema
    setting passed through ``**overrides``, or a module-level constant the
    function body reads. Anything with no route is reported.
    """
    function = getattr(module, ENTRY_POINT_FUNCTION)
    parameters = inspect.signature(function).parameters
    named = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and name != "settings"
    }
    has_var_keyword = any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    schema = getattr(module, "SCHEMA", None)
    api_style = (
        "settings-dict (**overrides validated by SCHEMA)"
        if has_var_keyword and schema is not None
        else f"keyword-arguments ({len(named)} named parameters)"
    )
    derived = set(spec.get("derived") or ())
    # JSON has no path type, and the keyword-argument entry points do no
    # coercion of their own -- they just use what they are given, so a string
    # where a Path is expected fails deep inside the run.
    path_settings = set(spec.get("path_settings") or ())

    def as_path_if_needed(setting: str, alias: str, value):
        if not isinstance(value, str):
            return value
        default = parameters[alias].default if alias in parameters else None
        if setting in path_settings or isinstance(default, Path):
            return Path(value)
        return value

    kwargs: dict = {}
    globals_to_set: dict = {}
    applied: dict[str, str] = {}
    unapplied: dict[str, str] = {}

    for name, value in spec["settings"].items():
        aliases = spec.get("aliases", {}).get(name) or [name]

        # Arguments first, across every alias. A module constant of the same
        # name is usually just that argument's default, evaluated at import
        # time, so setting it would change nothing while looking like success.
        for alias in aliases:
            if alias in named:
                kwargs[alias] = as_path_if_needed(name, alias, value)
                applied[name] = f"keyword argument '{alias}'"
                break
            if has_var_keyword and (
                _schema_accepts(schema, alias) or alias in derived
            ):
                kwargs[alias] = value
                applied[name] = f"settings override '{alias}'"
                break
        if name in applied:
            continue

        # Only settings the entry point does not take at all can be reached
        # through the module constant its body reads.
        for alias in aliases:
            constant = alias.upper()
            if hasattr(module, constant):
                globals_to_set[constant] = as_path_if_needed(name, alias, value)
                applied[name] = f"module constant '{constant}'"
                break
        else:
            unapplied[name] = (
                "no parameter, schema setting or module constant of this name "
                f"(tried {', '.join(aliases)})"
            )

    for constant, value in globals_to_set.items():
        setattr(module, constant, value)

    return {
        "api_style": api_style,
        "kwargs": kwargs,
        "applied": applied,
        "unapplied": unapplied,
    }


def _install_graph_capture(destination: Path) -> dict:
    """Pickle the graph handed to the VTK export, on whichever branch this is.

    Older entry points return ``None``, so the export call is the one place
    both APIs expose the finished graph -- after haemodynamics and the solve.
    """
    state = {"captured": False}
    try:
        visualization = _import_library("visualization")
    except Exception:  # noqa: BLE001 - a broken import is reported by the run itself
        return state

    original = getattr(visualization, "graph_to_vtk", None)
    if original is None:
        return state

    def capturing_graph_to_vtk(graph, *args, **kwargs):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                pickle.dump(graph, handle)
            state["captured"] = True
        except Exception:  # noqa: BLE001 - never fail the run over the capture
            state["captured"] = False
        return original(graph, *args, **kwargs)

    visualization.graph_to_vtk = capturing_graph_to_vtk
    return state


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    result_path = Path(spec["result_path"])
    result: dict = {"ok": False, "error": None, "mode": spec["mode"]}

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:  # noqa: BLE001 - plotting is optional to the comparison
        pass

    try:
        checkout = Path(spec["checkout"]).resolve()
        _assert_checkout_is_in_use(checkout)
        module = _load_entry_point(checkout)
        plan = _apply_settings(module, spec)
        result.update(
            api_style=plan["api_style"],
            applied=plan["applied"],
            unapplied=plan["unapplied"],
        )

        if plan["unapplied"] and not spec.get("allow_unapplied"):
            required = set(spec.get("required") or ())
            blocking = sorted(set(plan["unapplied"]) & required)
            if blocking:
                raise RuntimeError(
                    "This branch cannot apply settings that define the "
                    f"comparison: {', '.join(blocking)}. Running anyway would "
                    "compare two different configurations."
                )

        if spec["mode"] == "resolve":
            resolver = getattr(module, "resolve_settings", None)
            if resolver is None:
                raise RuntimeError(
                    "This branch has no resolve_settings(); its full settings "
                    "cannot be read without running it."
                )
            resolved = dict(resolver(overrides=plan["kwargs"]))
            result["resolved_settings"] = _jsonable(resolved)
            # Recorded before the JSON round-trip flattens them to strings, so
            # the other side can restore the type its entry point expects.
            result["path_settings"] = sorted(
                name for name, value in resolved.items() if isinstance(value, Path)
            )
            result["ok"] = True
        else:
            capture = _install_graph_capture(Path(spec["graph_capture_path"]))
            started = time.perf_counter()
            returned = getattr(module, ENTRY_POINT_FUNCTION)(**plan["kwargs"])
            result["runtime_seconds"] = time.perf_counter() - started
            source = None
            if capture["captured"]:
                source = "graph passed to the VTK export"
            if returned is not None and hasattr(returned, "number_of_nodes"):
                with Path(spec["graph_capture_path"]).open("wb") as handle:
                    pickle.dump(returned, handle)
                source = "graph returned by image_to_model_pipeline()"
            result["final_graph_source"] = source
            result["ok"] = True
    except BaseException as error:  # noqa: BLE001 - the failure is the report
        result["error"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        result["ok"] = False

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    sys.exit(main(sys.argv))
