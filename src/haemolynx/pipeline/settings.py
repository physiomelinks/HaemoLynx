"""Resolving a pipeline run's settings from its config file.

Generic config machinery lives in :mod:`haemolynx.parsers`; this module holds
the pipeline's own rules — which settings are derived from which, and where the
plots go — so both examples resolve settings the same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from haemolynx import haemodynamics
from haemolynx.parsers import Schema, load_config


def resolve_settings(
    settings: dict | None = None,
    *,
    schema: Schema,
    config_path: Path | str,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Every setting for one run, validated against the schema.

    With no arguments this is exactly what the config file says. Pass
    ``settings`` to supply an already-loaded dict, and ``overrides`` to change
    individual values on top of either.
    """
    overrides = dict(overrides or {})
    plot_dir = overrides.pop("plot_dir", None)
    if settings is None:
        resolved = load_config(config_path, schema, overrides=overrides or None)
    else:
        # An already-resolved dict carries the derived entries too; they are not
        # schema settings, so set them aside rather than failing validation.
        merged = {**settings, **overrides}
        plot_dir = plot_dir or merged.pop("plot_dir", None)
        resolved = schema.validate({k: v for k, v in merged.items() if k in schema})
    fill_derived_settings(resolved)
    if plot_dir is not None:
        resolved["plot_dir"] = Path(plot_dir)
    return resolved


def fill_derived_settings(settings: dict) -> None:
    """Add the settings computed from other settings.

    The branch-order tables are functions of the manual diameter settings, so
    the config file states those and leaves these null rather than duplicating
    a 150-entry table that could then disagree with them.
    """
    settings.setdefault("plot_dir", Path(settings["base_plot_dir"]) / "nerve")
    if settings.get("diameter_by_branch_order") is None:
        settings["diameter_by_branch_order"] = haemodynamics.build_diameter_by_branch_order(
            all_diams_const=settings["all_diams_const"],
            max_branch_order=settings["max_branch_order"],
            default_diameter=settings["default_diameter"],
            manual_capillary_diameter_by_branch_order=settings[
                "manual_capillary_diameter_by_branch_order"
            ],
            manual_arteriole_diameter_by_branch_order=settings[
                "manual_arteriole_diameter_by_branch_order"
            ],
            manual_venule_diameter_by_branch_order=settings[
                "manual_venule_diameter_by_branch_order"
            ],
        )
    if settings.get("constriction_by_branch_order") is None:
        # Empty override map: every order keeps pericyte_constriction_factor.
        # Listed keys replace that global factor for those orders only
        # (see constriction_strategy.resolve_constriction_factor_table).
        settings["constriction_by_branch_order"] = {}
