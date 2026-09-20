"""Turning the comprehensive statistics dict into a readable, annotated CSV."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union
import csv
import json
import re

import numpy as np


def _flatten_statistics_dict(
    stats: Dict[str, Any], prefix: str = ""
) -> list[tuple[str, Any]]:
    """Flatten nested statistics dict for tabular export."""
    flattened: list[tuple[str, Any]] = []
    for key, value in stats.items():
        full_key = f"{prefix} > {key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.extend(_flatten_statistics_dict(value, full_key))
        else:
            flattened.append((full_key, value))
    return flattened


def _split_metric_path(metric_path: str) -> tuple[str, str]:
    """Split flattened key path into section and metric."""
    if " > " not in metric_path:
        return "Core Statistics", metric_path
    section, metric = metric_path.rsplit(" > ", 1)
    return section, metric


def _extract_unit(metric_name: str) -> tuple[str, str]:
    """Extract trailing '(unit)' from metric labels when present."""
    m = re.search(r"\(([^()]+)\)\s*$", metric_name)
    if not m:
        return metric_name, ""
    unit = m.group(1).strip()
    clean_metric = re.sub(r"\s*\([^()]+\)\s*$", "", metric_name).strip()
    return clean_metric, unit


def _numeric_csv_text(value: Any) -> str:
    """A metric value as plain CSV text: ``.6g`` for a real number, else its
    string form -- typically an "N/A (...)" explanation.

    Shared by both CSV exporters' own numeric-formatting rule --
    :func:`haemolynx.statistics.bifurcation.export_branch_order_statistics_to_csv`
    reuses this directly (its own per-field notes are specific enough that
    folding them into :func:`_annotation_for_metric`'s metric-name pattern
    matching would lose that detail, so only the formatting rule, not the
    note text, is shared).
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _format_stat_value(metric_name: str, value: Any) -> tuple[str, str]:
    """Format metric value plus a short human-readable note."""
    if isinstance(value, str):
        if value.startswith("N/A"):
            return value, "Not computable with current inputs."
        return value, ""

    if isinstance(value, (bool, np.bool_)):
        return ("True" if bool(value) else "False"), ""

    if isinstance(value, (int, np.integer)):
        return str(int(value)), ""

    if isinstance(value, (float, np.floating)):
        f_val = float(value)
        if "Coverage" in metric_name:
            return f"{f_val:.2%}", "Fraction of all node-pairs included."
        return _numeric_csv_text(value), ""

    if isinstance(value, (list, tuple, set)):
        serializable = list(value) if not isinstance(value, list) else value
        try:
            return json.dumps(serializable), "Serialized collection value."
        except TypeError:
            return str(serializable), "Collection converted to text."

    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True), "Serialized nested object."
        except TypeError:
            return str(value), "Nested object converted to text."

    return str(value), ""


def _annotation_for_metric(metric_name: str) -> str:
    """Return short metric-specific annotation used in CSV notes."""
    if metric_name == "Statistics Mode":
        return "fast=sampled/compact metrics, full=heavier/exact metrics."
    if "Tortuosity" in metric_name:
        return "Higher values indicate less straight vessels."
    if "Curvature" in metric_name:
        return "Relative deviation from straight vessel segments."
    if "Branching Points" in metric_name:
        return (
            "Count of junctions (degree > 2); see the per-branch-order "
            "Mean Emergence Angle for angle detail."
        )
    if "Asymmetry" in metric_name:
        return "Tree imbalance estimate after simplification to a tree."
    if "Fractal Dimension" in metric_name:
        return "Box-counting estimate of structural complexity."
    if "Path Efficiency" in metric_name:
        return "Inverse of mean shortest-path distance."
    if "Betweenness" in metric_name:
        return "Centrality based on shortest-path traffic."
    if "Community" in metric_name:
        return "Subnetwork partition summary."
    if "Density" in metric_name:
        return "Total vessel length normalized by available volume."
    return ""


def export_statistics_to_csv(
    stats: Dict[str, Any],
    output_csv_path: Union[str, Path],
) -> Path:
    """Export statistics dictionary to a readable annotated CSV file.

    The CSV is long-format with one row per metric and includes:
    section, metric name, value, parsed unit, and notes.
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flattened = _flatten_statistics_dict(stats)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric", "Value", "Unit", "Notes"])
        writer.writerow(
            [
                "Metadata",
                "Exported At (UTC)",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "",
                "Generated by HaemoLynx statistics exporter.",
            ]
        )
        writer.writerow(
            [
                "Metadata",
                "Metric Count",
                str(len(flattened)),
                "",
                "Number of metric rows in this file.",
            ]
        )

        for metric_path, value in flattened:
            section, metric = _split_metric_path(metric_path)
            metric_clean, unit = _extract_unit(metric)
            formatted_value, value_note = _format_stat_value(metric_clean, value)
            metric_note = _annotation_for_metric(metric_clean)
            notes = " ".join(n for n in [value_note, metric_note] if n).strip()
            writer.writerow([section, metric_clean, formatted_value, unit, notes])

    return output_path
