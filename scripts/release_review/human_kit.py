"""Scoring what the testers hand back: the SUS sheet, and hand-measured
diameters against the methods' (``manual_diameter_protocol.md``).

Run as ``python -m release_review.human_kit compare <lines.csv> <graph.pkl>
--voxel-size-zyx Z Y X`` (from ``scripts/``) to compare a napari Shapes
layer's lines with a run's graph. Nothing here decides an H check: the
tester's result goes into ``human_checks.yaml`` by hand.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: ``diameter.8b``'s agreement and how many vessels it needs.
AGREEMENT = 0.15
VESSELS_NEEDED = 30


def sus_score(answers: Sequence[int]) -> float:
    """The System Usability Scale score, 0 to 100, of ten answers 1-5: odd
    items score the answer minus 1, even items 5 minus the answer, and the
    sum is scaled by 2.5."""
    if len(answers) != 10 or any(answer not in (1, 2, 3, 4, 5) for answer in answers):
        raise ValueError("the SUS takes ten answers, each 1 to 5")
    total = sum(answer - 1 if item % 2 == 1 else 5 - answer for item, answer in enumerate(answers, start=1))
    return total * 2.5


# ---------------------------------------------------------------------------
# Hand-measured diameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandLine:
    index: int
    start_zyx: tuple[float, float, float]
    end_zyx: tuple[float, float, float]


def read_shapes_csv(path: Path) -> list[HandLine]:
    """The lines of a napari Shapes layer saved as CSV (``index``,
    ``shape-type``, ``vertex-index``, ``axis-0``..``axis-2``, in pixels)."""
    vertices: dict[int, list[tuple[int, tuple[float, ...]]]] = {}
    kinds: dict[int, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            kinds[index] = row.get("shape-type", "line")
            point = tuple(float(row[f"axis-{axis}"]) for axis in range(3))
            vertices.setdefault(index, []).append((int(row["vertex-index"]), point))
    lines = []
    for index in sorted(vertices):
        points = [point for _order, point in sorted(vertices[index])]
        if kinds[index] != "line" or len(points) != 2:
            raise ValueError(f"shape {index} is a {kinds[index]} of {len(points)} vertices, not a line")
        lines.append(HandLine(index, points[0], points[1]))
    return lines


def line_length_um(line: HandLine, voxel_size_zyx: Sequence[float]) -> float:
    step = (np.asarray(line.end_zyx) - np.asarray(line.start_zyx)) * np.asarray(voxel_size_zyx, dtype=float)
    return float(np.linalg.norm(step))


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    a, b = polyline[:-1], polyline[1:]
    direction = b - a
    lengths = np.maximum((direction * direction).sum(axis=1), 1e-12)
    t = np.clip(((point - a) * direction).sum(axis=1) / lengths, 0.0, 1.0)
    return float(np.min(np.linalg.norm(point - (a + t[:, None] * direction), axis=1)))


def nearest_edge(
    point_um: Sequence[float], centrelines: Mapping[Any, np.ndarray]
) -> tuple[Optional[Any], float]:
    """The edge whose centreline (an (n, 3) array in microns) passes nearest
    *point_um*, and how near."""
    point = np.asarray(point_um, dtype=float)
    best, distance = None, float("inf")
    for key, polyline in centrelines.items():
        found = _distance_to_polyline(point, polyline)
        if found < distance:
            best, distance = key, found
    return best, distance


def centrelines_um(graph, voxel_size_zyx: Sequence[float]) -> dict[Any, np.ndarray]:
    """Each edge's centreline in microns: its ``voxels`` scaled, or the
    straight line between its nodes when it has none."""
    scale = np.asarray(voxel_size_zyx, dtype=float)
    lines = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        voxels = data.get("voxels")
        if voxels is not None and len(voxels):
            lines[(u, v, key)] = np.asarray(voxels, dtype=float) * scale
        else:
            lines[(u, v, key)] = np.asarray([graph.nodes[u]["pos"], graph.nodes[v]["pos"]], dtype=float)
    return lines


def method_widths(data: Mapping[str, Any]) -> dict[str, float]:
    """Every diameter the edge carries: the one the model used
    (``diameter_um``) and each method's own (``*_diameter_um``)."""
    widths = {}
    for name, value in data.items():
        if name == "diameter_um" or (name.endswith("_diameter_um") and not name.endswith("samples_um")):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number) and number > 0:
                widths[name] = number
    return widths


def compare(lines: Sequence[HandLine], graph, voxel_size_zyx: Sequence[float]) -> dict[str, Any]:
    """Each hand line against the nearest vessel's widths, and per method the
    share within :data:`AGREEMENT` and the mean signed difference (bias)."""
    centrelines = centrelines_um(graph, voxel_size_zyx)
    edges = {(u, v, key): data for u, v, key, data in graph.edges(keys=True, data=True)}
    rows, unmatched = [], []
    for line in lines:
        width = line_length_um(line, voxel_size_zyx)
        middle = (np.asarray(line.start_zyx) + np.asarray(line.end_zyx)) / 2.0 * np.asarray(voxel_size_zyx)
        key, distance = nearest_edge(middle, centrelines)
        if key is None or distance > width:
            unmatched.append({"line": line.index, "hand_um": round(width, 2), "nearest_um": round(distance, 2)})
            continue
        for method, measured in method_widths(edges[key]).items():
            rows.append({"line": line.index, "edge": list(key), "method": method, "hand_um": round(width, 3),
                         "method_um": round(measured, 3), "relative": (measured - width) / width})
    methods = {}
    for method in sorted({row["method"] for row in rows}):
        differences = np.array([row["relative"] for row in rows if row["method"] == method])
        methods[method] = {
            "vessels": int(differences.size),
            "within": float(np.mean(np.abs(differences) <= AGREEMENT)),
            "bias": float(np.mean(differences)),
            "enough": bool(differences.size >= VESSELS_NEEDED),
        }
    return {"rows": rows, "unmatched": unmatched, "methods": methods}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare hand-measured diameters with a run's (diameter.8b).")
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("lines", type=Path)
    compare_parser.add_argument("graph", type=Path)
    compare_parser.add_argument("--voxel-size-zyx", type=float, nargs=3, required=True)
    arguments = parser.parse_args(argv)
    from .exports import load_graph

    found = compare(read_shapes_csv(arguments.lines), load_graph(arguments.graph), arguments.voxel_size_zyx)
    for method, summary in found["methods"].items():
        print(f"{method}: {summary['vessels']} vessels, {100 * summary['within']:.0f}% within "
              f"{100 * AGREEMENT:.0f}%, bias {100 * summary['bias']:+.1f}%"
              + ("" if summary["enough"] else f" (fewer than {VESSELS_NEEDED})"))
    for each in found["unmatched"]:
        print(f"line {each['line']}: no vessel within its own width ({each['nearest_um']} um away)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
