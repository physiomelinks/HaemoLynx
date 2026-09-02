"""Compact per-grid-point edge flows for sweep perturbations.

Sweep helpers solve many networks; napari needs one Vectors geometry and a way
to swap ``flow_abs`` (and related columns) as the user moves along the grid.
Keeping a full ``MultiGraph`` per grid point is wasteful; this module records
only the edge-feature arrays, in the graph's ``edges(keys=True)`` order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from .resistance import set_edge_flows

__all__ = [
    "SweepFlowGrid",
    "edge_attr_array",
    "node_pressure_array",
    "record_flows_after_solve",
    "build_sweep_flow_grid",
]


def edge_attr_array(G: nx.Graph, name: str) -> np.ndarray:
    """One float per edge in ``edges(keys=True)`` order; missing -> NaN."""
    values: list[float] = []
    if getattr(G, "is_multigraph", lambda: False)():
        edges = G.edges(keys=True, data=True)
        for _u, _v, _key, data in edges:
            value = data.get(name)
            values.append(float(value) if value is not None else np.nan)
    else:
        for _u, _v, data in G.edges(data=True):
            value = data.get(name)
            values.append(float(value) if value is not None else np.nan)
    return np.asarray(values, dtype=float)


def node_pressure_array(G: nx.Graph, node_list: Sequence[Any]) -> np.ndarray:
    """Nodal pressures in *node_list* order; missing -> NaN."""
    return np.asarray(
        [
            float(G.nodes[node_id].get("pressure", np.nan))
            if node_id in G
            else np.nan
            for node_id in node_list
        ],
        dtype=float,
    )


def record_flows_after_solve(
    G: nx.Graph, node_list: Sequence[Any], pressure: np.ndarray
) -> dict[str, np.ndarray]:
    """Write flows onto *G* and return compact edge/node feature arrays."""
    set_edge_flows(G, list(node_list), pressure)
    return {
        "flow_abs": edge_attr_array(G, "flow_abs"),
        "flow_signed": edge_attr_array(G, "flow_signed"),
        "pressure_drop": edge_attr_array(G, "pressure_drop"),
        "node_pressure": node_pressure_array(G, node_list),
    }


@dataclass(frozen=True)
class SweepFlowGrid:
    """Per-grid-point edge flows for a slider-backed napari Vectors layer.

    *axis_names* are the swept coordinates in nest order (outer first). Rows of
    *flow_abs* are C-order over those axes: for dilation × pressure, row
    ``i * n_pressure + j`` is dilation ``i`` at pressure ``j``.
    """

    axis_names: tuple[str, ...]
    axis_values: Mapping[str, np.ndarray]
    flow_abs: np.ndarray  # (n_points, n_edges)
    flow_signed: np.ndarray | None = None
    pressure_drop: np.ndarray | None = None
    node_pressure: np.ndarray | None = None  # (n_points, n_nodes)
    node_list: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.flow_abs.ndim != 2:
            raise ValueError(
                f"flow_abs must be 2-D (n_points, n_edges); got shape "
                f"{self.flow_abs.shape}."
            )
        expected = 1
        for name in self.axis_names:
            if name not in self.axis_values:
                raise ValueError(f"axis {name!r} missing from axis_values.")
            expected *= int(len(self.axis_values[name]))
        if self.flow_abs.shape[0] != expected:
            raise ValueError(
                f"flow_abs has {self.flow_abs.shape[0]} rows but axes "
                f"{self.axis_names} imply {expected} grid points."
            )

    @property
    def n_points(self) -> int:
        return int(self.flow_abs.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.flow_abs.shape[1])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.axis_values[name]) for name in self.axis_names)

    def flat_index(self, *indices: int) -> int:
        """Multi-index along *axis_names* -> row into ``flow_abs`` (C-order)."""
        if len(indices) != len(self.axis_names):
            raise ValueError(
                f"expected {len(self.axis_names)} index(es) for axes "
                f"{self.axis_names}, got {len(indices)}."
            )
        flat = 0
        for index, name in zip(indices, self.axis_names):
            size = len(self.axis_values[name])
            if index < 0 or index >= size:
                raise IndexError(
                    f"index {index} out of range for axis {name!r} "
                    f"(size {size})."
                )
            flat = flat * size + int(index)
        return flat

    def flow_abs_at(self, *indices: int) -> np.ndarray:
        return self.flow_abs[self.flat_index(*indices)]

    def flow_signed_at(self, *indices: int) -> np.ndarray | None:
        if self.flow_signed is None:
            return None
        return self.flow_signed[self.flat_index(*indices)]

    def pressure_drop_at(self, *indices: int) -> np.ndarray | None:
        if self.pressure_drop is None:
            return None
        return self.pressure_drop[self.flat_index(*indices)]

    def node_pressure_at(self, *indices: int) -> np.ndarray | None:
        if self.node_pressure is None:
            return None
        return self.node_pressure[self.flat_index(*indices)]

    def global_flow_abs_limits(self) -> tuple[float, float] | None:
        """Contrast limits spanning every grid point (nan-aware)."""
        values = self.flow_abs[np.isfinite(self.flow_abs)]
        if values.size == 0:
            return None
        low = float(np.min(values))
        high = float(np.max(values))
        if low == high:
            high = low + 1.0
        return low, high


def build_sweep_flow_grid(
    *,
    axis_names: Sequence[str],
    axis_values: Mapping[str, Sequence[Any]],
    recorded: Sequence[Mapping[str, np.ndarray]],
    node_list: Sequence[Any] | None = None,
) -> SweepFlowGrid:
    """Stack per-point ``record_flows_after_solve`` dicts into a grid."""
    if not recorded:
        raise ValueError("recorded must contain at least one grid point.")
    names = tuple(str(name) for name in axis_names)
    axes = {
        name: np.asarray(axis_values[name])
        for name in names
    }
    flow_abs = np.stack([np.asarray(row["flow_abs"], dtype=float) for row in recorded])
    flow_signed = np.stack(
        [np.asarray(row["flow_signed"], dtype=float) for row in recorded]
    )
    pressure_drop = np.stack(
        [np.asarray(row["pressure_drop"], dtype=float) for row in recorded]
    )
    node_pressure = None
    if all("node_pressure" in row for row in recorded):
        node_pressure = np.stack(
            [np.asarray(row["node_pressure"], dtype=float) for row in recorded]
        )
    return SweepFlowGrid(
        axis_names=names,
        axis_values=axes,
        flow_abs=flow_abs,
        flow_signed=flow_signed,
        pressure_drop=pressure_drop,
        node_pressure=node_pressure,
        node_list=tuple(node_list or ()),
    )
