"""Constriction sites placed at a fixed spacing, each active with a probability.

This is the model for a run with no pericyte mask: assume pericytes sit at
regular intervals along every capillary, then let each one contract or not.
The narrowing itself, and how it becomes an edge resistance, is
:mod:`haemolynx.haemodynamics.constriction`.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .constriction import (
    apply_constriction_sites,
    is_capillary_branch_order,
    require_enough_integration_points,
    require_positive_constriction_length,
    resolve_generator,
    select_active_pericyte_indices,
    validate_active_pericyte_indices,
)
from .viscosity import DEFAULT_HAEMATOCRIT

__all__ = [
    "is_capillary_branch_order",
    "resolve_generator",
    "select_active_pericyte_indices",
    "validate_active_pericyte_indices",
    "set_poiseuille_resistances_with_probabilistic_periodic_constrictions",
]


def _periodic_center_positions(
    length: float,
    constriction_length: float,
    constriction_spacing: float,
) -> list[float]:
    if length <= 0 or constriction_length <= 0 or constriction_spacing <= 0:
        return []
    positions: list[float] = []
    sample_pos = float(constriction_length) / 2.0
    while sample_pos <= float(length):
        positions.append(float(sample_pos))
        sample_pos += float(constriction_spacing)
    return positions


class PeriodicConstrictionSites:
    """Sites every ``constriction_spacing`` microns along each capillary.

    Each site is then activated independently with ``constriction_probability``,
    unless ``active_center_indices_by_edge`` names a fixed cohort — which is how
    a comparison run applies the very same pericytes to its baseline and its
    constricted graph. Its keys are ``"u|v|key"``.
    """

    def __init__(
        self,
        *,
        constriction_length: float,
        constriction_spacing: float,
        constriction_probability: float,
        active_center_indices_by_edge: dict[str, list[int]] | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        self.constriction_length = float(constriction_length)
        self.constriction_spacing = float(constriction_spacing)
        self.constriction_probability = float(constriction_probability)
        self.active_center_indices_by_edge = active_center_indices_by_edge
        self._rng = resolve_generator(rng, seed)
        self._total_sites = 0
        self._active_sites = 0
        self._active_indices_by_edge: dict[str, list[int]] = {}

    def centers_for_edge(
        self,
        u: Any,
        v: Any,
        key: Any,
        edge_data: dict[str, Any],
        *,
        length: float,
    ) -> list[float]:
        if is_capillary_branch_order(str(edge_data.get("branch_order"))):
            all_centers = _periodic_center_positions(
                length=length,
                constriction_length=self.constriction_length,
                constriction_spacing=self.constriction_spacing,
            )
        else:
            # Rule: pericyte placement/assignment is capillary-only.
            all_centers = []
        self._total_sites += int(len(all_centers))

        edge_id = f"{u}|{v}|{key}"
        if self.active_center_indices_by_edge is not None:
            active_indices = validate_active_pericyte_indices(
                self.active_center_indices_by_edge.get(edge_id, []),
                total_pericytes=len(all_centers),
            )
        else:
            active_indices = select_active_pericyte_indices(
                total_pericytes=len(all_centers),
                constriction_probability=self.constriction_probability,
                rng=self._rng,
            )
        active_centers = [all_centers[ii] for ii in active_indices]
        self._active_sites += int(len(active_centers))
        self._active_indices_by_edge[edge_id] = [int(ii) for ii in active_indices]
        return active_centers

    def summary(self) -> dict[str, Any]:
        return {
            "total_periodic_pericyte_sites": self._total_sites,
            "active_periodic_pericyte_sites": self._active_sites,
            "constriction_probability": self.constriction_probability,
            "active_center_indices_by_edge": self._active_indices_by_edge,
        }


def set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    prefer_edge_fwhm_baseline: bool = False,
    constriction_length: float = 40.0,
    constriction_spacing: float = 100.0,
    constriction_probability: float = 1.0,
    active_center_indices_by_edge: dict[str, list[int]] | None = None,
    num_integration_points: int = 1000,
    viscosity_law: str = "pries",
    haematocrit: float = DEFAULT_HAEMATOCRIT,
    diameter_basis: str = "plasma_column",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """Apply periodic constrictions with random per-center activation.

    If ``active_center_indices_by_edge`` is provided, those fixed center indices
    are used per edge (no re-sampling). Edge keys use format ``"u|v|key"``.

    ``viscosity_law``, ``haematocrit`` and ``diameter_basis`` select the
    apparent-viscosity law the resistances are computed with; see
    :mod:`haemolynx.haemodynamics.viscosity`.
    """
    require_enough_integration_points(num_integration_points)
    if constriction_spacing <= 0:
        raise ValueError(
            f"constriction_spacing must be > 0, got {constriction_spacing}."
        )
    require_positive_constriction_length(constriction_length)
    if not (0.0 <= float(constriction_probability) <= 1.0):
        raise ValueError(
            "constriction_probability must be in [0, 1], "
            f"got {constriction_probability}."
        )

    sites = PeriodicConstrictionSites(
        constriction_length=constriction_length,
        constriction_spacing=constriction_spacing,
        constriction_probability=constriction_probability,
        active_center_indices_by_edge=active_center_indices_by_edge,
        rng=rng,
        seed=seed,
    )
    return apply_constriction_sites(
        graph,
        sites,
        diameter_by_branch_order=diameter_by_branch_order,
        constriction_factor_by_branch_order=constriction_factor_by_branch_order,
        prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
        constriction_length=constriction_length,
        num_integration_points=num_integration_points,
        viscosity_law=viscosity_law,
        haematocrit=haematocrit,
        diameter_basis=diameter_basis,
    )
