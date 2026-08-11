"""Choosing which constriction strategy a run uses.

Four settings decide how a run narrows its vessels — whether pericytes come from
a mask, whether each one contracts by chance, whether baseline diameters are
measured per edge, and the periodic fallback when none of that applies. The
choice was made in two places that had to agree: once for the pipeline's own
run, and once per scenario inside the baseline-versus-constricted comparison.
It is made here instead, so the comparison cannot drift away from the run it is
supposed to be comparing.

The strategies are called through their modules rather than through imported
names, so a test that patches, say,
``haemodynamics.probability.set_poiseuille_resistances_with_probabilistic_periodic_constrictions``
still intercepts every call site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER

from .constriction import resolve_generator
from . import pericyte_mask as pericyte_mask_strategy
from . import probability as probability_strategy
from .poiseuille import PoiseuilleModel

#: Result keys, one per strategy, so a run summary says which model produced it.
PERICYTE_MASK_STRATEGY = "pericyte_mask"
PROBABILISTIC_STRATEGY = "probabilistic"
PERIODIC_CONSTRICTION_STRATEGY = "constrictions"


def uniform_constriction_factors(
    diameter_by_branch_order: dict,
    factor_value: float,
) -> dict[str, float]:
    """One constriction factor applied to every branch order.

    A comparison scenario overrides the configured per-branch-order factors with
    a single value, so that the only difference between its two graphs is that
    value.
    """
    return {
        str(branch_order): float(factor_value)
        for branch_order in diameter_by_branch_order
    }


def set_resistances_for_constriction_strategy(
    graph: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict,
    constriction_factor_by_branch_order: dict[str, float] | None,
    use_pericyte_mask_constriction: bool,
    use_probabilistic_constriction: bool,
    prefer_edge_fwhm_baseline: bool,
    constriction_length: float,
    constriction_spacing: float,
    viscosity_law: str = "capillary_power_law",
    haematocrit: float = 0.45,
    constriction_probability: float = 1.0,
    pericyte_mask_path: str | Path | None = None,
    pericyte_mask_h5_dataset_name: str | None = None,
    active_pericyte_indices: list[int] | None = None,
    active_center_indices_by_edge: dict[str, list[int]] | None = None,
    max_assignment_distance_um: float | None = 3.0,
    min_pericyte_diameter_um: float | None = 5.0,
    max_pericyte_diameter_um: float | None = 12.0,
    axis_order: str = CANONICAL_AXIS_ORDER,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[nx.MultiGraph, str, dict[str, Any]]:
    """Apply one constriction strategy to ``graph``.

    Returns ``(graph, strategy, results)``, where *strategy* names the model that
    ran so a caller can file the summary under it.

    Both randomised strategies draw their cohort from ``rng`` when given, else
    from a generator built on ``seed``, so a run repeats for a given seed.
    """
    if use_pericyte_mask_constriction:
        if pericyte_mask_path is None:
            raise ValueError(
                "pericyte_mask_path must be set when use_pericyte_mask_constriction=True."
            )
        graph, results = pericyte_mask_strategy.set_poiseuille_resistances_with_pericyte_mask(
            graph,
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=constriction_factor_by_branch_order,
            pericyte_mask_path=pericyte_mask_path,
            pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
            prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
            constriction_length=constriction_length,
            use_probabilistic_constriction=use_probabilistic_constriction,
            constriction_probability=float(constriction_probability),
            active_pericyte_indices=active_pericyte_indices,
            max_assignment_distance_um=max_assignment_distance_um,
            min_pericyte_diameter_um=min_pericyte_diameter_um,
            max_pericyte_diameter_um=max_pericyte_diameter_um,
            axis_order=axis_order,
            rng=rng,
            seed=seed,
        )
        return graph, PERICYTE_MASK_STRATEGY, results

    if use_probabilistic_constriction:
        graph, results = (
            probability_strategy
            .set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
                graph,
                diameter_by_branch_order=diameter_by_branch_order,
                constriction_factor_by_branch_order=constriction_factor_by_branch_order,
                prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
                constriction_length=float(constriction_length),
                constriction_spacing=float(constriction_spacing),
                constriction_probability=float(constriction_probability),
                active_center_indices_by_edge=active_center_indices_by_edge,
                rng=rng,
                seed=seed,
            )
        )
        return graph, PROBABILISTIC_STRATEGY, results

    poiseuille_model = PoiseuilleModel(
        constriction_length=float(constriction_length),
        constriction_spacing=float(constriction_spacing),
        viscosity_law=viscosity_law,
        haematocrit=float(haematocrit),
    )
    if prefer_edge_fwhm_baseline:
        graph, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
            graph,
            diameter_by_branch_order,
            prefer_edge_fwhm_baseline=True,
            constriction_factor_by_branch_order=constriction_factor_by_branch_order,
        )
        return graph, PERIODIC_CONSTRICTION_STRATEGY, results

    factors = constriction_factor_by_branch_order or {}
    constricted_diameters = {
        branch_order: {
            "d1": float(diameter),
            "d2": float(diameter) * float(factors.get(branch_order, 1.0)),
        }
        for branch_order, diameter in diameter_by_branch_order.items()
    }
    graph, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
        graph,
        constricted_diameters,
    )
    return graph, PERIODIC_CONSTRICTION_STRATEGY, results
