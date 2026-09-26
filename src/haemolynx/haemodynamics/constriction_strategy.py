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
from typing import Any, Mapping

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


def constriction_factor_for_order(
    branch_order: str,
    constriction_factor_by_branch_order: Mapping[str, float] | None,
    *,
    default_factor: float = 1.0,
) -> float:
    """Effective constriction factor for one branch order.

    A key present in *constriction_factor_by_branch_order* **replaces**
    *default_factor* for that order only (not multiplied). Orders absent from
    the map keep *default_factor* (``1.0`` = no narrowing; ``0.8`` = 20%
    narrower at focal sites).
    """
    order = str(branch_order)
    if (
        constriction_factor_by_branch_order is not None
        and order in constriction_factor_by_branch_order
    ):
        return float(constriction_factor_by_branch_order[order])
    return float(default_factor)


def resolve_constriction_factor_table(
    diameter_by_branch_order: Mapping[Any, Any],
    constriction_factor_by_branch_order: Mapping[str, float] | None,
    *,
    default_factor: float = 1.0,
) -> dict[str, float]:
    """Complete per-order factor table: map entries replace *default_factor*."""
    return {
        str(branch_order): constriction_factor_for_order(
            str(branch_order),
            constriction_factor_by_branch_order,
            default_factor=default_factor,
        )
        for branch_order in diameter_by_branch_order
    }


def constriction_strategy_kwargs(
    settings: Mapping[str, Any],
    *,
    diameter_by_branch_order: Mapping[str, float],
    constriction_length: float | None = None,
    constriction_spacing: float | None = None,
) -> dict[str, Any]:
    """Every :func:`set_resistances_for_constriction_strategy` argument a
    perturbation takes from its merged *settings* -- one place, so a pericyte
    diameter change and every pericyte sweep configure the strategy alike.

    *constriction_length*/*constriction_spacing* override the settings' own
    values for a geometry sweep's grid point.
    """
    probability = settings.get("pericyte_constriction_probability")
    factor = settings.get("pericyte_constriction_factor")
    return dict(
        diameter_by_branch_order=dict(diameter_by_branch_order),
        constriction_factor_by_branch_order=settings.get("constriction_by_branch_order"),
        use_pericyte_mask_constriction=bool(settings.get("use_pericyte_mask_constriction", False)),
        use_probabilistic_constriction=bool(
            settings.get("use_probabilistic_pericyte_constriction", False)
        ),
        prefer_edge_fwhm_baseline=bool(settings.get("use_fwhm_edge_diameters", False)),
        constriction_length=float(
            settings.get("constriction_length_um", 40.0)
            if constriction_length is None
            else constriction_length
        ),
        constriction_spacing=float(
            settings.get("constriction_spacing_um", 100.0)
            if constriction_spacing is None
            else constriction_spacing
        ),
        viscosity_law=settings.get("viscosity_law", "pries"),
        haematocrit=float(settings.get("haematocrit", 0.45)),
        diameter_basis=settings.get("diameter_basis", "plasma_column"),
        constriction_probability=1.0 if probability is None else float(probability),
        default_constriction_factor=1.0 if factor is None else float(factor),
        pericyte_mask_path=settings.get("pericyte_mask_path"),
        pericyte_mask_h5_dataset_name=settings.get("pericyte_mask_h5_dataset_name"),
        max_assignment_distance_um=settings.get("pericyte_max_assignment_distance_um", 3.0),
        min_pericyte_diameter_um=settings.get("pericyte_min_diameter_um", 5.0),
        max_pericyte_diameter_um=settings.get("pericyte_max_diameter_um", 12.0),
        axis_order=settings.get("image_axis_order", CANONICAL_AXIS_ORDER),
        seed=settings.get("pericyte_constriction_seed"),
    )


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
    viscosity_law: str = "pries",
    haematocrit: float = 0.45,
    diameter_basis: str = "plasma_column",
    constriction_probability: float = 1.0,
    default_constriction_factor: float = 1.0,
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

    All three strategies read ``viscosity_law``, ``haematocrit`` and
    ``diameter_basis``: the choice of law changes every resistance, and it must
    not depend on which strategy placed the constrictions.

    *default_constriction_factor* (settings: ``pericyte_constriction_factor``)
    applies to every branch order; keys in *constriction_factor_by_branch_order*
    replace that base for the listed orders only.
    """
    # Every order on the graph, not only the table's: a vessel past
    # max_branch_order still has its own diameter, and it takes the default
    # factor like any order the map does not name -- rather than failing the
    # whole strategy ("No constriction factor for branch_order 'B59'").
    orders = dict.fromkeys(diameter_by_branch_order)
    orders.update(
        dict.fromkeys(
            str(data["branch_order"])
            for _u, _v, data in graph.edges(data=True)
            if data.get("branch_order") is not None
        )
    )
    factors = resolve_constriction_factor_table(
        orders,
        constriction_factor_by_branch_order,
        default_factor=float(default_constriction_factor),
    )
    if use_pericyte_mask_constriction:
        if pericyte_mask_path is None:
            raise ValueError(
                "pericyte_mask_path must be set when use_pericyte_mask_constriction=True."
            )
        graph, results = pericyte_mask_strategy.set_poiseuille_resistances_with_pericyte_mask(
            graph,
            diameter_by_branch_order=diameter_by_branch_order,
            constriction_factor_by_branch_order=factors,
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
            viscosity_law=viscosity_law,
            haematocrit=float(haematocrit),
            diameter_basis=diameter_basis,
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
                constriction_factor_by_branch_order=factors,
                prefer_edge_fwhm_baseline=prefer_edge_fwhm_baseline,
                constriction_length=float(constriction_length),
                constriction_spacing=float(constriction_spacing),
                constriction_probability=float(constriction_probability),
                active_center_indices_by_edge=active_center_indices_by_edge,
                viscosity_law=viscosity_law,
                haematocrit=float(haematocrit),
                diameter_basis=diameter_basis,
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
        diameter_basis=diameter_basis,
    )
    if prefer_edge_fwhm_baseline:
        graph, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
            graph,
            diameter_by_branch_order,
            prefer_edge_fwhm_baseline=True,
            constriction_factor_by_branch_order=factors,
        )
        return graph, PERIODIC_CONSTRICTION_STRATEGY, results

    constricted_diameters = {
        branch_order: {
            "d1": float(diameter),
            "d2": float(diameter) * float(factors[str(branch_order)]),
        }
        for branch_order, diameter in diameter_by_branch_order.items()
    }
    graph, results = poiseuille_model.set_poiseuille_resistances_with_constrictions(
        graph,
        constricted_diameters,
    )
    return graph, PERIODIC_CONSTRICTION_STRATEGY, results
