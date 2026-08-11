"""High-level haemodynamics steps for vascular graphs."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from haemolynx import io
from haemolynx.io.axis_order import CANONICAL_AXIS_ORDER
from haemolynx.parsers import prefixed_arguments
from haemolynx.haemodynamics import automated
from haemolynx.haemodynamics.poiseuille import PoiseuilleModel
from haemolynx.haemodynamics.viscosity import describe_law
from haemolynx.haemodynamics import pericyte_comparison as pericyte_comparison_haemodynamics
from haemolynx.haemodynamics.constriction import resolve_generator
from haemolynx.haemodynamics.constriction_strategy import (
    set_resistances_for_constriction_strategy,
)

logger = logging.getLogger(__name__)


#: FWHM settings are named `fwhm_<parameter>` in the config, matching the
#: measurement function's parameters one for one.
FWHM_SETTING_PREFIX = "fwhm_"

#: Seed used for the probabilistic pericyte cohort when the settings do not
#: name one, so a run made through the library API repeats by default. The
#: config file's ``pericyte_constriction_seed`` overrides it, and setting that
#: to null asks for a fresh cohort each run.
DEFAULT_PERICYTE_CONSTRICTION_SEED = 20240917

#: Values the constriction model needs that are not config settings today. They
#: were defaults on this dataclass before the settings arrived as a group, and
#: are kept here so behaviour is unchanged and the numbers stay findable.
DIAMETER_DEFAULTS: dict[str, Any] = {
    "custom_edge_diameter": 6.0,
    "constriction_length": 40.0,
    "constriction_spacing": 100.0,
    "constriction_by_branch_order": {},
    "custom_edges": [],
    "viscosity_law": "pries",
    "diameter_basis": "plasma_column",
    "haematocrit": 0.45,
}


@dataclass
class HaemodynamicsApplyConfig:
    """Settings for Poiseuille conductance assignment on a vascular graph.

    The two large groups arrive as dicts rather than as forty-odd separate
    fields, keyed exactly as they are in the config file so a value can be
    traced from YAML to here by name:

    ``diameters``
        The ``diameters_and_pericytes`` section — branch-order diameters, the
        constriction factors, custom edges, and every pericyte setting.
    ``fwhm``
        The ``fwhm_diameter_measurement`` section — whether to measure diameters
        from the raw image, and the fitting parameters if so.

    Anything a run computes rather than configures stays an ordinary field.
    """

    diameters: dict[str, Any] = field(default_factory=dict)
    fwhm: dict[str, Any] = field(default_factory=dict)

    # Computed per run, not configured.
    comparison_output_csv_path: Path | None = None
    resistance_node_pair: tuple[int, int] | None = None
    #: Generator for the probabilistic pericyte cohort. Wins over the seed in
    #: ``diameters``, for a caller driving several runs off one stream.
    rng: np.random.Generator | None = None
    #: Spacing per array axis in canonical (z, y, x) order, not image-metadata (x, y, z).
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    axis_order: str = CANONICAL_AXIS_ORDER

    def __post_init__(self) -> None:
        if not self.diameters.get("diameter_by_branch_order"):
            raise ValueError(
                "HaemodynamicsApplyConfig needs 'diameter_by_branch_order' in its "
                "diameters settings; pass the diameters_and_pericytes section of "
                "the config."
            )

    def diameter(self, name: str, default: Any = None) -> Any:
        """One value from the diameters/pericytes group.

        Falls back to :data:`DIAMETER_DEFAULTS` for the few values the config
        does not carry, then to *default*.
        """
        if name in self.diameters and self.diameters[name] is not None:
            return self.diameters[name]
        if default is not None:
            return default
        return DIAMETER_DEFAULTS.get(name)

    @property
    def pericyte_constriction_seed(self) -> int | None:
        """Seed for the probabilistic pericyte cohort; ``None`` means unseeded.

        A settings group that says nothing about the seed gets
        :data:`DEFAULT_PERICYTE_CONSTRICTION_SEED`, because a run that cannot be
        repeated is the worse default. An explicit null asks for a fresh cohort.
        """
        if "pericyte_constriction_seed" not in self.diameters:
            return DEFAULT_PERICYTE_CONSTRICTION_SEED
        seed = self.diameters["pericyte_constriction_seed"]
        return None if seed is None else int(seed)

    def pericyte_rng(self) -> np.random.Generator:
        """The generator every pericyte draw in this run comes from."""
        return resolve_generator(self.rng, self.pericyte_constriction_seed)

    def fwhm_setting(self, name: str, default: Any = None) -> Any:
        """One value from the FWHM group, named as it is in the config."""
        return self.fwhm.get(name, default)

    @property
    def use_fwhm_edge_diameters(self) -> bool:
        return bool(self.fwhm.get("use_fwhm_edge_diameters", False))

    @property
    def do_pericyte_constriction(self) -> bool:
        return bool(self.diameters.get("do_pericyte_construction", False))

    def fwhm_measurement_arguments(self, valid_parameters: Iterable[str]) -> dict[str, Any]:
        """FWHM settings as measurement-function arguments."""
        return prefixed_arguments(self.fwhm, FWHM_SETTING_PREFIX, valid_parameters)


def _measure_fwhm_diameters(G: nx.MultiGraph, config: HaemodynamicsApplyConfig) -> dict[str, Any]:
    raw_tiff_path = config.fwhm_setting("fwhm_raw_tiff_path")
    if raw_tiff_path is None:
        raise ValueError("use_fwhm_edge_diameters=True requires fwhm_raw_tiff_path.")
    voxel_sz = tuple(
        float(v) for v in G.graph.get("image_voxel_size_zyx", config.voxel_size_zyx)
    )
    measurement_parameters = inspect.signature(
        automated.measure_edge_diameters_fwhm_from_raw_tiff
    ).parameters
    return automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        voxel_size_zyx=voxel_sz,
        axis_order=config.axis_order,
        **{
            **config.fwhm_measurement_arguments(measurement_parameters),
            "raw_tiff_path": io.resolve_image_path_with_optional_zip(Path(raw_tiff_path)),
        },
    )


def _run_pericyte_comparison(
    G: nx.MultiGraph,
    config: HaemodynamicsApplyConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[list[int] | None, dict[str, list[int]] | None, dict[str, Any]]:
    if not config.diameter("run_pericyte_resistance_comparison"):
        return None, None, {}
    if config.comparison_output_csv_path is None:
        raise ValueError("comparison_output_csv_path required for pericyte comparison.")
    if config.resistance_node_pair is None:
        raise ValueError("resistance_node_pair required for pericyte comparison.")

    comparison_results = pericyte_comparison_haemodynamics.compare_baseline_vs_pericyte_constriction(
        G,
        diameter_by_branch_order=config.diameter("diameter_by_branch_order"),
        constriction_factor_by_branch_order=config.diameter("constriction_by_branch_order"),
        resistance_node_pair=config.resistance_node_pair,
        output_csv_path=config.comparison_output_csv_path,
        baseline_factor_value=float(config.diameter("pericyte_comparison_baseline_value")),
        constricted_factor_value=float(config.diameter("pericyte_comparison_constricted_value")),
        use_pericyte_mask_constriction=bool(config.diameter("use_pericyte_mask_constriction")),
        pericyte_mask_path=config.diameter("pericyte_mask_path"),
        pericyte_mask_h5_dataset_name=config.diameter("pericyte_mask_h5_dataset_name"),
        max_assignment_distance_um=config.diameter("pericyte_max_assignment_distance_um"),
        min_pericyte_diameter_um=config.diameter("pericyte_min_diameter_um"),
        max_pericyte_diameter_um=config.diameter("pericyte_max_diameter_um"),
        prefer_edge_fwhm_baseline=bool(config.use_fwhm_edge_diameters),
        constriction_length=config.diameter("constriction_length"),
        constriction_spacing=config.diameter("constriction_spacing"),
        use_probabilistic_pericyte_constriction=bool(config.diameter("use_probabilistic_pericyte_constriction")),
        pericyte_constriction_probability=float(config.diameter("pericyte_constriction_probability")),
        axis_order=config.axis_order,
        rng=rng,
    )

    active_pericyte_indices: list[int] | None = None
    active_center_indices_by_edge: dict[str, list[int]] | None = None
    if (
        config.diameter("reuse_comparison_pericyte_cohort_for_main_run")
        and config.diameter("use_probabilistic_pericyte_constriction")
    ):
        if config.diameter("use_pericyte_mask_constriction"):
            selected = comparison_results.get("active_pericyte_indices")
            active_pericyte_indices = [int(idx) for idx in selected] if selected else []
        else:
            selected_map = comparison_results.get("active_center_indices_by_edge")
            if isinstance(selected_map, dict):
                active_center_indices_by_edge = {
                    str(edge_id): [int(idx) for idx in idx_list]
                    for edge_id, idx_list in selected_map.items()
                }
    return active_pericyte_indices, active_center_indices_by_edge, comparison_results


def _assign_poiseuille_resistances(
    G: nx.MultiGraph,
    config: HaemodynamicsApplyConfig,
    *,
    active_pericyte_indices: list[int] | None,
    active_center_indices_by_edge: dict[str, list[int]] | None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    poiseuille_model = PoiseuilleModel(
        constriction_length=config.diameter("constriction_length"),
        constriction_spacing=config.diameter("constriction_spacing"),
        viscosity_law=config.diameter("viscosity_law"),
        haematocrit=config.diameter("haematocrit"),
        diameter_basis=config.diameter("diameter_basis"),
    )
    results: dict[str, Any] = {}

    if config.do_pericyte_constriction:
        reuse_comparison_cohort = bool(
            config.diameter("reuse_comparison_pericyte_cohort_for_main_run")
            and config.diameter("use_probabilistic_pericyte_constriction")
        )
        # Only the probabilistic strategies read this, and a run that uses
        # neither need not configure it.
        configured_probability = config.diameter("pericyte_constriction_probability")
        G, strategy, strategy_results = set_resistances_for_constriction_strategy(
            G,
            diameter_by_branch_order=config.diameter("diameter_by_branch_order"),
            constriction_factor_by_branch_order=config.diameter("constriction_by_branch_order"),
            use_pericyte_mask_constriction=bool(config.diameter("use_pericyte_mask_constriction")),
            use_probabilistic_constriction=bool(config.diameter("use_probabilistic_pericyte_constriction")),
            prefer_edge_fwhm_baseline=bool(config.use_fwhm_edge_diameters),
            constriction_length=config.diameter("constriction_length"),
            constriction_spacing=config.diameter("constriction_spacing"),
            viscosity_law=config.diameter("viscosity_law"),
            haematocrit=config.diameter("haematocrit"),
            diameter_basis=config.diameter("diameter_basis"),
            constriction_probability=(
                1.0 if configured_probability is None else float(configured_probability)
            ),
            pericyte_mask_path=config.diameter("pericyte_mask_path"),
            pericyte_mask_h5_dataset_name=config.diameter("pericyte_mask_h5_dataset_name"),
            active_pericyte_indices=active_pericyte_indices if reuse_comparison_cohort else None,
            active_center_indices_by_edge=(
                active_center_indices_by_edge if reuse_comparison_cohort else None
            ),
            max_assignment_distance_um=config.diameter("pericyte_max_assignment_distance_um"),
            min_pericyte_diameter_um=config.diameter("pericyte_min_diameter_um"),
            max_pericyte_diameter_um=config.diameter("pericyte_max_diameter_um"),
            axis_order=config.axis_order,
            rng=rng,
        )
        results[strategy] = strategy_results
    else:
        G, results["poiseuille"] = poiseuille_model.set_poiseuille_resistances(
            G,
            config.diameter("diameter_by_branch_order"),
            prefer_edge_fwhm_diameter=bool(config.use_fwhm_edge_diameters),
        )

    G, results["custom_edges"] = poiseuille_model.set_poiseuille_edge_resistances(
        G,
        config.diameter("custom_edges"),
        edge_diameter=config.diameter("custom_edge_diameter"),
    )
    # Which law produced these resistances travels with them. They are not
    # comparable across laws -- several times apart in the smallest vessels --
    # so a graph pickled today and read next month has to say which it was.
    G.graph["viscosity_law"] = poiseuille_model.viscosity_law
    G.graph["haematocrit"] = poiseuille_model.haematocrit
    G.graph["diameter_basis"] = poiseuille_model.diameter_basis
    return results


def apply_poiseuille_haemodynamics(
    G: nx.MultiGraph,
    *,
    diameter_by_branch_order: dict[str, float] | None = None,
    constriction_by_branch_order: dict[str, float] | None = None,
    custom_edges: list | dict | None = None,
    config: HaemodynamicsApplyConfig | None = None,
) -> tuple[nx.MultiGraph, dict[str, Any]]:
    """
    Assign Poiseuille edge conductances on ``G``.

    For the simple tutorial path, pass ``diameter_by_branch_order`` and optional
    ``custom_edges``. For the full example-pipeline path, pass a
    :class:`HaemodynamicsApplyConfig` via ``config`` (other kwargs are ignored).

    Returns
    -------
    tuple
        ``(G, results)`` where ``results`` summarizes FWHM, pericyte, and resistance steps.
    """
    if config is None:
        if diameter_by_branch_order is None:
            raise ValueError(
                "diameter_by_branch_order is required when config is not provided."
            )
        config = HaemodynamicsApplyConfig(
            diameters={
                "diameter_by_branch_order": diameter_by_branch_order,
                "constriction_by_branch_order": constriction_by_branch_order or {},
                "custom_edges": custom_edges or [],
            }
        )
    summary: dict[str, Any] = {}

    if config.use_fwhm_edge_diameters:
        summary["fwhm"] = _measure_fwhm_diameters(G, config)

    pericyte_rng = config.pericyte_rng()
    active_pericyte_indices, active_center_indices_by_edge, comparison_results = (
        _run_pericyte_comparison(G, config, rng=pericyte_rng)
    )
    if comparison_results:
        summary["pericyte_comparison"] = comparison_results

    summary["resistances"] = _assign_poiseuille_resistances(
        G,
        config,
        active_pericyte_indices=active_pericyte_indices,
        active_center_indices_by_edge=active_center_indices_by_edge,
        rng=pericyte_rng,
    )
    # Top level rather than beside a step's counters: it describes every
    # resistance in the graph, and they are not comparable across laws.
    summary["viscosity"] = describe_law(
        G.graph["viscosity_law"],
        G.graph["haematocrit"],
        G.graph["diameter_basis"],
    )

    return G, summary
