"""Focal constrictions inside pericyte dilation / combined sweeps.

``run_pericyte_dilation_pressure_sweep`` used to dilate diameters and assign
uniform Poiseuille resistances, so entry ``constriction_length_um`` /
``constriction_spacing_um`` / probability settings were inert for numerics.
These tests pin that dilation sweeps now go through the constriction strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haemolynx.haemodynamics.pericyte_sweep import (  # noqa: E402
    run_pericyte_dilation_pressure_sweep,
)

#: Long enough for several periodic sites at 50–200 µm spacing.
_LONG_EDGE_UM = 600.0


def _build_long_capillary_network() -> tuple[
    nx.MultiGraph, list[int], list[int], dict[str, float]
]:
    """A chain long enough that constriction length/spacing change resistance."""
    G = nx.MultiGraph()
    orders = ("Art1", "B01", "B01", "Ven1")
    diameters = {"Art1": 7.0, "B01": 5.0, "Ven1": 8.0}
    for node in range(len(orders) + 1):
        G.add_node(
            node, pos=np.asarray([0.0, 0.0, node * _LONG_EDGE_UM], dtype=float)
        )
    for node, branch_order in enumerate(orders):
        G.add_edge(
            node,
            node + 1,
            key=0,
            length=_LONG_EDGE_UM,
            branch_order=branch_order,
            fwhm_diameter_um=diameters[branch_order],
            voxels=[
                (0.0, 0.0, node * _LONG_EDGE_UM),
                (0.0, 0.0, (node + 1) * _LONG_EDGE_UM),
            ],
        )
    return G, [0], [len(orders)], diameters


def _base_settings(diameters: dict[str, float], **overrides) -> dict:
    settings = {
        "diameter_by_branch_order": diameters,
        "constriction_by_branch_order": {
            "Art1": 1.0,
            "B01": 0.5,
            "Ven1": 1.0,
        },
        "outlet_p_bc": 1000.0,
        "inlet_p_bc": 4500.0,
        "viscosity_law": "constant",
        "haematocrit": 0.45,
        "diameter_basis": "plasma_column",
        "use_fwhm_edge_diameters": False,
        "use_pericyte_mask_constriction": False,
        "use_probabilistic_pericyte_constriction": False,
        "constriction_length_um": 40.0,
        "constriction_spacing_um": 100.0,
        "pericyte_dilation_min_percent": 0,
        "pericyte_dilation_max_percent": 0,
        "pericyte_dilation_step_percent": 1,
        "inlet_pressure_min_pa": 4500,
        "inlet_pressure_max_pa": 4500,
        "inlet_pressure_step_pa": 500,
    }
    settings.update(overrides)
    return settings


def _first_resistance(sweep: dict) -> float:
    return float(sweep["results"][0]["equivalent_resistance"])


def test_dilation_sweep_spacing_changes_equivalent_resistance(tmp_path: Path):
    """constriction_spacing_um is no longer inert on a pericyte dilation sweep."""
    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    sparse = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_spacing_um=200.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "sparse",
        sweep_dilation=True,
        sweep_pressure=False,
    )
    dense = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_spacing_um=50.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "dense",
        sweep_dilation=True,
        sweep_pressure=False,
    )
    assert _first_resistance(dense) > _first_resistance(sparse)


def test_pressure_and_pericyte_sweep_length_changes_resistance(tmp_path: Path):
    """constriction_length_um moves a combined dilation×pressure sweep metric."""
    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    shorter = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_length_um=10.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "short",
        sweep_dilation=True,
        sweep_pressure=True,
    )
    longer = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_length_um=80.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "long",
        sweep_dilation=True,
        sweep_pressure=True,
    )
    assert _first_resistance(longer) > _first_resistance(shorter)


def test_two_spacing_settings_produce_different_sweep_results(tmp_path: Path):
    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    results = []
    for spacing in (200.0, 50.0):
        sweep = run_pericyte_dilation_pressure_sweep(
            G,
            _base_settings(diameters, constriction_spacing_um=spacing),
            inlet_nodes=inlet_nodes,
            outlet_nodes=outlet_nodes,
            output_dir=tmp_path / f"spacing_{int(spacing)}",
            sweep_dilation=True,
            sweep_pressure=False,
        )
        results.append(_first_resistance(sweep))
    assert results[0] != results[1]


def test_probabilistic_dilation_sweep_honours_seed_and_probability(tmp_path: Path):
    from haemolynx.haemodynamics.probability import (
        set_poiseuille_resistances_with_probabilistic_periodic_constrictions,
    )

    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    factors = {"Art1": 1.0, "B01": 0.5, "Ven1": 1.0}

    def _sites(*, probability: float, seed: int) -> int:
        _graph, summary = (
            set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
                G.copy(),
                diameter_by_branch_order=diameters,
                constriction_factor_by_branch_order=factors,
                prefer_edge_fwhm_baseline=False,
                constriction_length=40.0,
                constriction_spacing=100.0,
                constriction_probability=probability,
                viscosity_law="constant",
                seed=seed,
            )
        )
        return int(summary["active_periodic_pericyte_sites"])

    assert _sites(probability=1.0, seed=7) == _sites(probability=1.0, seed=7)
    assert _sites(probability=1.0, seed=7) > _sites(probability=0.1, seed=7)

    full = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(
            diameters,
            use_probabilistic_pericyte_constriction=True,
            pericyte_constriction_probability=1.0,
            pericyte_constriction_seed=3,
        ),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "prob_full",
        sweep_dilation=True,
        sweep_pressure=False,
    )
    sparse = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(
            diameters,
            use_probabilistic_pericyte_constriction=True,
            pericyte_constriction_probability=0.1,
            pericyte_constriction_seed=3,
        ),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "prob_sparse",
        sweep_dilation=True,
        sweep_pressure=False,
    )
    assert _first_resistance(full) > _first_resistance(sparse)


def test_pressure_only_sweep_does_not_apply_focal_constrictions(tmp_path: Path):
    """sweep_dilation=False stays uniform Poiseuille; spacing knobs stay inert."""
    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    sparse = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_spacing_um=200.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "p_sparse",
        sweep_dilation=False,
        sweep_pressure=True,
    )
    dense = run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(diameters, constriction_spacing_um=50.0),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path / "p_dense",
        sweep_dilation=False,
        sweep_pressure=True,
    )
    assert _first_resistance(sparse) == _first_resistance(dense)


def test_dilation_sweep_does_not_mutate_baseline_geometry(tmp_path: Path):
    """Dilate/constrict rebind attributes; voxels and pos stay shared-safe."""
    G, inlet_nodes, outlet_nodes, diameters = _build_long_capillary_network()
    pos_before = {
        node: np.asarray(data["pos"], dtype=float).copy()
        for node, data in G.nodes(data=True)
    }
    voxels_before = {
        (u, v, key): np.asarray(data["voxels"], dtype=float).copy()
        for u, v, key, data in G.edges(keys=True, data=True)
    }
    voxels_ids = {
        (u, v, key): id(data["voxels"])
        for u, v, key, data in G.edges(keys=True, data=True)
    }
    run_pericyte_dilation_pressure_sweep(
        G,
        _base_settings(
            diameters,
            constriction_spacing_um=50.0,
            pericyte_dilation_max_percent=10,
            pericyte_dilation_step_percent=10,
        ),
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        output_dir=tmp_path,
        sweep_dilation=True,
        sweep_pressure=False,
    )
    for node, data in G.nodes(data=True):
        np.testing.assert_array_equal(
            np.asarray(data["pos"], dtype=float), pos_before[node]
        )
    for u, v, key, data in G.edges(keys=True, data=True):
        edge = (u, v, key)
        np.testing.assert_array_equal(
            np.asarray(data["voxels"], dtype=float), voxels_before[edge]
        )
        assert id(data["voxels"]) == voxels_ids[edge]
