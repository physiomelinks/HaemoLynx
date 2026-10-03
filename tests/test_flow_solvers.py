"""The flow and equivalent-resistance solver options, and both resistances.

``haemodynamics_solver`` picks how every flow solve holds the network: a dense
N x N array (the default, as it always was) or a sparse matrix of the vessels
alone, factorised by sparse LU. ``equivalent_resistance_solver`` picks how the
two-point resistance is found: every eigenpair of the dense Laplacian (the
default, as it always was) or one sparse solve. Either way the numbers must be
the same to rounding, the same errors must be raised and the same warnings
logged -- what changes is memory and time. The solve also reports the whole
network's resistance, pressure drop over total inflow, beside the two-point
one between the first inlet and the first outlet.
"""
from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from haemolynx.haemodynamics import (
    EQUIVALENT_RESISTANCE_SOLVERS,
    FLOW_SOLVERS,
    build_conductance_matrix_from_graph,
    calc_laplacian_from_conductance_matrix,
    calc_two_point_from_laplacian_matrix_nodeID,
    calc_two_point_resistance_sparse,
    flow_conservation_residuals,
    network_resistance,
    set_edge_flows,
    solve_flow_from_conductance_matrix,
)
from haemolynx.pipeline import default_schema

SI = 1e-16  # m^3/(Pa.s), a capillary's conductance
SCHEMA = default_schema()


def _capillary_bed(seed: int = 0) -> tuple[nx.MultiGraph, list[int], list[int]]:
    """A 6 x 6 x 3 lattice thinned to a capillary bed's ~3 vessels a node,
    with a parallel vessel, a vessel with no conductance and a conductive
    piece no boundary node reaches. Three inlets at one corner, three outlets
    at the opposite one."""
    rng = np.random.default_rng(seed)
    lattice = nx.grid_graph(dim=[6, 6, 3])
    tree = nx.minimum_spanning_tree(lattice)
    extra = [edge for edge in lattice.edges() if not tree.has_edge(*edge)]
    keep = rng.permutation(len(extra))[: len(extra) // 4]
    G = nx.MultiGraph()
    G.add_edges_from(tree.edges())
    G.add_edges_from(extra[i] for i in keep)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    for _u, _v, data in G.edges(data=True):
        data["conductance"] = SI * float(rng.uniform(0.5, 2.0))
    u, v = next(iter(G.edges()))
    G.add_edge(u, v, conductance=0.7 * SI)  # a parallel vessel
    n = G.number_of_nodes()
    G.add_edge(n, n + 1)  # a vessel that never got a conductance
    G.add_edge(n + 2, n + 3, conductance=SI)  # conductive, but no boundary reaches it
    lattice_nodes = sorted(range(n))
    return G, lattice_nodes[:3], lattice_nodes[-3:]


def _solve(G, inlets, outlets, solver, *, inlet_p_bc=1000.0, outlet_p_bc=500.0):
    conductance, node_list = build_conductance_matrix_from_graph(G, solver=solver)
    flow = solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        inlet_p_bc=inlet_p_bc,
        outlet_p_bc=outlet_p_bc,
        inlet_nodes=inlets,
        outlet_nodes=outlets,
    )
    return conductance, node_list, flow


# --- the conductance matrix ---------------------------------------------------


def test_the_sparse_matrix_holds_exactly_the_dense_conductances():
    G, _inlets, _outlets = _capillary_bed()
    dense, dense_nodes = build_conductance_matrix_from_graph(G)
    sparse, sparse_nodes = build_conductance_matrix_from_graph(G, solver="sparse")

    assert sp.issparse(sparse)
    assert sparse_nodes == dense_nodes
    np.testing.assert_array_equal(sparse.toarray(), dense)
    # Only the vessels are stored: two entries per conductive vessel pair.
    assert sparse.nnz < 0.05 * dense.size


def test_the_sparse_laplacian_is_the_dense_one():
    G, _inlets, _outlets = _capillary_bed()
    dense, _ = build_conductance_matrix_from_graph(G)
    sparse, _ = build_conductance_matrix_from_graph(G, solver="sparse")
    np.testing.assert_allclose(
        calc_laplacian_from_conductance_matrix(sparse).toarray(),
        calc_laplacian_from_conductance_matrix(dense),
        rtol=0,
        atol=1e-30,
    )


def test_an_asymmetric_sparse_matrix_is_refused_as_a_dense_one_is():
    matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
    for candidate in (matrix, sp.csr_matrix(matrix)):
        with pytest.raises(ValueError, match="must be symmetric"):
            calc_laplacian_from_conductance_matrix(candidate)


def test_the_sparse_solver_takes_no_dense_out_array_and_no_unknown_solver():
    G, _inlets, _outlets = _capillary_bed()
    n = G.number_of_nodes()
    with pytest.raises(ValueError, match="dense array"):
        build_conductance_matrix_from_graph(G, solver="sparse", out=np.zeros((n, n)))
    with pytest.raises(ValueError, match="Unknown flow solver"):
        build_conductance_matrix_from_graph(G, solver="cholesky")


# --- the flow solve -------------------------------------------------------------


def test_the_sparse_solve_gives_the_dense_pressures_and_conserves_flow():
    G, inlets, outlets = _capillary_bed()
    _c, dense_nodes, dense = _solve(G, inlets, outlets, "dense")
    _c, sparse_nodes, sparse = _solve(G, inlets, outlets, "sparse")

    assert sparse_nodes == dense_nodes
    np.testing.assert_allclose(sparse["pressure"], dense["pressure"], rtol=1e-10, atol=1e-9)
    n = G.number_of_nodes()
    # The piece no boundary reaches stays at 0 Pa, as the dense solve leaves it.
    idx = {node: i for i, node in enumerate(sparse_nodes)}
    assert sparse["pressure"][idx[n - 2]] == 0.0 == sparse["pressure"][idx[n - 1]]

    set_edge_flows(G, sparse_nodes, sparse["pressure"])
    max_flow = max(abs(d["flow_signed"]) for _u, _v, d in G.edges(data=True) if "flow_signed" in d)
    residuals = flow_conservation_residuals(G, boundary_nodes=inlets + outlets)
    assert max(abs(r) for r in residuals.values()) < 1e-9 * max_flow


def _disconnected():
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=SI)
    G.add_edge(2, 3, conductance=SI)
    return G, dict(inlet_nodes=[0], outlet_nodes=[3])


def _non_finite():
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=float("nan"))
    G.add_edge(1, 2, conductance=SI)
    return G, dict(inlet_nodes=[0], outlet_nodes=[2])


def _both_ends():
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=SI)
    G.add_edge(1, 2, conductance=SI)
    return G, dict(inlet_nodes=[0, 1], outlet_nodes=[1, 2])


@pytest.mark.parametrize("case", [_disconnected, _non_finite, _both_ends])
def test_the_sparse_solve_refuses_what_the_dense_one_refuses(case):
    G, boundaries = case()
    messages = []
    for solver in FLOW_SOLVERS:
        conductance, node_list = build_conductance_matrix_from_graph(G, solver=solver)
        with pytest.raises(ValueError) as raised:
            solve_flow_from_conductance_matrix(
                conductance, node_list, inlet_p_bc=1000.0, outlet_p_bc=500.0, **boundaries
            )
        messages.append(str(raised.value))
    assert messages[0] == messages[1]


def test_the_sparse_solve_warns_as_the_dense_one_does(caplog):
    """A conductive piece no boundary reaches, and a piece whose only boundary
    node is an inlet: both warnings, from either solver."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, conductance=SI)
    G.add_edge(1, 2, conductance=SI)
    G.add_edge(5, 6, conductance=SI)  # stranded
    G.add_edge(3, 4, conductance=SI)  # 4 is an inlet, nothing drains it
    logs = []
    for solver in FLOW_SOLVERS:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="haemolynx.haemodynamics.resistance"):
            _solve(G, [0, 4], [2], solver)
        logs.append(sorted(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING))
    assert logs[0] == logs[1]
    assert any("no conductive path to any boundary node" in m for m in logs[1])
    assert any("only reaches boundary nodes at 1000.0 Pa" in m for m in logs[1])


# --- the two-point resistance ---------------------------------------------------


def _chain(*conductances) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, conductance in enumerate(conductances):
        G.add_edge(node, node + 1, conductance=conductance)
    return G


def test_the_sparse_two_point_resistance_adds_series_vessels_and_parallel_ones():
    series = _chain(1.0 * SI, 0.5 * SI)
    C, nodes = build_conductance_matrix_from_graph(series, solver="sparse")
    assert calc_two_point_resistance_sparse(C, nodes, 0, 2) == pytest.approx(3.0 / SI, rel=1e-12)

    parallel = nx.MultiGraph()
    parallel.add_edge(0, 1, conductance=SI)
    parallel.add_edge(0, 1, conductance=SI)
    C, nodes = build_conductance_matrix_from_graph(parallel, solver="sparse")
    assert calc_two_point_resistance_sparse(C, nodes, 0, 1) == pytest.approx(0.5 / SI, rel=1e-12)


def test_the_sparse_two_point_resistance_is_the_eigendecompositions():
    G, inlets, outlets = _capillary_bed()
    dense, nodes = build_conductance_matrix_from_graph(G)
    sparse, _ = build_conductance_matrix_from_graph(G, solver="sparse")
    laplacian = calc_laplacian_from_conductance_matrix(dense)
    for a, b in [(inlets[0], outlets[0]), (inlets[1], outlets[2]), (inlets[0], inlets[2])]:
        eigen = calc_two_point_from_laplacian_matrix_nodeID(laplacian, G, a, b)
        assert calc_two_point_resistance_sparse(sparse, nodes, a, b) == pytest.approx(eigen, rel=1e-9)
        # A dense matrix is accepted too.
        assert calc_two_point_resistance_sparse(dense, nodes, a, b) == pytest.approx(eigen, rel=1e-9)


def test_two_unconnected_nodes_have_infinite_two_point_resistance(caplog):
    """No path, no current: the sparse solve says infinite where the
    eigendecomposition gives a finite number."""
    G, _boundaries = _disconnected()
    C, nodes = build_conductance_matrix_from_graph(G, solver="sparse")
    with caplog.at_level(logging.WARNING):
        assert calc_two_point_resistance_sparse(C, nodes, 0, 3) == math.inf
    assert "separate conductive parts" in caplog.text

    dense, _ = build_conductance_matrix_from_graph(G)
    eigen = calc_two_point_from_laplacian_matrix_nodeID(
        calc_laplacian_from_conductance_matrix(dense), G, 0, 3
    )
    assert math.isfinite(eigen)


def test_a_node_has_no_resistance_to_itself_and_an_unknown_one_is_refused():
    C, nodes = build_conductance_matrix_from_graph(_chain(SI, SI), solver="sparse")
    assert calc_two_point_resistance_sparse(C, nodes, 1, 1) == 0.0
    with pytest.raises(ValueError, match="not found"):
        calc_two_point_resistance_sparse(C, nodes, 0, 99)


# --- the network resistance -----------------------------------------------------


def _two_inlets_one_outlet() -> nx.MultiGraph:
    """Inlet 0 --R1-- outlet 2 --R2-- inlet 1."""
    G = nx.MultiGraph()
    G.add_edge(0, 2, conductance=1.0 * SI)  # R1 = 1/SI
    G.add_edge(1, 2, conductance=0.25 * SI)  # R2 = 4/SI
    return G


@pytest.mark.parametrize("solver", FLOW_SOLVERS)
def test_the_network_resistance_counts_every_inlet_and_the_two_point_one_does_not(solver):
    """Two inlets into one outlet: the network's resistance is the two
    vessels in parallel, R1 R2 / (R1 + R2); the two-point resistance between
    the first inlet and the outlet, the second inlet left floating, is R1."""
    G = _two_inlets_one_outlet()
    conductance, nodes, flow = _solve(G, [0, 1], [2], solver, inlet_p_bc=1000.0, outlet_p_bc=0.0)

    network = network_resistance(
        conductance, nodes, flow["pressure"], inlet_nodes=[0, 1],
        inlet_p_bc=1000.0, outlet_p_bc=0.0,
    )
    assert network == pytest.approx((1.0 * 4.0) / (1.0 + 4.0) / SI, rel=1e-12)
    assert calc_two_point_resistance_sparse(conductance, nodes, 0, 2) == pytest.approx(1.0 / SI)


def test_the_network_resistance_needs_a_pressure_drop_and_a_flow():
    G = _two_inlets_one_outlet()
    conductance, nodes = build_conductance_matrix_from_graph(G)
    pressure = np.zeros(3)
    assert network_resistance(
        conductance, nodes, pressure, inlet_nodes=[0], inlet_p_bc=5.0, outlet_p_bc=5.0
    ) is None
    assert network_resistance(
        conductance, nodes, pressure, inlet_nodes=[0], inlet_p_bc=10.0, outlet_p_bc=0.0
    ) == math.inf


# --- the settings and the panel ---------------------------------------------------


def test_both_dropdowns_default_to_what_the_pipeline_always_did():
    flow = SCHEMA["haemodynamics_solver"]
    equivalent = SCHEMA["equivalent_resistance_solver"]
    assert (flow.kind, flow.default, tuple(flow.choices)) == ("choice", "dense", FLOW_SOLVERS)
    assert (equivalent.kind, equivalent.default, tuple(equivalent.choices)) == (
        "choice",
        "eigendecomposition",
        EQUIVALENT_RESISTANCE_SOLVERS,
    )
    assert flow.advanced and equivalent.advanced


def test_the_dropdowns_read_as_the_panel_names_them():
    from haemolynx.gui.form import CHOICE_VALUE_LABELS, label_for

    assert label_for("haemodynamics_solver") == "Haemodynamics solver"
    assert label_for("equivalent_resistance_solver") == "Equivalent resistance solver"
    assert CHOICE_VALUE_LABELS["haemodynamics_solver"] == {
        "dense": "Dense solver",
        "sparse": "Sparse solver",
    }
    assert CHOICE_VALUE_LABELS["equivalent_resistance_solver"] == {
        "eigendecomposition": "Eigendecomposition solver",
        "sparse": "Sparse solver",
    }


def test_both_dropdowns_are_behind_the_haemodynamics_tabs_advanced_button():
    from haemolynx.gui.layout import layout_for
    from haemolynx.gui.tabs import tabs_for

    tab = next(t for t in tabs_for(SCHEMA) if t.stage.title == "6. Haemodynamics")
    layout = layout_for(tab.stage.call, [field.name for field in tab.fields], SCHEMA)
    flow = layout.disclosure_of("haemodynamics_solver")
    equivalent = layout.disclosure_of("equivalent_resistance_solver")
    assert flow is not None and flow.anchor == "run_haemodynamics"
    # Beside the toggle it depends on, which is itself advanced.
    assert equivalent is flow
    assert "do_equiv_resistance_calculation" in flow.names
    assert "haemodynamics_solver" not in layout.box("haemodynamics").rows


# --- the solve stage --------------------------------------------------------------


def _bifurcation() -> nx.MultiGraph:
    """0 (inlet) -> 1 -> {2, 3} (outlets), real diameters, lengths in um."""
    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (0, 0, 50), 2: (0, 40, 90), 3: (0, -40, 90)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v, order, length, diameter in [
        (0, 1, "B01", 50.0, 15.0), (1, 2, "B02", 56.6, 20.0), (1, 3, "B03", 56.6, 8.0),
    ]:
        G.add_edge(u, v, key=0, length=length, branch_order=order, diameter_um=diameter,
                   voxels=[list(G.nodes[u]["pos"]), list(G.nodes[v]["pos"])])
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    return G


def _solve_stage(**overrides):
    from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, resolve_settings
    from haemolynx.pipeline.stages import build_haemodynamic_model, solve

    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "run_haemodynamics": True,
            "viscosity_law": "pries",
            "haematocrit": 0.45,
            "all_diams_const": False,
            "inlet_p_bc": 1000.0,
            "outlet_p_bc": 0.0,
            "inlet_nodes": [0],
            "outlet_nodes": [2, 3],
            "do_equiv_resistance_calculation": True,
            "diameter_by_branch_order": {"B01": 15.0, "B02": 20.0, "B03": 8.0},
            "haematocrit_distribution_max_iterations": 30,
            "haematocrit_distribution_tolerance": 1e-4,
        }
    )
    values.update(overrides)
    settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    boundaries = BoundaryNodes(inlet_nodes=[0], outlet_nodes=[2, 3], resistance_node_pair=(0, 2))
    model = build_haemodynamic_model(settings, HaemodynamicModel(graph=_bifurcation()), SCHEMA)
    return solve(settings, model, boundaries, SCHEMA)


@pytest.mark.parametrize("haematocrit_model", ["fixed", "distributed_iterative"])
def test_the_solve_stage_gives_the_same_answer_with_either_solver(haematocrit_model):
    default = _solve_stage(haematocrit_model=haematocrit_model)
    sparse = _solve_stage(
        haematocrit_model=haematocrit_model,
        haemodynamics_solver="sparse",
        equivalent_resistance_solver="sparse",
    )
    assert sparse.node_list == default.node_list
    np.testing.assert_allclose(sparse.pressure, default.pressure, rtol=1e-10)
    assert sparse.equivalent_resistance == pytest.approx(default.equivalent_resistance, rel=1e-9)
    assert sparse.network_resistance == pytest.approx(default.network_resistance, rel=1e-10)
    for u, v, key, data in default.graph.edges(keys=True, data=True):
        assert sparse.graph.edges[u, v, key]["flow_signed"] == pytest.approx(
            data["flow_signed"], rel=1e-9
        )


def test_the_solve_stage_reports_both_resistances():
    solution = _solve_stage()
    G = solution.graph
    inflow = G.edges[0, 1, 0]["flow_abs"]  # the one inlet's one vessel
    assert solution.network_resistance == pytest.approx(1000.0 / inflow, rel=1e-9)
    assert solution.equivalent_resistance_nodes == (0, 2)
    # Outlet 3 floats in the two-point one, so it is larger: no route via 3.
    assert solution.equivalent_resistance > solution.network_resistance


def test_both_resistances_go_into_the_statistics_csv(tmp_path):
    from haemolynx.pipeline.stages import Solution, resistance_statistics
    from haemolynx.statistics import export_statistics_to_csv

    solution = Solution(
        equivalent_resistance=2.5e16, equivalent_resistance_nodes=(0, 2), network_resistance=1.5e16
    )
    stats = resistance_statistics(solution)
    path = export_statistics_to_csv(stats, tmp_path / "stats.csv")
    text = path.read_text(encoding="utf-8")
    assert "Two-Point Equivalent Resistance" in text and "Pa.s/m^3" in text
    assert "Network Resistance, Pressure Drop / Total Inflow" in text
    assert "0 -> 2" in text
    assert resistance_statistics(Solution()) == {}
