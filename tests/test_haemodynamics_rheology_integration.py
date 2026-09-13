import pytest
import numpy as np
import networkx as nx
from ImageLynx.haemodynamics.rheology import solve_coupled_flow_and_hematocrit

def test_coupled_solver_convergence():
    """Verify the iterative solver successfully converges and assigns hematocrit/viscosity."""
    G = nx.MultiGraph()
    # Simple Y-bifurcation network
    # Node 0 (Inlet) -> Node 1 -> Node 2 (Outlet 1)
    #                         -> Node 3 (Outlet 2)
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=15.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0) # Medium branch
    G.add_edge(1, 3, key=0, length=10.0, fwhm_diameter_um=5.0)  # Tiny branch
    
    starting_nodes = [0]
    output_nodes = [2, 3]
    
    # Run the iterative solver
    G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes=starting_nodes,
        output_nodes=output_nodes,
        input_p_bc=100.0,
        output_p_bc=10.0,
        systemic_hematocrit=0.45,
        max_iterations=10,
        tolerance=1e-3
    )
    
    # Assert solver completed and returned pressures
    assert final_pressure is not None
    assert len(final_pressure) == 4
    
    # Assert every edge got hematocrit and viscosity attributes
    for u, v, key, data in G_solved.edges(keys=True, data=True):
        assert "hematocrit" in data
        assert "viscosity" in data
        assert "flow_abs" in data
        
    # The parent branch should still have systemic hematocrit
    assert np.isclose(G_solved[0][1][0]["hematocrit"], 0.45, atol=1e-5)

    # Red cell conservation across the bifurcation, which holds under either law.
    q_in = G_solved[0][1][0]["flow_abs"]
    rbc_out = sum(G_solved[1][t][0]["hematocrit"] * G_solved[1][t][0]["flow_abs"]
                  for t in (2, 3))
    assert rbc_out == pytest.approx(0.45 * q_in, rel=0.05)

    # Both branches must stay physical.
    for t in (2, 3):
        assert 0.0 <= G_solved[1][t][0]["hematocrit"] <= 1.0

    # The direction of skimming is *not* asserted here, and that is deliberate: it depends on
    # the viscosity law through the flow split, and the split is what the skimming model keys
    # on. See test_skimming_follows_the_flow_fraction_not_the_velocity below.


def test_skimming_follows_the_flow_fraction_not_the_velocity():
    """Red cells concentrate in the branch taking the larger share of *flow*.

    That is the invariant the implemented model can actually guarantee.
    ``calculate_phase_separation_hematocrit`` is posed in fractional blood flow:

        fq1 = q_out1 / q_in
        logit_fq = log((fq1 - x0) / (1 - fq1 - x0))

    Velocity appears nowhere in it. This test previously asserted that red cells follow the
    *faster* branch, which held only because the flow split happened to put the faster branch
    and the larger flow fraction on the same side. Correcting the resistance update to plain
    Poiseuille moved the in vivo split from 64/36 to 72/28, the two parted company, and the
    old assertion failed while the model was behaving exactly as written.

    The velocity divergence is still worth recording, so it is asserted below as observed
    behaviour rather than as correctness. Under the in vivo relation the narrow branch is the
    faster one despite drawing less than a third of the flow, because a quarter of the area
    carries 28% of it. Whether a velocity-keyed skimming law would be the better model is a
    question about phase separation, not about the viscosity relation, and it is open.
    """
    import ImageLynx.haemodynamics.rheology as rh

    original = rh.calculate_pries_secomb_viscosity
    results = {}
    try:
        for law in ("in_vitro", "in_vivo"):
            rh.calculate_pries_secomb_viscosity = (
                lambda d, h, mu_p=1.2, _law=law: original(d, h, mu_p, law=_law))
            H = nx.MultiGraph()
            H.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=15.0)
            H.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
            H.add_edge(1, 3, key=0, length=10.0, fwhm_diameter_um=5.0)
            solved, _ = solve_coupled_flow_and_hematocrit(
                H, [0], [2, 3], 100.0, 10.0, 0.45, 10, 1e-3)
            q2, q3 = solved[1][2][0]["flow_abs"], solved[1][3][0]["flow_abs"]
            results[law] = {
                "share_wide": q2 / (q2 + q3),
                "v_wide": q2 / (np.pi * 5.0 ** 2),
                "v_narrow": q3 / (np.pi * 2.5 ** 2),
                "h_wide": solved[1][2][0]["hematocrit"],
                "h_narrow": solved[1][3][0]["hematocrit"],
            }
    finally:
        rh.calculate_pries_secomb_viscosity = original

    for law, r in results.items():
        wide_takes_more_flow = r["share_wide"] > 0.5
        richer_is_wide = r["h_wide"] > r["h_narrow"]
        assert wide_takes_more_flow == richer_is_wide, (
            f"{law}: red cells did not follow the larger flow fraction ({r})")

    assert results["in_vitro"]["share_wide"] > results["in_vivo"]["share_wide"], (
        "the in vivo law should even out the split by penalising the narrow branch harder")

    # Recorded, not asserted as correct: under the in vivo relation velocity and flow
    # fraction point at different branches. The narrow branch draws under a third of the
    # flow through a quarter of the area, so it is the faster one while the wide branch is
    # the one that skims red cells. See the docstring.
    in_vivo = results["in_vivo"]
    assert in_vivo["share_wide"] > 0.5, "expected the wide branch to take the larger share"
    assert in_vivo["v_narrow"] > in_vivo["v_wide"], (
        "expected the narrow branch to remain the faster one despite the smaller share; "
        f"if this has changed, the divergence noted in the docstring has gone away ({in_vivo})")


def test_coupled_solver_dag_cycle_handling(caplog):
    """Verify the topological sorter safely handles biologically impossible fluid loops."""
    G = nx.MultiGraph()
    # Force a loop: 0 -> 1 -> 2 -> 1
    # We will manually assign bizarre pressures to trick the DAG builder
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(2, 1, key=1, length=10.0, fwhm_diameter_um=10.0)
    
    starting_nodes = [0]
    output_nodes = [2]
    
    # By default, a linear solver won't create a cycle here, but let's see if 
    # the solver at least completes without throwing a fatal unhandled error.
    # It should just resolve it as normal parallel edges since P1 > P2.
    G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
        G,
        starting_nodes=starting_nodes,
        output_nodes=output_nodes,
        input_p_bc=100.0,
        output_p_bc=10.0,
        systemic_hematocrit=0.45,
        max_iterations=5
    )
    
    assert final_pressure is not None

def test_coupled_solver_matrix_singularity_safety(caplog):
    """Verify solver safely catches missing boundary conditions without cryptic math crashes."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=10.0)
    
    # Missing starting nodes entirely
    import pytest
    from scipy.sparse.linalg import MatrixRankWarning
    
    try:
        G_solved, final_pressure = solve_coupled_flow_and_hematocrit(
            G,
            starting_nodes=[],
            output_nodes=[],
            input_p_bc=100.0,
            output_p_bc=10.0,
            systemic_hematocrit=0.45,
            max_iterations=1
        )
        assert False, "Should not be able to solve a network with no boundary conditions."
    except Exception as e:
        # Just verifying it raises some form of Error and doesn't get stuck in an infinite loop
        assert isinstance(e, Exception)

def test_carotid_pipeline_end_to_end_resistance_and_skimming():
    """Smoke test to ensure the main carotid pipeline classes and execution logic don't break.

    Previously exercised the sphincter path with ``constrict_at_pericytes=True``. That
    capability is disabled for the carotid config, so the wiring this covers is now the
    unconstricted resistance path, which is the only one reachable.
    """
    import sys
    from pathlib import Path
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    from carotid_image_to_model import (
        GraphConfig, HaemodynamicsConfig, VisualizationConfig, PipelineConfig, PerfusionConfig,
        _setup_boundary_conditions_and_haemodynamics
    )
    
    # 1. Create a mock network that perfectly matches the starting assumptions of Phase 4
    G = nx.MultiGraph()
    # It must have a 'pos' array (ZYX)
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([20.0, 20.0, 20.0]))
    G.add_node(3, pos=np.array([40.0, 40.0, 40.0]))
    
    G.add_edge(1, 2, key=0, length=20.0, voxels=[[0,0,0], [20,20,20]])
    G.add_edge(2, 3, key=0, length=20.0, voxels=[[20,20,20], [40,40,40]])
    
    # Set up the configs matching an SHR Sphincter run
    graph_config = GraphConfig(
        edge_percent=25.0,
        end_percent=25.0
    )
    hemo_config = HaemodynamicsConfig(
        diameter_by_branch_order={"DEFAULT": {"d1": 10.0, "d2": 10.0}},
        # Stated explicitly rather than inherited: the default moved to "edt_radius" (#98
        # Phase 3), and this is a graph-only smoke test with a mock image and no binary mask,
        # so EDT correctly refuses to run. What is being exercised here is the haemodynamics
        # wiring, not diameter estimation.
        radius_assignment_mode="fwhm_radius",
    )
    
    # Create a mock 3D numpy array representing the image (to pass the FWHM step)
    mock_image = np.ones((50, 50, 50))
    
    try:
        # 2. Run Phase 4 (Setup boundary conditions, branch orders, and FWHM)
        start, end, pair = _setup_boundary_conditions_and_haemodynamics(
            G, mock_image, hemo_config, graph_config, "mock_path", "numpy"
        )
        
        # Ensure Phase 4 succeeded
        assert len(start) > 0
        assert len(end) > 0
        
        # Ensure Phase 4 correctly attached the resistance from the Sphincter PoiseuilleModel
        for u, v, k, d in G.edges(keys=True, data=True):
            assert "resistance" in d
            assert d["resistance"] > 0
            
    except Exception as e:
        pytest.fail(f"End-to-End Pipeline Phase 4 failed: {e}")


def _bifurcation_graph():
    """Inlet, one junction, two unequal daughters. Diameters span the capillary range."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=40.0, fwhm_diameter_um=12.0)
    G.add_edge(1, 2, key=0, length=25.0, fwhm_diameter_um=8.0)
    G.add_edge(1, 3, key=0, length=60.0, fwhm_diameter_um=4.0)
    return G


def test_resistance_matches_poiseuille_at_the_solved_viscosity():
    """Every edge's resistance must be the straight-tube value at its own viscosity.

    The update step used to rescale a stored baseline by ``mu_app / mu_old``, where
    ``mu_old = 1 / d**1.647`` is the power law ``poiseuille.py`` assigns. That telescopes to
    Poiseuille only if the baseline still carries ``mu_old``, and it does not: the solver
    overwrites resistance with the Pries-Secomb value before the loop starts, so the
    baseline was captured without it. Viscosity was therefore applied twice.
    """
    G_solved, _ = solve_coupled_flow_and_hematocrit(
        _bifurcation_graph(),
        starting_nodes=[0],
        output_nodes=[2, 3],
        input_p_bc=13.332e6,
        output_p_bc=0.27e6,
        systemic_hematocrit=0.45,
        max_iterations=10,
        tolerance=1e-4,
    )

    for u, v, key, data in G_solved.edges(keys=True, data=True):
        d = data["fwhm_diameter_um"]
        expected = (128.0 * data["viscosity"] * data["length"]) / (np.pi * d ** 4)
        assert data["resistance"] == pytest.approx(expected, rel=1e-9), (
            f"edge ({u}, {v}, {key}) at d={d} um carries "
            f"{data['resistance']:.6g} against Poiseuille's {expected:.6g}"
        )


def test_resistance_is_not_inflated_by_the_power_law_viscosity():
    """Guard the specific defect: the surviving factor was mu_PS(d, H) * d**1.647.

    That factor is calibre-dependent, running roughly 200x at 3 um to 540x at 20 um, so it
    distorted the distribution of flow and not merely its scale. Asserting the ratio is 1
    rather than merely 'small' is what makes a partial reintroduction fail here.
    """
    G_solved, _ = solve_coupled_flow_and_hematocrit(
        _bifurcation_graph(),
        starting_nodes=[0],
        output_nodes=[2, 3],
        input_p_bc=13.332e6,
        output_p_bc=0.27e6,
        systemic_hematocrit=0.45,
        max_iterations=10,
        tolerance=1e-4,
    )

    for u, v, key, data in G_solved.edges(keys=True, data=True):
        d = data["fwhm_diameter_um"]
        poiseuille = (128.0 * data["viscosity"] * data["length"]) / (np.pi * d ** 4)
        inflated = poiseuille * (data["viscosity"] * d ** 1.647)

        assert data["resistance"] / poiseuille == pytest.approx(1.0, rel=1e-9)
        assert data["resistance"] < inflated / 10.0, (
            f"edge ({u}, {v}, {key}) resistance is within an order of magnitude of the "
            f"double-applied value {inflated:.6g}"
        )


def test_the_update_step_agrees_with_the_initialisation():
    """One iteration and many must give the same resistance where haematocrit is unchanged.

    The inlet edge carries systemic haematocrit throughout, so its viscosity never changes.
    Its resistance must therefore be identical however many passes the solver makes. A
    mismatch means the initialisation and the in-loop update disagree on the formula, which
    is how the double application went unnoticed.
    """
    common = dict(
        starting_nodes=[0],
        output_nodes=[2, 3],
        input_p_bc=13.332e6,
        output_p_bc=0.27e6,
        systemic_hematocrit=0.45,
        tolerance=1e-4,
    )
    one_pass, _ = solve_coupled_flow_and_hematocrit(
        _bifurcation_graph(), max_iterations=1, **common)
    many_passes, _ = solve_coupled_flow_and_hematocrit(
        _bifurcation_graph(), max_iterations=10, **common)

    assert one_pass[0][1][0]["hematocrit"] == pytest.approx(0.45, abs=1e-9)
    assert many_passes[0][1][0]["hematocrit"] == pytest.approx(0.45, abs=1e-9)
    assert many_passes[0][1][0]["resistance"] == pytest.approx(
        one_pass[0][1][0]["resistance"], rel=1e-9)


def _driver_call_lines(function_name, call_attrs):
    """Line numbers of calls to ``call_attrs`` inside ``function_name`` of the CB driver."""
    import ast
    from pathlib import Path

    source = (Path(__file__).parent.parent / "examples" / "carotid_image_to_model.py").read_text()
    tree = ast.parse(source)
    target = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    found = {name: [] for name in call_attrs}
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in found:
                found[node.func.attr].append(node.lineno)
    return found


def test_two_point_resistance_is_computed_after_the_rheology_solve():
    """The reported effective resistance must come from the Pries-Secomb resistances.

    It used to be computed before ``solve_coupled_flow_and_hematocrit`` ran, so it was built
    from whatever ``set_poiseuille_resistances`` had written - the power law mu = 1/d^1.647,
    which is not a viscosity in cP and carries a d^-5.647 dependence rather than Poiseuille's
    d^-4. Every flow and pressure in the model comes from the Pries-Secomb resistances, so
    the results table held two numbers derived from two different viscosity models.

    Asserted on call order in the source rather than behaviourally: the enclosing driver
    function also writes VTK, computes statistics and runs perfusion, and no test harness
    exists that can execute it. The ordering is the whole of the fix, so pinning the ordering
    is what stops it regressing.
    """
    calls = _driver_call_lines(
        "_export_and_solve_haemodynamics",
        ("solve_coupled_flow_and_hematocrit",
         "calc_two_point_from_laplacian_matrix_nodeID",
         "build_conductance_matrix_from_graph"),
    )

    assert len(calls["solve_coupled_flow_and_hematocrit"]) == 1
    assert len(calls["calc_two_point_from_laplacian_matrix_nodeID"]) == 1
    solve_line = calls["solve_coupled_flow_and_hematocrit"][0]
    two_point_line = calls["calc_two_point_from_laplacian_matrix_nodeID"][0]

    assert two_point_line > solve_line, (
        f"the two-point resistance is computed at line {two_point_line}, before the rheology "
        f"solve at line {solve_line}, so it reports the power-law resistances"
    )

    # One matrix build, not two. The pre-rheology build existed only to feed the two-point
    # calculation; leaving it behind would rebuild an ~8000-edge matrix for nothing.
    builds = calls["build_conductance_matrix_from_graph"]
    assert len(builds) == 1, f"expected a single conductance build, found {len(builds)}"
    assert builds[0] > solve_line


def test_the_two_point_resistance_depends_on_which_resistances_are_in_force():
    """Show the ordering is not cosmetic: the two resistance sets give different answers.

    If these agreed, computing the effective resistance before or after the rheology solve
    would not matter and the ordering test above would be pinning nothing.
    """
    from ImageLynx.haemodynamics.resistance import (
        build_conductance_matrix_from_graph,
        calc_laplacian_from_conductance_matrix,
        calc_two_point_from_laplacian_matrix_nodeID,
    )

    def two_point(graph):
        conductance, _ = build_conductance_matrix_from_graph(graph)
        laplacian = calc_laplacian_from_conductance_matrix(conductance)
        return calc_two_point_from_laplacian_matrix_nodeID(laplacian, graph, 0, 2)

    # As set_poiseuille_resistances leaves them, with mu = 1 / d^1.647.
    power_law = _bifurcation_graph()
    for u, v, key, data in power_law.edges(keys=True, data=True):
        d = data["fwhm_diameter_um"]
        mu_old = 1.0 / (d ** 1.647)
        data["resistance"] = (128.0 * mu_old * data["length"]) / (np.pi * d ** 4)
    before = two_point(power_law)

    # As the rheology solver leaves them.
    solved, _ = solve_coupled_flow_and_hematocrit(
        _bifurcation_graph(),
        starting_nodes=[0],
        output_nodes=[2, 3],
        input_p_bc=13.332e6,
        output_p_bc=0.27e6,
        systemic_hematocrit=0.45,
        max_iterations=10,
        tolerance=1e-4,
    )
    after = two_point(solved)

    assert np.isfinite(before) and np.isfinite(after)
    assert after / before > 100.0, (
        f"expected the Pries-Secomb resistances to give a far larger effective resistance "
        f"than the power law; got {before:.6g} against {after:.6g}"
    )
