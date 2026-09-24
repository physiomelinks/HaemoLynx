"""The coupled solver keeps the resistance an edge arrives with and rescales it by viscosity.

It used to overwrite every resistance with 128 mu_app(D, 0.45) L / (pi D^4) on entry, then
multiply that by mu_app / D^-1.647 each pass: viscosity counted twice, a stray D^1.647, and
upstream sphincter or pericyte geometry thrown away.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.probability import (
    set_poiseuille_resistances_with_probabilistic_periodic_constrictions,
)
from ImageLynx.haemodynamics.rheology import (
    calculate_pries_secomb_viscosity,
    solve_coupled_flow_and_hematocrit,
)

P_IN, P_OUT = 100.0, 10.0


def _mu_upstream(d):
    return 1.0 / d ** 1.647


def _chain(**edge_attrs):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=20.0, assigned_diameter_um=8.0, **edge_attrs)
    return G


def _solve(G, **kwargs):
    return solve_coupled_flow_and_hematocrit(
        G, starting_nodes=[0], output_nodes=[1], input_p_bc=P_IN, output_p_bc=P_OUT,
        systemic_hematocrit=0.45, **kwargs,
    )[0]


def test_upstream_resistance_is_rescaled_not_overwritten():
    """R = R_upstream * mu_app / mu_upstream, and the flow follows from it."""
    r_up = 3.7  # any value: stands in for an integrated sphincter or pericyte resistance
    G = _solve(_chain(resistance=r_up), max_iterations=5)
    data = G[0][1][0]
    expected = r_up * calculate_pries_secomb_viscosity(8.0, 0.45) / _mu_upstream(8.0)
    assert data["original_resistance"] == r_up
    assert data["resistance"] == pytest.approx(expected, rel=1e-12)
    assert data["flow_abs"] == pytest.approx((P_IN - P_OUT) / expected, rel=1e-12)


def test_the_first_pass_already_uses_the_rescaled_resistance():
    G = _solve(_chain(resistance=3.7), max_iterations=1)
    expected = 3.7 * calculate_pries_secomb_viscosity(8.0, 0.45) / _mu_upstream(8.0)
    assert G[0][1][0]["flow_abs"] == pytest.approx((P_IN - P_OUT) / expected, rel=1e-12)


def test_an_edge_without_resistance_gets_plain_poiseuille_at_the_apparent_viscosity():
    """E21: 128 mu_app L / (pi D^4), with no leftover D^1.647 factor."""
    G = _solve(_chain(), max_iterations=5)
    expected = 128.0 * calculate_pries_secomb_viscosity(8.0, 0.45) * 20.0 / (np.pi * 8.0 ** 4)
    assert G[0][1][0]["resistance"] == pytest.approx(expected, rel=1e-12)


def test_a_narrower_upstream_resistance_carries_through_to_the_flow_split():
    """Two identical parallel branches, one with double the upstream resistance.

    The haematocrit each branch receives differs, so the split is not exactly 2:1, but the
    constricted branch must carry clearly less. Under the old overwrite both carried the same.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, assigned_diameter_um=12.0)
    for target, factor in ((2, 1.0), (3, 2.0)):
        r = factor * 128.0 * _mu_upstream(8.0) * 10.0 / (np.pi * 8.0 ** 4)
        G.add_edge(1, target, key=0, length=10.0, assigned_diameter_um=8.0, resistance=r)
    G, _ = solve_coupled_flow_and_hematocrit(
        G, starting_nodes=[0], output_nodes=[2, 3], input_p_bc=P_IN, output_p_bc=P_OUT,
        systemic_hematocrit=0.45, max_iterations=200, tolerance=1e-10,
    )
    q_open, q_constricted = G[1][2][0]["flow_abs"], G[1][3][0]["flow_abs"]
    assert q_open / q_constricted > 1.5


def test_solving_the_same_graph_twice_gives_the_same_answer():
    G = _chain(resistance=3.7)
    first = _solve(G, max_iterations=5)[0][1][0]["flow_abs"]
    second = _solve(G, max_iterations=5)[0][1][0]["flow_abs"]
    assert G[0][1][0]["original_resistance"] == 3.7
    assert second == pytest.approx(first, rel=1e-12)


def test_probabilistic_setter_records_the_unconstricted_diameter():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=200.0, branch_order="B01")
    G, _ = set_poiseuille_resistances_with_probabilistic_periodic_constrictions(
        G,
        diameter_by_branch_order={"B01": 6.0},
        constriction_factor_by_branch_order={"B01": 0.5},
        rng=np.random.default_rng(0),
    )
    assert G[0][1][0]["assigned_diameter_um"] == 6.0


def test_pericyte_mask_setter_records_the_unconstricted_diameter(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("skimage.measure")
    from ImageLynx.haemodynamics.pericyte_mask import (
        set_poiseuille_resistances_with_pericyte_mask,
    )

    G = nx.MultiGraph()
    G.add_node(0, pos=np.asarray([16.0, 16.0, 4.0]))
    G.add_node(1, pos=np.asarray([16.0, 16.0, 28.0]))
    G.add_edge(0, 1, key=0, length=24.0, branch_order="B01",
               voxels=[[16.0, 16.0, float(z)] for z in range(4, 29)])
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[15:18, 15:18, 15:18] = 1
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_path), mask)

    G, _ = set_poiseuille_resistances_with_pericyte_mask(
        G,
        diameter_by_branch_order={"B01": 6.0},
        constriction_factor_by_branch_order={"B01": 0.5},
        pericyte_mask_path=mask_path,
        constriction_length=8.0,
    )
    assert G[0][1][0]["assigned_diameter_um"] == 6.0
