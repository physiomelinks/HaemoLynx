"""The phase separation law against the Pries form, and how the solver picks D_F.

Reference form as printed in Rasmussen, Secomb & Pries (2018), with A, B and X0 all scaled by
the feeding vessel diameter D_F.
"""
import logging

import networkx as nx
import numpy as np
import pytest

from ImageLynx.haemodynamics.rheology import (
    calculate_phase_separation_hematocrit,
    feeding_vessel_diameter,
    solve_coupled_flow_and_hematocrit,
)


def _pries(fq1, h, d1, d2, d_f):
    """Independent transcription of the published law, returning (H1, H2)."""
    x0 = 0.964 * (1 - h) / d_f
    if fq1 <= x0:
        fe1 = 0.0
    elif fq1 >= 1 - x0:
        fe1 = 1.0
    else:
        r = (d1 / d2) ** 2
        a = -13.29 * (r - 1) / (r + 1) * (1 - h) / d_f
        b = 1 + 6.98 * (1 - h) / d_f
        y = (fq1 - x0) / (1 - 2 * x0)
        fe1 = 1 / (1 + np.exp(-(a + b * np.log(y / (1 - y)))))
    return h * fe1 / fq1, h * (1 - fe1) / (1 - fq1)


def _split(fq1, h, d1, d2, d_f, q_in=1.0):
    return calculate_phase_separation_hematocrit(
        q_in, h, fq1 * q_in, d1, (1 - fq1) * q_in, d2, d_f
    )


@pytest.mark.parametrize(
    "d1, d2, fq1, expected",
    [
        # The worked cases from the audit, D_F = 10 um, H = 0.45
        (6.0, 10.0, 0.4, (0.483, 0.428)),
        (4.0, 8.0, 0.3, (0.435, 0.456)),
    ],
)
def test_matches_the_pries_form_on_the_worked_cases(d1, d2, fq1, expected):
    h1, h2 = _split(fq1, 0.45, d1, d2, 10.0)
    assert h1 == pytest.approx(expected[0], abs=1e-3)
    assert h2 == pytest.approx(expected[1], abs=1e-3)
    assert (h1, h2) == pytest.approx(_pries(fq1, 0.45, d1, d2, 10.0), abs=1e-12)


def test_matches_the_pries_form_across_a_grid():
    for h in (0.2, 0.45, 0.6):
        for d_f in (4.0, 8.0, 15.0, 40.0):
            for d1, d2 in ((3.0, 12.0), (7.0, 7.0), (20.0, 5.0)):
                for fq1 in np.linspace(0.02, 0.98, 13):
                    got = _split(fq1, h, d1, d2, d_f)
                    assert got == pytest.approx(_pries(fq1, h, d1, d2, d_f), abs=1e-12)


@pytest.mark.parametrize("d1, d2, fq1", [(6.0, 10.0, 0.4), (4.0, 8.0, 0.3), (15.0, 3.0, 0.7)])
def test_swapping_branch_labels_gives_the_same_haematocrits(d1, d2, fq1):
    """With D_F in place of D_1 the law no longer depends on graph edge order."""
    h1, h2 = _split(fq1, 0.45, d1, d2, 10.0)
    h2_swapped, h1_swapped = _split(1 - fq1, 0.45, d2, d1, 10.0)
    assert h1 == pytest.approx(h1_swapped, abs=1e-12)
    assert h2 == pytest.approx(h2_swapped, abs=1e-12)


def test_x0_scales_with_the_feeding_diameter():
    """X0 = 0.964 (1 - H) / D_F: 0.106 at 5 um, 0.027 at 20 um, so FQ = 0.08 sits between."""
    h_narrow, _ = _split(0.08, 0.45, 8.0, 8.0, 5.0)
    h_wide, _ = _split(0.08, 0.45, 8.0, 8.0, 20.0)
    assert h_narrow == 0.0
    assert h_wide > 0.0


def test_branch_above_one_minus_x0_takes_every_red_cell():
    h1, h2 = _split(0.95, 0.45, 8.0, 8.0, 5.0)
    assert h2 == 0.0
    assert h1 == pytest.approx(0.45 / 0.95)


def test_red_cell_flux_is_conserved():
    q_in = 3.7
    for h in (0.3, 0.45):
        for d_f in (5.0, 10.0, 30.0):
            for d1, d2 in ((4.0, 9.0), (10.0, 10.0), (25.0, 6.0)):
                for fq1 in np.linspace(0.01, 0.99, 25):
                    h1, h2 = _split(fq1, h, d1, d2, d_f, q_in=q_in)
                    flux_out = fq1 * q_in * h1 + (1 - fq1) * q_in * h2
                    assert flux_out == pytest.approx(q_in * h, abs=1e-10)


def test_x0_of_one_half_or_more_falls_back_to_proportional_splitting():
    # X0 = 0.964 * 0.55 / 1.0 = 0.53
    assert _split(0.3, 0.45, 1.0, 2.0, 1.0) == (0.45, 0.45)


def test_d_parent_is_required():
    with pytest.raises(TypeError):
        calculate_phase_separation_hematocrit(1.0, 0.45, 0.4, 6.0, 0.6, 10.0)


# --- Choosing D_F in the solver ---


def _dag(in_edges, node=1):
    dag = nx.MultiDiGraph()
    for src, q, d in in_edges:
        dag.add_edge(src, node, flow_abs=q, assigned_diameter_um=d)
    dag.add_edge(node, 10, flow_abs=1.0, assigned_diameter_um=5.0)
    dag.add_edge(node, 11, flow_abs=1.0, assigned_diameter_um=7.0)
    return dag


def test_feeding_diameter_is_the_single_in_edge():
    assert feeding_vessel_diameter(_dag([(0, 2.0, 12.0)]), 1, 5.0, 7.0) == (12.0, False)


def test_feeding_diameter_at_a_merge_is_the_flow_weighted_mean():
    d_f, fell_back = feeding_vessel_diameter(_dag([(0, 3.0, 12.0), (2, 1.0, 4.0)]), 1, 5.0, 7.0)
    assert d_f == pytest.approx((3.0 * 12.0 + 1.0 * 4.0) / 4.0)
    assert not fell_back


def test_feeding_diameter_falls_back_to_the_larger_daughter():
    assert feeding_vessel_diameter(_dag([]), 1, 5.0, 7.0) == (7.0, True)
    # An in-edge with no flow says nothing about the feed either
    assert feeding_vessel_diameter(_dag([(0, 0.0, 12.0)]), 1, 5.0, 7.0) == (7.0, True)


def test_solver_uses_the_parent_diameter_at_a_bifurcation():
    """Y-split: 0 -> 1 (15 um), then 1 -> 2 (10 um) and 1 -> 3 (5 um)."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=15.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(1, 3, key=0, length=10.0, fwhm_diameter_um=5.0)

    G, _ = solve_coupled_flow_and_hematocrit(
        G, starting_nodes=[0], output_nodes=[2, 3], input_p_bc=100.0, output_p_bc=10.0,
        systemic_hematocrit=0.45, max_iterations=10, tolerance=1e-6,
    )

    q2, q3 = G[1][2][0]["flow_abs"], G[1][3][0]["flow_abs"]
    fq2 = q2 / (q2 + q3)
    expected = _pries(fq2, 0.45, 10.0, 5.0, 15.0)
    assert G[1][2][0]["hematocrit"] == pytest.approx(expected[0], rel=1e-9)
    assert G[1][3][0]["hematocrit"] == pytest.approx(expected[1], rel=1e-9)


def test_solver_logs_bifurcations_without_an_inflowing_parent(caplog):
    """A node whose two edges both drain away has no parent; the fallback is counted."""
    G = nx.MultiGraph()
    # 0 (inlet) -> 1 -> 2 (outlet); node 1 also hangs a stagnant fork 1 -> 4 <- 5 -> 6
    # that carries no flow, so node 5 has two out-edges and no inflowing parent.
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(5, 4, key=0, length=10.0, fwhm_diameter_um=6.0)
    G.add_edge(5, 6, key=0, length=10.0, fwhm_diameter_um=6.0)
    G.add_edge(4, 1, key=0, length=10.0, fwhm_diameter_um=6.0)

    with caplog.at_level(logging.INFO, logger="ImageLynx.haemodynamics.rheology"):
        solve_coupled_flow_and_hematocrit(
            G, starting_nodes=[0], output_nodes=[2], input_p_bc=100.0, output_p_bc=10.0,
            systemic_hematocrit=0.45, max_iterations=3, tolerance=1e-6,
        )
    assert any("no inflowing parent" in r.getMessage() for r in caplog.records)
