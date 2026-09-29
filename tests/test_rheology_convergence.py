"""The rheology loop converges, and says so (open item 37).

Until item 37 the loop stopped on its 15-pass cap in every CB specimen, still wandering, and
every flow and haematocrit downstream came from whichever pass it stopped on. Three changes
fixed it, and each is tested here:

1. Stagnant edges (rounding-level flow) are left out of the haematocrit transport, so their
   noise direction no longer decides whether a junction skims.
2. Junctions with three or more outflows skim one-vs-rest instead of mixing proportionally,
   so a small daughter reversing no longer switches the junction's rule.
3. The stop is scale-free, so the pipeline (mPa) and the H2 drivers (mmHg) apply the same test.
"""
import itertools

import networkx as nx
import pytest

from ImageLynx import cb_settings
from ImageLynx.haemodynamics.resistance import STAGNANT_FLOW_FRACTION
from ImageLynx.haemodynamics.rheology import (
    RHEOLOGY_STATUS_KEYS,
    calculate_multiway_phase_separation_hematocrit,
    calculate_phase_separation_hematocrit,
    report_unconverged,
    rheology_status,
    solve_coupled_flow_and_hematocrit,
)

#: 1 mmHg in mPa, the pipeline's pressure unit.
_MPA_PER_MMHG = 133.322387415e3


def _y(d_parent=12.0, d1=8.0, d2=4.0, l2=60.0):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=40.0, fwhm_diameter_um=d_parent)
    G.add_edge(1, 2, key=0, length=25.0, fwhm_diameter_um=d1)
    G.add_edge(1, 3, key=0, length=l2, fwhm_diameter_um=d2)
    return G


def _solve(G, scale=1.0, outlets=(2, 3), **overrides):
    kwargs = {**cb_settings.rheology_solver_kwargs(), **overrides}
    return solve_coupled_flow_and_hematocrit(
        G, [0], list(outlets), 60.0 * scale, 20.0 * scale, systemic_hematocrit=0.45, **kwargs)


@pytest.mark.parametrize("diameters", [(12.0, 8.0, 4.0), (15.0, 10.0, 5.0)])
def test_asymmetric_y_converges_at_the_settings(diameters):
    """Both Y-junctions the reference used to call non-convergent now converge."""
    G, _ = _solve(_y(*diameters))
    status = rheology_status(G)
    assert status["rheology_stop_reason"] == "converged"
    assert status["rheology_relative_flow_change"] <= cb_settings.RHEOLOGY_FLOW_RTOL
    assert status["rheology_hematocrit_residual"] <= cb_settings.RHEOLOGY_HEMATOCRIT_ATOL


def test_the_stop_is_independent_of_the_pressure_unit():
    """mmHg and mPa give the same haematocrit and pass count, and flows scaled by the unit."""
    mmhg, _ = _solve(_y())
    mpa, _ = _solve(_y(), scale=_MPA_PER_MMHG)
    assert mmhg.graph["rheology_iterations"] == mpa.graph["rheology_iterations"]
    for u, v, k in mmhg.edges(keys=True):
        a, b = mmhg[u][v][k], mpa[u][v][k]
        assert a["hematocrit"] == pytest.approx(b["hematocrit"], abs=1e-12)
        assert b["flow_abs"] == pytest.approx(a["flow_abs"] * _MPA_PER_MMHG, rel=1e-9)


def _y_with_dead_end():
    """The Y plus a two-edge dead-end spur off the junction: no flow, only rounding."""
    G = _y()
    G.add_edge(1, 4, key=0, length=30.0, fwhm_diameter_um=6.0)
    G.add_edge(4, 5, key=0, length=30.0, fwhm_diameter_um=6.0)
    return G


def test_a_dead_end_spur_is_stagnant_and_does_not_change_the_split():
    """The spur carries rounding only. It used to make the junction a trifurcation whenever
    its noise pointed outward, which switched skimming off there."""
    plain, _ = _solve(_y())
    spurred, _ = _solve(_y_with_dead_end())
    assert spurred.graph["rheology_stop_reason"] == "converged"
    assert spurred.graph["rheology_stagnant_edges"] == 2
    for u, v in ((1, 2), (1, 3)):
        assert spurred[u][v][0]["hematocrit"] == pytest.approx(
            plain[u][v][0]["hematocrit"], abs=1e-9)
    # Not traversed, so it keeps its starting haematocrit.
    for u, v in ((1, 4), (4, 5)):
        assert spurred[u][v][0]["hematocrit"] == 0.45
        assert spurred[u][v][0]["flow_abs"] <= STAGNANT_FLOW_FRACTION * max(
            d["flow_abs"] for *_, d in spurred.edges(data=True))


def test_a_trifurcation_converges_and_conserves_red_cells():
    """One-vs-rest skimming at a three-way split: flux in = flux out at the junction."""
    G = _y()
    G.add_edge(1, 4, key=0, length=40.0, fwhm_diameter_um=6.0)
    G, _ = _solve(G, outlets=(2, 3, 4))
    assert G.graph["rheology_stop_reason"] == "converged"
    flux_in = G[0][1][0]["hematocrit"] * G[0][1][0]["flow_abs"]
    flux_out = sum(G[1][n][0]["hematocrit"] * G[1][n][0]["flow_abs"] for n in (2, 3, 4))
    # Stored haematocrit is within the 1e-4 residual of the exact split, not on it.
    assert flux_out == pytest.approx(flux_in, rel=1e-3)
    # Skimming is on: the daughters no longer all carry the feed haematocrit.
    hs = [G[1][n][0]["hematocrit"] for n in (2, 3, 4)]
    assert max(hs) - min(hs) > 0.01


def test_multiway_with_two_daughters_is_the_pries_law():
    for q1, q2 in ((0.7, 0.3), (0.5, 0.5), (0.2, 0.8)):
        expected = calculate_phase_separation_hematocrit(q1 + q2, 0.45, q1, 8.0, q2, 5.0, 10.0)
        got = calculate_multiway_phase_separation_hematocrit(0.45, [q1, q2], [8.0, 5.0], 10.0)
        assert got == pytest.approx(list(expected), rel=1e-12)


def test_multiway_is_continuous_as_a_daughter_vanishes():
    """A third daughter whose flow falls to zero leaves the two-way result behind it."""
    two = calculate_multiway_phase_separation_hematocrit(0.45, [0.6, 0.4], [8.0, 5.0], 10.0)
    three = calculate_multiway_phase_separation_hematocrit(
        0.45, [0.6, 0.4, 1e-9], [8.0, 5.0, 6.0], 10.0)
    assert three[:2] == pytest.approx(two, rel=1e-6)
    assert three[2] == 0.0


def test_multiway_conserves_flux_and_ignores_order():
    flows, diams = [0.5, 0.3, 0.2], [9.0, 6.0, 4.0]
    base = calculate_multiway_phase_separation_hematocrit(0.45, flows, diams, 10.0)
    assert sum(h * q for h, q in zip(base, flows)) == pytest.approx(0.45 * sum(flows), rel=1e-12)
    for perm in itertools.permutations(range(3)):
        got = calculate_multiway_phase_separation_hematocrit(
            0.45, [flows[i] for i in perm], [diams[i] for i in perm], 10.0)
        assert got == pytest.approx([base[i] for i in perm], rel=1e-12)


def test_status_records_every_key_and_the_settings_used():
    G, _ = _solve(_y())
    status = rheology_status(G)
    assert tuple(status) == RHEOLOGY_STATUS_KEYS
    assert status["rheology_settings"]["relaxation"] == cb_settings.RHEOLOGY_RELAXATION
    assert status["rheology_settings"]["stagnant_flow_fraction"] == STAGNANT_FLOW_FRACTION
    assert status["rheology_murray_parent_bifurcations"] == 0


def test_status_of_an_unsolved_graph_raises():
    with pytest.raises(ValueError, match="rheology status"):
        rheology_status(_y())


def test_an_unconverged_solve_is_reported(capsys):
    G, _ = _solve(_y(), max_iterations=3, relaxation=1.0)
    assert report_unconverged(rheology_status(G), "WKY-Z") is True
    assert "WARNING: WKY-Z" in capsys.readouterr().out
    G, _ = _solve(_y())
    assert report_unconverged(rheology_status(G), "WKY-Z") is False
