"""Haematocrit under-relaxation and the non-convergence warning in the coupled solver."""
import logging

import networkx as nx
import pytest

from ImageLynx.haemodynamics.rheology import solve_coupled_flow_and_hematocrit


def _y_junction():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, length=10.0, fwhm_diameter_um=15.0)
    G.add_edge(1, 2, key=0, length=10.0, fwhm_diameter_um=10.0)
    G.add_edge(1, 3, key=0, length=10.0, fwhm_diameter_um=5.0)
    return G


def _solve(G, **kwargs):
    return solve_coupled_flow_and_hematocrit(
        G, starting_nodes=[0], output_nodes=[2, 3], input_p_bc=100.0, output_p_bc=10.0,
        systemic_hematocrit=0.45, **kwargs,
    )


@pytest.mark.parametrize("relaxation", [0.0, -0.1, 1.5])
def test_relaxation_outside_the_unit_interval_is_rejected(relaxation):
    with pytest.raises(ValueError, match="relaxation"):
        _solve(_y_junction(), relaxation=relaxation)


def test_one_pass_steps_part_of_the_way_to_the_skimmed_haematocrit():
    """After one pass, H = 0.45 + relaxation * (H_skim - 0.45), with H_skim the undamped value."""
    full, _ = _solve(_y_junction(), max_iterations=1, relaxation=1.0)
    for relaxation in (0.25, 0.5):
        damped, _ = _solve(_y_junction(), max_iterations=1, relaxation=relaxation)
        for u, v in ((1, 2), (1, 3)):
            h_skim = full[u][v][0]["hematocrit"]
            assert h_skim != pytest.approx(0.45)
            assert damped[u][v][0]["hematocrit"] == pytest.approx(
                0.45 + relaxation * (h_skim - 0.45), rel=1e-12)


def test_running_out_of_iterations_is_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="ImageLynx.haemodynamics.rheology"):
        _solve(_y_junction(), max_iterations=2, tolerance=1e-12)
    assert any("did not converge" in r.getMessage() for r in caplog.records)


def test_convergence_is_not_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="ImageLynx.haemodynamics.rheology"):
        _solve(_y_junction(), max_iterations=200, tolerance=1e-6)
    assert not any("did not converge" in r.getMessage() for r in caplog.records)


def test_damped_in_vivo_y_junction_records_convergence():
    """Undamped, this case flips between two states and stops on max_iterations."""
    import ImageLynx.haemodynamics.rheology as rh

    original = rh.calculate_pries_secomb_viscosity
    try:
        rh.calculate_pries_secomb_viscosity = (
            lambda d, h, mu_p=1.2: original(d, h, mu_p, law="in_vivo"))
        undamped, _ = _solve(_y_junction(), max_iterations=50, tolerance=1e-3, relaxation=1.0)
        damped, _ = _solve(_y_junction(), max_iterations=50, tolerance=1e-3)
    finally:
        rh.calculate_pries_secomb_viscosity = original
    assert undamped.graph["rheology_stop_reason"] == "max_iterations"
    assert damped.graph["rheology_stop_reason"] == "converged"
