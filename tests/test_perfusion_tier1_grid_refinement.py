"""Open item 30: Tier 1 does not converge under grid refinement, because each vessel is a line.

``map_vessels_to_grid`` deposits a vessel only in the cells its centreline crosses, whatever its
diameter. Refining the grid therefore shrinks the vessel with the cell: in the limit it is a line
source with a fixed exchange conductance per unit length, and around a line the tissue field goes
as ln r. The cell holding the line sits ever closer to the blood, delivers less, and the tissue
drifts down by about the same amount at every refinement. It never settles.

The same vessel spread over its real cross-section (the flow shared by parallel centrelines on a
lattice finer than every grid here) converges. So the drift is the mapping, not the solver.
``vessel_mapping="cross_section"`` does that spreading inside ``map_vessels_to_grid``, from one
centreline and the vessel's diameter, and converges the same way.

Both cases are a thin slab along the vessel. Tier 1 feeds every vessel cell arterial blood, so
the field is uniform along it and the cross-section is the whole problem.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.cb_settings import BASE_M_MAX, PerfusionSettings
from ImageLynx.haemodynamics.perfusion import (
    PerfusionGrid,
    build_adr_matrix,
    calculate_blood_oxygen_content,
    cell_discharge_hematocrit,
    map_vessels_to_grid,
    solve_perfusion_steady_state,
)
from ImageLynx.haemodynamics.resistance import POISEUILLE_FLOW_TO_UM3_PER_S

_WIDTH = 81.0            # um, square cross-section
_CENTRE = 40.5           # on a cell centre at every h below
_H = (9.0, 3.0, 1.0, 1.0 / 3.0)
_FLOW_PER_UM = 1111.0    # um^3/s per um of vessel: delivery close to the tissue's demand
_RADIUS = 4.5            # um
_LATTICE = 0.25          # um, finer than every h, so the spread vessel stays spread


def _offsets(spread):
    if not spread:
        return [(0.0, 0.0)]
    ticks = np.arange(-_RADIUS + _LATTICE / 2, _RADIUS, _LATTICE)
    return [(dy, dx) for dy in ticks for dx in ticks if dy * dy + dx * dx <= _RADIUS ** 2]


def _solve(h, spread, vessel_mapping="centreline"):
    """Mean tissue PO2 and net O2 delivered per um^3, on a slab a cell or two thick."""
    offsets = _offsets(spread)
    G = nx.MultiGraph()
    z = np.linspace(0.01 * h, 0.99 * h, 5)
    for i, (dy, dx) in enumerate(offsets):
        y, x = _CENTRE + dy, _CENTRE + dx
        G.add_node(2 * i, pos=np.array([z[0], y, x]))
        G.add_node(2 * i + 1, pos=np.array([z[-1], y, x]))
        G.add_edge(2 * i, 2 * i + 1,
                   voxels=np.stack([z, np.full_like(z, y), np.full_like(z, x)], axis=1),
                   length=z[-1] - z[0], assigned_diameter_um=2 * _RADIUS, hematocrit=0.45,
                   flow_abs=_FLOW_PER_UM * h / len(offsets) / POISEUILLE_FLOW_TO_UM3_PER_S)

    grid = PerfusionGrid(G, (h, h, h), bounds_zyx=((0.0, 0.0, 0.0), (h, _WIDTH, _WIDTH)))
    config = PerfusionSettings(BASE_M_MAX)
    cells = map_vessels_to_grid(G, grid, vessel_mapping=vessel_mapping)
    A, q, s = build_adr_matrix(grid, cells, config)
    h_cell = cell_discharge_hematocrit(cells, grid.n_cells)
    po2, info = solve_perfusion_steady_state(grid, A, q, s, config, cell_hematocrit=h_cell,
                                             return_info=True)
    assert info["converged"]
    perfused = np.flatnonzero(q > 0)
    washout = sum(q[i] * calculate_blood_oxygen_content(po2[i], h_cell[i]) for i in perfused)
    return float(po2.mean()), float(s.sum() - washout) / (h * _WIDTH ** 2)


@pytest.fixture(scope="module")
def line():
    return [_solve(h, spread=False) for h in _H]


@pytest.fixture(scope="module")
def spread():
    return [_solve(h, spread=True) for h in _H]


@pytest.fixture(scope="module")
def disc():
    return [_solve(h, spread=False, vessel_mapping="cross_section") for h in _H]


def test_a_centreline_vessel_keeps_drifting_down_with_every_refinement(line):
    """Mean PO2 17.0, 11.5, 8.6, 6.8 mmHg at h 9, 3, 1, 1/3 um: no sign of a limit."""
    means = np.array([m for m, _ in line])
    steps = np.diff(means)
    assert np.all(steps < -1.0), means
    # Delivery per unit volume falls too: the line gives up less O2 the thinner it is drawn.
    delivered = np.array([d for _, d in line])
    assert np.all(np.diff(delivered) < 0), delivered


def test_the_same_vessel_spread_over_its_cross_section_converges(spread):
    """Mean PO2 17.0, 23.4, 23.8, 23.4 mmHg: once the cells resolve the vessel, it settles."""
    means = np.array([m for m, _ in spread])
    assert np.all(np.abs(np.diff(means)[1:]) < 0.5), means


def test_the_two_agree_while_the_vessel_fits_in_one_cell(line, spread):
    """At h = 9 um the vessel (9 um across) is one cell either way, so the mapping is the same."""
    assert line[0][0] == pytest.approx(spread[0][0], rel=1e-9)


def test_cross_section_mapping_settles_where_the_hand_spread_vessel_does(disc, spread, line):
    """Mean PO2 16.99, 23.36, 23.79, 23.23 mmHg at h 9, 3, 1, 1/3 um: a +-0.3 wobble, no trend.

    The centreline mapping loses 4.7 mmHg over the same three refinements. The two spread
    versions agree to 0.04 mmHg while the hand-spread lattice (0.25 um) is finer than the cell;
    at h = 1/3 um it is coarser, and the disc sampled inside map_vessels_to_grid is the finer.
    """
    means = np.array([m for m, _ in disc])
    assert means[0] == pytest.approx(line[0][0], abs=1e-6)
    assert np.ptp(means[1:]) < 0.6, means
    np.testing.assert_allclose(means[:3], [m for m, _ in spread[:3]], atol=0.05)
