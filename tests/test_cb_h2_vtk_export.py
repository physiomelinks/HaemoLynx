"""The H2 ParaView exports, and the one thing that can go silently wrong in them.

Every array written here is correct in isolation and still useless if the three frames
disagree: the glomus mask at 1.866 um, the perfusion grid at 4 um, and the centrelines in
micrometres. A transposed volume overlays the vessels perfectly well and puts every capillary
in the wrong nest. So the frame is tested against a placed voxel rather than against the
argument in the module docstring, and ``verify`` is tested by giving it a frame that is wrong.
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

pv = pytest.importorskip("pyvista")

from cb_h2_vtk import (                                                   # noqa: E402
    PENETRATION, _image, verify, write_glomus_clusters, write_glomus_surface,
    write_glomus_volume, write_perfusion, write_vessels,
)
from ImageLynx.haemodynamics.perfusion import PerfusionGrid               # noqa: E402
from ImageLynx.specimens import PROCESSING_VOXEL_UM                       # noqa: E402


def _mask_state(mask):
    return {"mask": mask, "prob": mask.astype(np.float32)}


def _line_graph():
    """Two edges along the array's z axis, with explicit centreline voxels."""
    G = nx.MultiGraph()
    for name, pos in (("a", (0.0, 10.0, 10.0)), ("b", (40.0, 10.0, 10.0)),
                      ("c", (80.0, 10.0, 10.0))):
        G.add_node(name, pos=np.asarray(pos))
    for u, v in (("a", "b"), ("b", "c")):
        p, q = G.nodes[u]["pos"], G.nodes[v]["pos"]
        G.add_edge(u, v, key=0, voxels=[p, (p + q) / 2, q], length=40.0,
                   flow_abs=1e-6, hematocrit=0.42, viscosity=3.1,
                   assigned_diameter_um=6.0)
    return G


# --- the frame ------------------------------------------------------------------------------

def test_the_written_volume_puts_the_arrays_z_axis_on_vtks_x_axis():
    """The convention the whole overlay rests on, checked on a deliberately asymmetric shape.

    If the axes were written in the natural (x, y, z) reading of the array, a cube would hide
    it and only the spacing would betray it. 4x6x8 cannot hide it.
    """
    img = _image((4, 6, 8), PROCESSING_VOXEL_UM)
    sz, sy, sx = PROCESSING_VOXEL_UM
    assert img.bounds[1] == pytest.approx(4 * sz)     # VTK x spans the array's z
    assert img.bounds[3] == pytest.approx(6 * sy)
    assert img.bounds[5] == pytest.approx(8 * sx)


def test_a_marked_voxel_lands_at_the_physical_point_its_index_names(tmp_path):
    """The stronger form: not just the extent, but where one particular voxel ends up."""
    mask = np.zeros((4, 6, 8), dtype=bool)
    mask[3, 1, 5] = True                              # asymmetric in all three axes
    path = write_glomus_volume(_mask_state(mask), "T", tmp_path)

    saved = pv.read(path)
    centres = saved.cell_centers().points
    marked = centres[np.asarray(saved.cell_data["glomus"]) > 0]
    sz, sy, sx = PROCESSING_VOXEL_UM

    assert len(marked) == 1
    assert marked[0] == pytest.approx([3.5 * sz, 1.5 * sy, 5.5 * sx])


def test_the_probability_and_the_mask_both_travel_with_the_volume(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 2, 2] = True
    saved = pv.read(write_glomus_volume(_mask_state(mask), "T", tmp_path))

    assert {"th_probability", "glomus"} <= set(saved.cell_data.keys())
    assert saved.n_cells == 64


def test_the_perfusion_volume_sits_at_the_grids_own_origin(tmp_path):
    """The grid is padded half a cell below the lowest node, so its origin is negative.

    Dropping that origin would shift the whole oxygen field by 2 um against the glomus mask,
    which is a third of a cell and entirely invisible by eye.
    """
    grid = PerfusionGrid(_line_graph(), (4.0, 4.0, 4.0))
    state = {"grid": grid, "po2": np.zeros(grid.n_cells), "th_cell": np.zeros(grid.n_cells),
             "m_max": np.zeros(grid.n_cells), "q_total": np.zeros(grid.n_cells),
             "s_incoming": np.zeros(grid.n_cells)}
    saved = pv.read(write_perfusion(state, "T", tmp_path))

    assert saved.origin == pytest.approx(tuple(grid.min_xyz))
    assert saved.origin[0] < 0.0
    assert set(saved.cell_data.keys()) == {
        "PO2_mmHg", "th_fraction", "metabolic_rate", "q_total_um3_s", "s_incoming"}


# --- the clusters ---------------------------------------------------------------------------

def test_two_separated_nests_get_different_ids_and_their_own_volumes(tmp_path):
    mask = np.zeros((24, 16, 16), dtype=bool)
    mask[3:7, 5:11, 5:11] = True                      # 4x6x6 = 144 voxels
    mask[16:20, 5:9, 5:9] = True                      # 4x4x4 =  64 voxels
    voxel_volume = float(np.prod(PROCESSING_VOXEL_UM))

    path, n, volumes = write_glomus_clusters(_mask_state(mask), "T", 0.0, tmp_path)
    saved = pv.read(path)

    assert n == 2
    assert sorted(volumes[volumes > 0]) == pytest.approx(
        sorted([64 * voxel_volume, 144 * voxel_volume]))

    ids = np.asarray(saved.point_data["cluster_id"])
    lower = ids[saved.points[:, 0] < 12 * PROCESSING_VOXEL_UM[0]]
    upper = ids[saved.points[:, 0] > 12 * PROCESSING_VOXEL_UM[0]]
    assert len(set(lower[lower > 0])) == 1 and len(set(upper[upper > 0])) == 1
    assert set(lower[lower > 0]).isdisjoint(set(upper[upper > 0]))


def test_the_volume_a_surface_point_carries_is_its_own_nests_volume(tmp_path):
    """The join is the point of the export: pick a vertex, get that nest's size, not a mean."""
    mask = np.zeros((24, 16, 16), dtype=bool)
    mask[3:7, 5:11, 5:11] = True
    mask[16:20, 5:9, 5:9] = True
    voxel_volume = float(np.prod(PROCESSING_VOXEL_UM))

    path, _n, _volumes = write_glomus_clusters(_mask_state(mask), "T", 0.0, tmp_path)
    saved = pv.read(path)
    ids = np.asarray(saved.point_data["cluster_id"])
    vols = np.asarray(saved.point_data["cluster_volume_um3"])

    on_big = vols[saved.points[:, 0] < 12 * PROCESSING_VOXEL_UM[0]]
    on_small = vols[saved.points[:, 0] > 12 * PROCESSING_VOXEL_UM[0]]
    assert np.median(on_big[on_big > 0]) == pytest.approx(144 * voxel_volume)
    assert np.median(on_small[on_small > 0]) == pytest.approx(64 * voxel_volume)
    assert set(np.unique(ids[ids > 0])) == {1, 2}


def test_one_nest_is_not_split_by_the_surfacing(tmp_path):
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    _path, n, volumes = write_glomus_clusters(_mask_state(mask), "T", 0.0, tmp_path)

    assert n == 1
    assert float(volumes[1]) == pytest.approx(512 * float(np.prod(PROCESSING_VOXEL_UM)))


def test_an_empty_mask_writes_a_file_rather_than_raising(tmp_path):
    """A specimen with no glomus signal must not take the whole batch down."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    path, n, _volumes = write_glomus_clusters(_mask_state(mask), "T", 0.0, tmp_path)

    assert n == 0 and path.exists()
    assert pv.read(path).n_points == 0


def test_the_surface_is_written_in_the_same_frame_as_the_volume(tmp_path):
    mask = np.zeros((24, 12, 12), dtype=bool)
    mask[16:20, 4:8, 4:8] = True
    path, surface = write_glomus_surface(_mask_state(mask), "T", 0.0, tmp_path)

    assert surface.n_points > 0
    # The nest sits high on the array's z axis, so it must sit high on VTK's x axis.
    assert pv.read(path).bounds[0] > 14 * PROCESSING_VOXEL_UM[0]


# --- the vessels ----------------------------------------------------------------------------

def _vessel_state(fractions, arrival=None):
    G = _line_graph()
    return {"graph": G, "edge_fraction": fractions,
            "arrival": arrival if arrival is not None else {"a": 0.0, "b": 1.0, "c": 2.0}}


def test_the_centrelines_carry_every_quantity_h2_measures(tmp_path):
    state = _vessel_state({("a", "b", 0): 0.9, ("b", "c", 0): 0.0})
    path, mesh = write_vessels(state, "T", tmp_path)

    assert mesh.n_cells == 2
    assert set(pv.read(path).cell_data.keys()) == {
        "th_fraction", "penetrating", "flow_um3_s", "hematocrit", "viscosity_cP",
        "diameter_um", "length_um", "transit_time"}


def test_the_penetrating_flag_follows_the_tissue_fraction_cutoff(tmp_path):
    """The classification every H2 comparison is stratified on, at the boundary itself."""
    state = _vessel_state({("a", "b", 0): PENETRATION, ("b", "c", 0): PENETRATION - 1e-6})
    _path, mesh = write_vessels(state, "T", tmp_path)

    flags = dict(zip(np.asarray(mesh.cell_data["th_fraction"]),
                     np.asarray(mesh.cell_data["penetrating"])))
    assert flags[PENETRATION] == 1.0
    assert flags[PENETRATION - 1e-6] == 0.0


def test_an_unreachable_edge_records_nan_rather_than_infinity(tmp_path):
    """VTK cannot hold inf. Written as inf it would reload as a finite garbage transit time."""
    state = _vessel_state({("a", "b", 0): 0.9, ("b", "c", 0): 0.9},
                          arrival={"a": 0.0, "b": 1.0})     # c never reached
    _path, mesh = write_vessels(state, "T", tmp_path)
    transit = np.asarray(mesh.cell_data["transit_time"])

    assert np.isnan(transit).sum() == 1
    assert np.isfinite(transit).sum() == 1


def test_flow_is_written_in_physical_units_not_solver_units(tmp_path):
    """The correction that moved absolute perfusion by five orders. Unit-less here would pass
    every other assertion in this file and be wrong by 1.3e5."""
    state = _vessel_state({("a", "b", 0): 0.9, ("b", "c", 0): 0.9})
    _path, mesh = write_vessels(state, "T", tmp_path)

    from ImageLynx.haemodynamics.resistance import poiseuille_flow_to_um3_per_s
    assert float(np.asarray(mesh.cell_data["flow_um3_s"])[0]) == pytest.approx(
        poiseuille_flow_to_um3_per_s(1e-6))


# --- the check itself -----------------------------------------------------------------------

def _verifiable_state(mask):
    """A graph whose nodes span the mask, with one edge deliberately outside the tissue.

    Both are needed for the check to mean anything: without the span the grid cannot contain
    the mask, and without a non-penetrating edge the specificity term has nothing to divide.
    """
    G = _line_graph()
    for name, pos in (("d", (90.0, 10.0, 10.0)), ("e", (110.0, 10.0, 10.0)),
                      ("low", (0.0, 0.0, 0.0)), ("high", (110.0, 36.0, 36.0))):
        G.add_node(name, pos=np.asarray(pos))
    p, q = G.nodes["d"]["pos"], G.nodes["e"]["pos"]
    G.add_edge("d", "e", key=0, voxels=[p, (p + q) / 2, q], length=20.0,
               flow_abs=1e-6, hematocrit=0.42, viscosity=3.1, assigned_diameter_um=6.0)

    grid = PerfusionGrid(G, (4.0, 4.0, 4.0))
    return {"mask": mask, "graph": G, "grid": grid,
            "edge_fraction": {("a", "b", 0): 0.9, ("b", "c", 0): 0.9, ("d", "e", 0): 0.0},
            "q_total": np.ones(grid.n_cells)}


def _tube_mask():
    """Tissue along the vessel line for the first 80 um, and nowhere else.

    A mostly-true mask would pass the containment test and make the transposed control
    meaningless, because everything is inside everything.
    """
    mask = np.zeros((60, 20, 20), dtype=bool)
    mask[0:43, 4:8, 4:8] = True
    return mask


class _Spec:
    specimen_id = "TEST"


def test_verify_passes_when_the_penetrating_edges_really_sit_in_the_mask():
    result = verify(_Spec(), _verifiable_state(_tube_mask()))

    assert result["penetrating_midpoint_inside_pct"] == 100.0
    assert result["non_penetrating_midpoint_inside_pct"] == 0.0
    assert result["perfusion_grid_contains_glomus_volume"] is True
    assert result["ok"] is True


def test_verify_fails_when_the_mask_does_not_reach_the_edges_it_claims_to_contain():
    """The failure the flag exists for: edges scored as penetrating whose midpoints are not in
    the mask at all. The overlay would still render, wrongly."""
    mask = _tube_mask()
    mask[10:] = False                                 # nowhere near the edge midpoints
    result = verify(_Spec(), _verifiable_state(mask))

    assert result["penetrating_midpoint_inside_pct"] < 70.0
    assert result["ok"] is False


def test_glomus_outside_the_grid_is_measured_and_does_not_block_the_export():
    """Coverage and registration are different failures and must not share a verdict.

    A grid that does not span the mask gives an overlay that is right as far as it goes: the
    tissue in the gap simply has no oxygen field. Blocking on it refused two of the six real
    specimens whose frames were as well registered as any that passed. What it must not do is
    pass unmentioned, so the shortfall is measured.
    """
    tall = np.zeros((100, 20, 20), dtype=bool)
    tall[0:43, 4:8, 4:8] = True                      # inside the grid, which reaches ~112 um
    tall[80:100, 4:8, 4:8] = True                    # genuinely beyond it
    result = verify(_Spec(), _verifiable_state(tall))

    assert result["perfusion_grid_contains_glomus_volume"] is False
    assert result["glomus_outside_grid_pct"] > 0.0
    assert result["ok"] is True                      # registration is sound, so it is written


def test_a_grid_that_spans_the_mask_reports_nothing_outside_it():
    result = verify(_Spec(), _verifiable_state(_tube_mask()))

    assert result["perfusion_grid_contains_glomus_volume"] is True
    assert result["glomus_outside_grid_pct"] == 0.0


def test_the_shortfall_is_the_true_fraction_of_tissue_the_grid_misses():
    """A percentage that is merely non-zero would pass the test above while being wrong."""
    mask = np.zeros((100, 20, 20), dtype=bool)
    mask[0:43, 4:8, 4:8] = True                      # inside the grid, which reaches ~112 um
    mask[80:100, 4:8, 4:8] = True                    # beyond it: 20 of 63 slabs

    result = verify(_Spec(), _verifiable_state(mask))
    assert result["glomus_outside_grid_pct"] == pytest.approx(100.0 * 20 / 63, abs=0.5)
