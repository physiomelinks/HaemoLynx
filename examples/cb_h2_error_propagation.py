"""Measurements behind Part 1 of the H2 capability assessment.

Four questions, answered from the batch outputs (``examples/outputs/cb_h1_batch/``) rather than
from a fresh pipeline run: the cached network graph for topology and ``per_edge_morphometry.csv``
for calibre and length. Every graph is checked against the placed ROI before it is read.

**Pressure boundaries are the pipeline's own.** Inlets and outlets come from
``select_boundary_terminal_nodes_by_face`` on ``cb_settings.BOUNDARY_AXIS`` at
``cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS``, the call the pipeline and every other H2 driver
make. Until package C of the 2026-09-29 re-run notes this script placed pressure with its own 25%
band on another axis, so its floors were measured on different pressure nodes.

**The solve is plain Poiseuille, not the Pries flow.** Viscosity is taken as uniform and folds
out, since only relative changes are read. That isolates how calibre error propagates through the
network from every other term in the chain, but it is not the H2 flow field: Fåhræus–Lindqvist
viscosity also depends on calibre and would move these numbers somewhat.

**Two perturbation sizes, and why both are reported.** One voxel, 1.866 um, is the scale at which a
diameter difference stops being physically resolved: H1 section 8.2 disqualifies calibre as a
finding because the between-group gap sits at one twentieth of that step. It is the conservative
bound. The threshold-calibrated size is the empirical one: the median calibre shift over the clean
threshold interval below the frozen value, measured by ``cb_h2_threshold_calibre.py``. Resistance
goes as the inverse fourth power of diameter, so the two do not simply scale, and the smaller one
is measured rather than inferred from the larger.

**Why independent and correlated are both run.** Resistance goes as the inverse fourth power of
diameter, so the per-edge uncertainty is near 94% at the median. Whether that matters at the network
level depends entirely on whether the errors cancel. They do when independent and do not when
correlated, and this pipeline's errors are correlated because every edge in a specimen comes from
one mask at one threshold.

Edges are counted on the MultiGraph, so parallel edges between one node pair are separate
conductances rather than merged.

Run with::

    venv/bin/python examples/cb_h2_error_propagation.py
    venv/bin/python examples/cb_h2_error_propagation.py --perturbation-um 0.740
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ImageLynx import cb_settings                                     # noqa: E402
from ImageLynx.batch_outputs import open_batch_run                     # noqa: E402
from ImageLynx.graph.boundaries import (                               # noqa: E402
    select_boundary_terminal_nodes_by_face,
)
from ImageLynx.specimens import PROCESSING_VOXEL_UM, get_specimen      # noqa: E402

SPECIMENS = ("WKY-A", "WKY-B", "WKY-C", "SHR-A", "SHR-B", "SHR-C")

# Analysis settings come from ImageLynx.cb_settings, which is their single owner.
ROI = cb_settings.ROI_VOXELS
BOUNDARY_AXIS = cb_settings.BOUNDARY_AXIS
FACE_TOLERANCE = cb_settings.BOUNDARY_FACE_TOLERANCE_VOXELS
# The in-plane voxel edge; the perturbation is quoted in these units.
VOXEL_UM = PROCESSING_VOXEL_UM[1]
# Median calibre shift over the clean threshold interval below the frozen value, averaged over
# the six specimens. Measured by cb_h2_threshold_calibre.py, not assumed: 0.740 um over 0.93 to
# 0.95 on the placed ROI, with the sensitivity runs holding the frozen seed (open item 41). It was
# 0.690 while the 0.93 run seeded at 0.98, and 0.922 over 0.85 to 0.90 on centre crops.
THRESHOLD_SHIFT_UM = 0.740
DRAWS = 24
SEED = 20260815


def network_arrays(G, edge_table):
    """Per-edge arrays over the MultiGraph: calibre from the graph, length from the edge table.

    ``G`` and ``edge_table`` are what ``BatchRun.graph()`` and ``BatchRun.edge_table()`` return:
    the graph carries ``assigned_diameter_um`` on every edge and the table is keyed by the same
    forward integer ``(u, v, key)``. Returns ``(u, v, length, diameter, index)``: ``u`` and ``v``
    are 0-based node indices and ``index`` maps each graph node to its index. Parallel edges stay
    separate conductances. Raises if any edge has no table row, rather than solving on a partial
    network.
    """
    index = {node: i for i, node in enumerate(G.nodes())}
    u, v, length, diameter, missing = [], [], [], [], []
    for a, b, key, data in G.edges(keys=True, data=True):
        row = edge_table.get((a, b, key))
        if row is None:
            missing.append((a, b, key))
            continue
        u.append(index[a])
        v.append(index[b])
        length.append(float(row["length_um"] or "nan"))
        diameter.append(data["assigned_diameter_um"])
    if missing:
        raise ValueError(f"{len(missing)} graph edges have no row in the edge table, "
                         f"e.g. {missing[:3]}; the table and the cached graph are out of step.")

    u, v = np.array(u, int), np.array(v, int)
    length, diameter = np.array(length, float), np.array(diameter, float)
    # Self-loops carry no pressure drop and a non-positive length or diameter would make the
    # conductance singular. Both are dropped rather than clamped, so nothing silently contributes.
    keep = (u != v) & np.isfinite(length) & (length > 0) & np.isfinite(diameter) & (diameter > 0)
    u, v, length, diameter = u[keep], v[keep], length[keep], diameter[keep]
    # A node left with no edge would be a zero row in the Laplacian and make the solve singular.
    stranded = len(index) - len(np.union1d(u, v))
    if stranded:
        raise ValueError(f"{stranded} nodes have no usable edge after dropping self-loops and "
                         f"non-positive calibre or length; the solve would be singular.")
    return u, v, length, diameter, index


def load_network(specimen_id):
    """The batch run's MultiGraph and its per-edge arrays, for the placed ROI only."""
    # Opening the run refuses a graph cut anywhere but the placed ROI (item 27).
    run = open_batch_run(get_specimen(specimen_id))
    G = run.graph()
    return (G, *network_arrays(G, run.edge_table()))


def face_boundaries(G, index, axis=BOUNDARY_AXIS, tolerance=FACE_TOLERANCE):
    """Inlet and outlet indices from the pipeline's face rule. Raises on an empty face."""
    inlets, outlets = select_boundary_terminal_nodes_by_face(
        G, ROI, axis=axis, face_tolerance_voxels=tolerance, voxel_size=PROCESSING_VOXEL_UM)
    return (np.array([index[n] for n in inlets], int),
            np.array([index[n] for n in outlets], int))


def terminals_on_any_face(G):
    """Degree-1 nodes, and which of them lie within one voxel of any ROI face."""
    terminals = [n for n, d in G.degree() if d == 1]
    pos = np.array([G.nodes[n]["pos"] for n in terminals], float).reshape(-1, 3)
    spacing = np.asarray(PROCESSING_VOXEL_UM, float)
    extent = (np.asarray(ROI, float) - 1.0) * spacing
    on_face = ((pos <= spacing) | (pos >= extent - spacing)).any(axis=1)
    return len(terminals), int(on_face.sum())


def edge_count_between_boundaries(G, inlet_nodes, outlet_nodes):
    """(edges in components holding both an inlet and an outlet, all edges), on the MultiGraph."""
    solvable = sum(
        G.subgraph(c).number_of_edges()
        for c in nx.connected_components(G)
        if (c & inlet_nodes) and (c & outlet_nodes)
    )
    return solvable, G.number_of_edges()


def solve_edge_flows(u, v, length, diameter, inlets, outlets, n_nodes):
    """Poiseuille network solve. Viscosity folds out, since only relative changes are read."""
    conductance = (np.pi * diameter ** 4) / (128.0 * length)

    fixed = np.zeros(n_nodes, bool)
    pressure = np.zeros(n_nodes)
    fixed[inlets] = True
    pressure[inlets] = 1.0
    fixed[outlets] = True
    pressure[outlets] = 0.0

    free = ~fixed
    index = -np.ones(n_nodes, int)
    index[free] = np.arange(free.sum())

    rows, cols, vals = [], [], []
    rhs = np.zeros(free.sum())
    for a, b in ((u, v), (v, u)):
        for k in range(len(a)):
            i, j, w = a[k], b[k], conductance[k]
            if not free[i]:
                continue
            rows.append(index[i])
            cols.append(index[i])
            vals.append(w)
            if free[j]:
                rows.append(index[i])
                cols.append(index[j])
                vals.append(-w)
            else:
                rhs[index[i]] += w * pressure[j]

    laplacian = coo_matrix((vals, (rows, cols)), shape=(free.sum(),) * 2).tocsr()
    pressure[free] = spsolve(laplacian, rhs)
    return np.abs(conductance * (pressure[u] - pressure[v]))


def main(perturbation_um=VOXEL_UM):
    rng = np.random.default_rng(SEED)
    print(f"perturbation = {perturbation_um} um "
          f"({perturbation_um / VOXEL_UM:.2f} voxel); boundaries: face rule, axis "
          f"{BOUNDARY_AXIS}, tolerance {FACE_TOLERANCE} voxel\n")

    networks = {specimen_id: load_network(specimen_id) for specimen_id in SPECIMENS}

    print("=== S10: terminal-node census, and where the boundary nodes come from ===")
    print(f"{'spec':8}{'term':>6}{'on face':>9}{'interior':>10}{'inlet':>7}{'outlet':>8}"
          f"{'in:out':>9}{'stranded':>10}")
    for specimen_id, (G, u, v, length, diameter, index) in networks.items():
        total, on_face = terminals_on_any_face(G)
        inlets, outlets = face_boundaries(G, index)
        stranded = total - len(inlets) - len(outlets)
        print(f"{specimen_id:8}{total:6}{on_face:9}{total - on_face:10}"
              f"{len(inlets):7}{len(outlets):8}{len(inlets)/max(len(outlets),1):9.2f}"
              f"{stranded:7} ({100*stranded/total:.0f}%)")

    print("\n=== S11: is the solve well posed? ===")
    for specimen_id, (G, u, v, length, diameter, index) in networks.items():
        inlet_nodes, outlet_nodes = select_boundary_terminal_nodes_by_face(
            G, ROI, axis=BOUNDARY_AXIS, face_tolerance_voxels=FACE_TOLERANCE,
            voxel_size=PROCESSING_VOXEL_UM)
        solvable, total = edge_count_between_boundaries(G, set(inlet_nodes), set(outlet_nodes))
        print(f"{specimen_id:8} components={nx.number_connected_components(G):3}  "
              f"edges between an inlet and an outlet: {solvable}/{total} "
              f"({100*solvable/total:.1f}%)")

    print("\n=== S12 and S13: how correlated calibre error propagates ===")
    print(f"{'spec':8}{'independent':>13}{'correlated':>12}{'shunt ratio':>13}")
    independent_all, correlated_all, ratio_all = [], [], []
    for specimen_id, (G, u, v, length, diameter, index) in networks.items():
        inlets, outlets = face_boundaries(G, index)

        # Throughput is the inflow at the inlets, not the sum over every edge. Inlets are
        # degree-1 terminals carrying one edge each, so this is the network's total perfusion;
        # summing all edges instead would count each internal path again on the way through.
        at_inlet = np.isin(u, inlets) | np.isin(v, inlets)

        def total_and_ratio(d):
            flows = solve_edge_flows(u, v, length, d, inlets, outlets, len(index))
            return flows[at_inlet].sum(), flows[shunt].sum() / flows.sum()

        # The shunt set is fixed at baseline calibre so that the perturbation measures flow
        # redistribution, not edges being reclassified in or out of the set.
        shunt = diameter >= np.percentile(diameter, 90)
        base_total, base_ratio = total_and_ratio(diameter)

        independent = [
            total_and_ratio(np.clip(diameter + perturbation_um * rng.choice([-1, 1], len(diameter)), 0.5, None))[0]
            for _ in range(DRAWS)
        ]
        correlated = [total_and_ratio(np.clip(diameter + perturbation_um * sign, 0.5, None)) for sign in (-1, 1)]

        ind = 100 * np.std(independent) / base_total
        cor = 100 * (max(c[0] for c in correlated) - min(c[0] for c in correlated)) / 2 / base_total
        rat = 100 * (max(c[1] for c in correlated) - min(c[1] for c in correlated)) / 2 / base_ratio
        independent_all.append(ind)
        correlated_all.append(cor)
        ratio_all.append(rat)
        print(f"{specimen_id:8}{ind:12.1f}%{cor:11.1f}%{rat:12.1f}%")

    ind_mean, cor_mean, rat_mean = np.mean(independent_all), np.mean(correlated_all), np.mean(ratio_all)
    print(f"\nindependent error averages down {cor_mean/ind_mean:.0f}x relative to correlated "
          f"({ind_mean:.1f}% against {cor_mean:.1f}%)")
    print(f"a within-specimen ratio cancels {100*(1-rat_mean/cor_mean):.0f}% of the correlated error "
          f"({cor_mean:.1f}% -> {rat_mean:.1f}%)")
    return rat_mean


def boundary_sensitivity(calibre_ratio_pct):
    """S20: how much does the shunt ratio move when the boundary choice moves?

    ``calibre_ratio_pct`` is the within-specimen ratio error ``main`` just measured, so the
    comparison line below matches the perturbation that was run.

    S13 established that a within-specimen ratio cancels calibre error. That is only half the
    picture: if the ratio moves when the boundary choice moves, being a ratio does not save it.
    Under the face rule the choice has two parts. The axis has no anatomical basis in a mid-organ
    ROI, so its term is irreducible without new information; it is pinned by
    ``cb_settings.BOUNDARY_AXIS``. The face tolerance is anchored to the voxel size, and 2 and 4
    voxels are only there to show how much the answer depends on it. Axes are graph axes (z, y, x).
    """
    cases = (("axis0 tol1", 0, 1.0), ("axis1 tol1", 1, 1.0), ("axis2 tol1", 2, 1.0),
             ("axis1 tol2", 1, 2.0), ("axis1 tol4", 1, 4.0))

    def ratio_for(network, axis, tolerance):
        G, u, v, length, diameter, index = network
        try:
            inlets, outlets = face_boundaries(G, index, axis, tolerance)
        except ValueError:
            # A face with no terminals: this axis cannot carry a pressure boundary here.
            return None
        shunt = diameter >= np.percentile(diameter, 90)
        flows = solve_edge_flows(u, v, length, diameter, inlets, outlets, len(index))
        return flows[shunt].sum() / flows.sum()

    print("\n=== S20: shunt ratio against the boundary choice (face rule) ===")
    print(f"{'spec':8}" + "".join(f"{name:>12}" for name, _, _ in cases)
          + f"{'axis':>8}{'tol':>8}{'total':>8}")
    axis_spread, tol_spread, total_spread = [], [], []
    for specimen_id in SPECIMENS:
        network = load_network(specimen_id)
        values = [ratio_for(network, axis, tolerance) for _, axis, tolerance in cases]
        present = [v for v in values if v is not None]
        by_axis = [v for v in values[:3] if v is not None]
        # The tolerance triple must hold the axis fixed at 1: indices 1, 3 and 4.
        by_tol = [v for v in (values[1], values[3], values[4]) if v is not None]

        def full_range(xs):
            return 100.0 * (max(xs) - min(xs)) / np.mean(xs) if len(xs) > 1 else float("nan")

        axis_spread.append(full_range(by_axis))
        tol_spread.append(full_range(by_tol))
        total_spread.append(full_range(present))
        print(f"{specimen_id:8}"
              + "".join(f"{(v if v is not None else float('nan')):12.4f}" for v in values)
              + f"{axis_spread[-1]:7.1f}%{tol_spread[-1]:7.1f}%{total_spread[-1]:7.1f}%")

    axis_mean, tol_mean = np.nanmean(axis_spread), np.nanmean(tol_spread)
    print(f"\nmean spread of the shunt ratio: axis {axis_mean:.1f}%, "
          f"tolerance {tol_mean:.1f}%, combined {np.nanmean(total_spread):.1f}%")
    larger = "the boundary choice" if tol_mean > calibre_ratio_pct else "calibre error"
    print(f"Calibre error moves the same ratio {calibre_ratio_pct:.1f}% (S13, S15), against "
          f"{tol_mean:.1f}% for the face tolerance at the pinned axis, so {larger} is the larger term.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--perturbation-um", type=float, default=VOXEL_UM,
                        help=f"diameter perturbation in um (default {VOXEL_UM}, one voxel; "
                             f"pass {THRESHOLD_SHIFT_UM} for the measured threshold shift)")
    boundary_sensitivity(main(parser.parse_args().perturbation_um))
