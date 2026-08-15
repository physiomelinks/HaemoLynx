"""Measurements behind Part 1 of the H2 capability assessment.

Four questions, all answered from the exported H1 artefacts rather than from a pipeline run.

**Why solve the network here rather than read the pipeline's own flow output.** The pipeline has
run on all six specimens and its flow output exists, but it was produced with
``constrict_at_pericytes = True``, so 12.3% of edges carry a fabricated narrowing that inflates
their resistance by a median of about 12x (assessment finding S14). Building the conductance
network directly from measured calibre and length sidesteps that entirely, and isolates the
propagation question from every other defect in the chain.

**Two perturbation sizes, and why both are reported.** One voxel, 1.866 um, is the scale at which a
diameter difference stops being physically resolved: H1 section 8.2 disqualifies calibre as a
finding because the between-group gap sits at one twentieth of that step. It is the conservative
bound.

The threshold-calibrated size is the empirical one. Measured across the three sensitivity runs, the
median calibre moves 0.922 um over the clean 0.85 to 0.90 interval, about half a voxel, in the same
direction for all six specimens. Since the threshold is the dominant correlated error term, that is
the size of the correlated perturbation actually at play. Resistance goes as the inverse fourth
power of diameter, so the two do not simply scale, and the smaller one is measured rather than
inferred from the larger.

**Why independent and correlated are both run.** Resistance goes as the inverse fourth power of
diameter, so the per-edge uncertainty is near 94% at the median. Whether that matters at the network
level depends entirely on whether the errors cancel. They do when independent and do not when
correlated, and this pipeline's errors are correlated because every edge in a specimen comes from
one mask at one threshold.

Run with::

    venv/bin/python examples/cb_h2_error_propagation.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))

VTK = Path(__file__).resolve().parent / "outputs" / "cb_h1_paraview"
SPECIMENS = ("WKY-A", "WKY-B", "WKY-C", "SHR-A", "SHR-B", "SHR-C")

# Graph 'pos' is (z, y, x) and node_edge_axis defaults to 0, which is this coordinate index in the
# exported geometry. Verified against the mask rather than assumed: sampling the mask at every raw
# skeleton point gives 100.0% foreground in this frame and 24.4% in the transposed one.
AXIS = 2
EDGE_PERCENT = END_PERCENT = 25.0
VOXEL_UM = 1.866
# Median calibre shift over the clean 0.85 to 0.90 threshold interval, averaged over the six
# specimens. Measured by cb_h2_threshold_calibre.py, not assumed.
THRESHOLD_SHIFT_UM = 0.922
DRAWS = 24
SEED = 20260815


def load(specimen_id):
    """Edges, terminal nodes and ROI bounds for one specimen."""
    edges = pv.read(VTK / f"{specimen_id}_vessels.vtp")
    nodes = pv.read(VTK / f"{specimen_id}_nodes.vtp")
    mask = pv.read(VTK / f"{specimen_id}_mask.vti")

    u = np.asarray(edges.cell_data["edge_u"]).astype(int)
    v = np.asarray(edges.cell_data["edge_v"]).astype(int)
    length = np.asarray(edges.cell_data["length_um"], float)
    diameter = np.asarray(edges.cell_data["edt_diameter_um"], float)

    # Self-loops carry no pressure drop and a non-positive length or diameter would make the
    # conductance singular. Both are dropped rather than clamped, so nothing silently contributes.
    keep = (u != v) & np.isfinite(length) & (length > 0) & np.isfinite(diameter) & (diameter > 0)
    return (u[keep], v[keep], length[keep], diameter[keep], nodes, np.array(mask.bounds).reshape(3, 2))


def boundary_nodes(nodes, bounds):
    """Inlets and outlets exactly as select_boundary_terminal_nodes would choose them."""
    node_id = np.asarray(nodes.point_data["node_id"]).astype(int)
    degree = np.asarray(nodes.point_data["degree"]).astype(int)
    points = nodes.points
    low, high = bounds[AXIS]
    extent = high - low
    terminal = degree == 1
    coord = points[terminal][:, AXIS]
    inlets = node_id[terminal][coord <= low + extent * EDGE_PERCENT / 100.0]
    outlets = node_id[terminal][coord >= low + extent * (1.0 - END_PERCENT / 100.0)]
    return set(inlets.tolist()), set(outlets.tolist())


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
          f"({perturbation_um / VOXEL_UM:.2f} voxel)\n")

    print("=== S10: terminal-node census, and where the boundary nodes come from ===")
    print(f"{'spec':8}{'term':>6}{'on face':>9}{'interior':>10}{'inlet':>7}{'outlet':>8}"
          f"{'in:out':>9}{'stranded':>10}")
    for specimen_id in SPECIMENS:
        u, v, length, diameter, nodes, bounds = load(specimen_id)
        degree = np.asarray(nodes.point_data["degree"]).astype(int)
        points = nodes.points[degree == 1]
        on_face = np.zeros(len(points), bool)
        for axis in range(3):
            for side in range(2):
                on_face |= np.abs(points[:, axis] - bounds[axis, side]) <= VOXEL_UM
        inlets, outlets = boundary_nodes(nodes, bounds)
        total = len(points)
        stranded = total - len(inlets) - len(outlets)
        print(f"{specimen_id:8}{total:6}{on_face.sum():9}{(~on_face).sum():10}"
              f"{len(inlets):7}{len(outlets):8}{len(inlets)/max(len(outlets),1):9.2f}"
              f"{stranded:7} ({100*stranded/total:.0f}%)")

    print("\n=== S11: is the solve well posed? ===")
    for specimen_id in SPECIMENS:
        u, v, length, diameter, nodes, bounds = load(specimen_id)
        inlets, outlets = boundary_nodes(nodes, bounds)
        graph = nx.Graph()
        graph.add_edges_from(zip(u, v))
        components = list(nx.connected_components(graph))
        solvable = sum(
            graph.subgraph(c).number_of_edges()
            for c in components
            if (c & inlets) and (c & outlets)
        )
        total = graph.number_of_edges()
        print(f"{specimen_id:8} components={len(components):3}  "
              f"edges between an inlet and an outlet: {solvable}/{total} "
              f"({100*solvable/total:.1f}%)")

    print("\n=== S12 and S13: how correlated calibre error propagates ===")
    print(f"{'spec':8}{'independent':>13}{'correlated':>12}{'shunt ratio':>13}")
    independent_all, correlated_all, ratio_all = [], [], []
    for specimen_id in SPECIMENS:
        u0, v0, length, diameter, nodes, bounds = load(specimen_id)
        ids = np.unique(np.concatenate([u0, v0]))
        remap = {x: i for i, x in enumerate(ids)}
        u = np.array([remap[x] for x in u0])
        v = np.array([remap[x] for x in v0])
        inlet_ids, outlet_ids = boundary_nodes(nodes, bounds)
        inlets = np.array([remap[x] for x in inlet_ids if x in remap])
        outlets = np.array([remap[x] for x in outlet_ids if x in remap])

        # Throughput is the inflow at the inlets, not the sum over every edge. Inlets are
        # degree-1 terminals carrying one edge each, so this is the network's total perfusion;
        # summing all edges instead would count each internal path again on the way through.
        at_inlet = np.isin(u, inlets) | np.isin(v, inlets)

        def total_and_ratio(d):
            flows = solve_edge_flows(u, v, length, d, inlets, outlets, len(ids))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--perturbation-um", type=float, default=VOXEL_UM,
                        help=f"diameter perturbation in um (default {VOXEL_UM}, one voxel; "
                             f"pass {THRESHOLD_SHIFT_UM} for the measured threshold shift)")
    main(parser.parse_args().perturbation_um)
