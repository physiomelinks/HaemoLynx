"""Tests for the Secomb ``Network.dat`` exporter and reader.

NetFlow and FlowEstimate take a segment's length from its two nodes, so what
has to be exact is that each vessel's chain of segments is as long as the
vessel, carries its diameter, runs from u to v, and that the boundary list
says what a solve needs: pressures (in mmHg) at the inlets and outlets,
closed dead ends as zero flows, unknown ends left out. The last test solves
the written file with an independent Poiseuille solve and gets HaemoLynx's
flows back.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.visualization import read_secomb_network, write_secomb_network
from haemolynx.visualization.secomb_io import BC_FLOW, BC_PRESSURE, NL_PER_MIN_PER_M3_S, PA_PER_MMHG


def _arc(start, end, bulge, n=25):
    """Points from *start* to *end* bowed by *bulge* um along y, (z, y, x)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    points = start + t * (end - start)
    points[:, 1] += bulge * np.sin(np.pi * t[:, 0])
    return points


def _polyline_length(points) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1)))


def _y_network() -> nx.MultiGraph:
    """An inlet vessel dividing into two curved daughters, each to an outlet.

    0 --(straight, 10 um)-- 1 --(curved, 6 um)-- 2
                              \\-(curved, 4 um)-- 3
    """
    G = nx.MultiGraph()
    positions = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 50.0), 2: (10.0, 0.0, 120.0), 3: (-10.0, 0.0, 130.0)}
    for node, pos in positions.items():
        G.add_node(node, pos=np.asarray(pos))
    G.add_edge(0, 1, voxels=[positions[0], positions[1]], length=50.0, diameter_um=10.0, conductance=1.0)
    for v, bulge, diameter in ((2, 15.0, 6.0), (3, -20.0, 4.0)):
        voxels = _arc(positions[1], positions[v], bulge)
        G.add_edge(1, v, voxels=voxels.tolist(), length=_polyline_length(voxels), diameter_um=diameter, conductance=1.0)
    return G


def _write(G, tmp_path, **kwargs):
    defaults = dict(inlet_nodes=[0], outlet_nodes=[2, 3], inlet_pressure_pa=6000.0, outlet_pressure_pa=1000.0)
    defaults.update(kwargs)
    export = write_secomb_network(G, tmp_path / "Network.dat", **defaults)
    return export, read_secomb_network(tmp_path / "Network.dat")


def _chain_length(network, names) -> float:
    total = 0.0
    for name in names:
        _type, start, end, *_rest = network.segments[name]
        total += math.dist(network.nodes[start], network.nodes[end])
    return total


def test_each_vessel_is_a_chain_as_long_as_the_vessel(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path)
    for (u, v, key), names in export.edge_segments.items():
        assert _chain_length(network, names) == pytest.approx(G.edges[u, v, key]["length"], rel=1e-7)
        assert export.edge_lengths_um[(u, v, key)] == pytest.approx(G.edges[u, v, key]["length"], rel=1e-7)


def test_a_chain_runs_from_u_to_v_with_the_vessels_diameter(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path)
    for (u, v, key), names in export.edge_segments.items():
        first, last = network.segments[names[0]], network.segments[names[-1]]
        assert first[1] == export.node_names[u]
        assert last[2] == export.node_names[v]
        for previous, following in zip(names, names[1:]):
            assert network.segments[previous][2] == network.segments[following][1]
        assert {network.segments[n][3] for n in names} == {G.edges[u, v, key]["diameter_um"]}
        assert all(network.segments[n][0] == 5 for n in names)


def test_segments_are_at_most_the_step_long(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path, max_segment_um=5.0)
    for names in export.edge_segments.values():
        for name in names:
            _type, start, end, *_rest = network.segments[name]
            # The chain is bent to the vessel's length, so a step can stretch
            # a little past the resampling step.
            assert math.dist(network.nodes[start], network.nodes[end]) < 5.5


def test_coordinates_are_written_x_y_z_from_pos_z_y_x(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path)
    assert network.nodes[export.node_names[2]] == pytest.approx((120.0, 0.0, 10.0))
    assert network.nodes[export.node_names[3]] == pytest.approx((130.0, 0.0, -10.0))


def test_a_straight_vessel_longer_than_its_chord_is_bent_to_its_length(tmp_path):
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 30.0))
    G.add_edge(0, 1, voxels=[(0.0, 0.0, 0.0), (0.0, 0.0, 30.0)], length=40.0, diameter_um=5.0, conductance=1.0)
    export, network = _write(G, tmp_path, outlet_nodes=[1])
    assert _chain_length(network, export.edge_segments[(0, 1, 0)]) == pytest.approx(40.0, rel=1e-7)
    assert export.shorter_than_chord == 0


def test_a_vessel_shorter_than_its_chord_is_written_straight_and_counted(tmp_path):
    G = nx.MultiGraph()
    G.add_node(0, pos=(0.0, 0.0, 0.0))
    G.add_node(1, pos=(0.0, 0.0, 30.0))
    G.add_edge(0, 1, length=20.0, diameter_um=5.0, conductance=1.0)
    export, network = _write(G, tmp_path, outlet_nodes=[1])
    assert _chain_length(network, export.edge_segments[(0, 1, 0)]) == pytest.approx(30.0)
    assert export.shorter_than_chord == 1


def test_inlets_and_outlets_are_pressures_in_mmhg(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path, boundary_haematocrit=0.4)
    inlet = network.boundary[export.node_names[0]]
    assert inlet[0] == BC_PRESSURE
    assert inlet[1] == pytest.approx(6000.0 / PA_PER_MMHG)
    assert inlet[2] == pytest.approx(0.4)
    for outlet in (2, 3):
        assert network.boundary[export.node_names[outlet]][:2] == (BC_PRESSURE, pytest.approx(1000.0 / PA_PER_MMHG))
    # NetFlow's own mmHg, so its flow factor gives back the same flows.
    export, network = _write(G, tmp_path, pa_per_mmhg=133.3)
    assert network.boundary[export.node_names[0]][1] == pytest.approx(6000.0 / 133.3)


def test_other_dead_ends_are_closed_and_unknown_ones_left_out(tmp_path):
    G = _y_network()
    G.add_node(4, pos=(0.0, 20.0, 50.0))
    G.add_node(5, pos=(0.0, -20.0, 25.0))
    G.add_edge(1, 4, length=20.0, diameter_um=3.0, conductance=1.0)
    G.add_edge(0, 5, length=math.dist((0.0, -20.0, 25.0), (0.0, 0.0, 0.0)), diameter_um=3.0, conductance=1.0)
    with pytest.raises(ValueError, match="one vessel"):
        _write(G, tmp_path)  # inlet 0 is now on two vessels
    G.remove_edge(0, 5)
    G.remove_node(5)
    export, network = _write(G, tmp_path, outlet_nodes=[2], unknown_boundary_nodes=[3])
    assert network.boundary[export.node_names[4]] == (BC_FLOW, 0.0, pytest.approx(0.45))
    assert export.node_names[3] not in network.boundary
    assert export.closed_nodes == [4] and export.unknown_nodes == [3]


def test_a_node_cannot_be_both_designated_and_unknown(tmp_path):
    with pytest.raises(ValueError, match="both"):
        _write(_y_network(), tmp_path, unknown_boundary_nodes=[2])


def test_self_loops_and_vessels_without_conductance_are_left_out(tmp_path):
    G = _y_network()
    G.add_edge(1, 1, length=30.0, diameter_um=5.0, conductance=1.0)
    G.add_edge(2, 3, length=20.0, diameter_um=5.0)
    export, network = _write(G, tmp_path)
    assert export.skipped == {"self_loop": 1, "no_conductance": 1}
    assert len(export.edge_segments) == 3


def test_a_vessel_with_conductance_needs_a_diameter(tmp_path):
    G = _y_network()
    del G.edges[0, 1, 0]["diameter_um"]
    with pytest.raises(ValueError, match="diameter_um"):
        _write(G, tmp_path)


def test_bridges_are_written_as_vessels_and_counted(tmp_path):
    G = _y_network()
    G.edges[1, 2, 0][IS_ZERO_RESISTANCE] = True
    export, network = _write(G, tmp_path)
    assert export.bridges_as_vessels == 1
    assert _chain_length(network, export.edge_segments[(1, 2, 0)]) == pytest.approx(G.edges[1, 2, 0]["length"], rel=1e-7)


def test_flows_are_written_in_nl_per_min_and_can_order_the_vessels(tmp_path):
    G = _y_network()
    for (u, v), flow in {(0, 1): 3e-14, (1, 2): 1e-14, (1, 3): -2e-14}.items():
        G.edges[u, v, 0]["flow_signed"] = flow
    export, network = _write(G, tmp_path)
    assert network.segments[export.edge_segments[(1, 3, 0)][0]][4] == pytest.approx(-2e-14 * NL_PER_MIN_PER_M3_S)
    export, network = _write(G, tmp_path, order_by_flow=True)
    firsts = [export.edge_segments[e][0] for e in ((0, 1, 0), (1, 3, 0), (1, 2, 0))]
    assert firsts == sorted(firsts)


def test_the_file_is_laid_out_as_netflow_reads_it(tmp_path):
    G = _y_network()
    export, network = _write(G, tmp_path)
    lines = (tmp_path / "Network.dat").read_text().splitlines()
    n_segments = sum(len(names) for names in export.edge_segments.values())
    # Title, five lines NetFlow skips, the segment count on the seventh line
    # and one header line under it (input.cpp).
    assert int(lines[5].split()[0]) == 3  # most segments at one node
    assert int(lines[6].split()[0]) == n_segments
    assert lines[8 + n_segments].split()[0] == str(len(network.nodes))
    assert len(network.boundary) == 3


def test_the_reader_takes_netflows_own_layout(tmp_path):
    text = "\n".join([
        "Network of 28-10-90",
        "4566. 7071. 20. box dimensions in microns",
        "92 142 1 number of tissue points in x,y,z directions",
        "100.\touter bound distance",
        "150.\tmax. segment length",
        "4\t\tmaximum number of segments per node",
        "2\ttotal number of segments",
        "SegName Type StartNode EndNode Diam   Flow[nl/min]    Hd",
        "1 5 830 1 27.650000 362.559998 0.433800 *",
        "2 5 1 5001 23.110001 344.230255 0.445569 *",
        "3 total number of nodes",
        "Name x y z",
        "830 0. 0. 10. *",
        "1 100. 0. 10. *",
        "5001 100. 50. 10. *",
        "2 Total number of boundary nodes",
        "Node\t Bctype\t Press/Flow\t HD\t PO2",
        "830 2 362.559998 0.433800 40.000000 *",
        "5001 0 13.800000 0.550000 40.000000 *",
    ])
    (tmp_path / "net.dat").write_text(text + "\n")
    network = read_secomb_network(tmp_path / "net.dat")
    assert network.segments[2] == (5, 1, 5001, pytest.approx(23.110001), pytest.approx(344.230255), pytest.approx(0.445569))
    assert network.nodes[5001] == (100.0, 50.0, 10.0)
    assert network.boundary == {830: (2, pytest.approx(362.559998), pytest.approx(0.4338)),
                                5001: (0, pytest.approx(13.8), pytest.approx(0.55))}


def test_the_written_file_solves_to_haemolynxs_flows(tmp_path):
    """Solve the file as NetFlow does -- Poiseuille on every segment, its
    length from its nodes, the pressures at the boundary nodes -- and get
    HaemoLynx's own solve back."""
    from haemolynx import haemodynamics as hd
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig, apply_poiseuille_resistances
    from haemolynx.haemodynamics.viscosity import pries_in_vivo_viscosity

    G = _y_network()
    for _u, _v, data in G.edges(data=True):
        data["branch_order"] = "B"
    config = HaemodynamicsApplyConfig(diameters={
        "diameter_by_branch_order": {"B": 5.0}, "viscosity_law": "pries", "pries_viscosity_form": "in_vivo_1994",
        "haematocrit": 0.45,
    })
    apply_poiseuille_resistances(G, config)
    conductance, nodes = hd.build_conductance_matrix_from_graph(G, solver="sparse")
    solved = hd.solve_flow_from_conductance_matrix(conductance, nodes, inlet_p_bc=6000.0, outlet_p_bc=1000.0,
                                                   inlet_nodes=[0], outlet_nodes=[2, 3])
    hd.set_edge_flows(G, nodes, solved["pressure"])
    export, network = _write(G, tmp_path)

    names = sorted(network.nodes)
    index = {name: i for i, name in enumerate(names)}
    laplacian = np.zeros((len(names), len(names)))
    segment_conductance = {}
    for name, (_type, start, end, diameter, _q, _hd) in network.segments.items():
        length_m = math.dist(network.nodes[start], network.nodes[end]) * 1e-6
        g = math.pi * (diameter * 1e-6) ** 4 / (128.0 * pries_in_vivo_viscosity(diameter, 0.45) * length_m)
        segment_conductance[name] = g
        i, j = index[start], index[end]
        laplacian[[i, j], [i, j]] += g
        laplacian[i, j] -= g
        laplacian[j, i] -= g
    held = {index[n]: value * PA_PER_MMHG for n, (kind, value, _h) in network.boundary.items() if kind == BC_PRESSURE}
    free = [i for i in range(len(names)) if i not in held]
    pressure = np.zeros(len(names))
    for i, value in held.items():
        pressure[i] = value
    known = list(held)
    pressure[free] = np.linalg.solve(laplacian[np.ix_(free, free)], -laplacian[np.ix_(free, known)] @ pressure[known])

    for (u, v, key), chain in export.edge_segments.items():
        _type, start, end, *_rest = network.segments[chain[0]]
        theirs = segment_conductance[chain[0]] * (pressure[index[start]] - pressure[index[end]])
        assert theirs == pytest.approx(G.edges[u, v, key]["flow_signed"], rel=1e-6)
