"""The connectivity table behind tab 10's "Export connectivity CSV".

Pinned on the network the format was specified with: an inlet vessel into a
diverging bifurcation, two vessels merging again, and an outlet vessel.
"""
from __future__ import annotations

import csv

import networkx as nx
import pytest

from haemolynx.graph import (
    CONNECTIVITY_COLUMNS,
    connectivity_rows,
    inlet_to_outlet_vessels,
    write_connectivity_csv,
)
from haemolynx.graph.connectivity import unsolved_pieces


def _example() -> nx.MultiGraph:
    """Inlet 0 -> 1 -> 2, splitting to 5 and 6, merging at 7, -> outlet 8."""
    G = nx.MultiGraph()
    for u, v, length, order in [
        (0, 1, 300.0, "arteriole"),
        (1, 2, 200.0, "1"),
        (2, 5, 100.0, "3"),
        (2, 6, 100.0, "3"),
        (5, 7, 100.0, "4"),
        (6, 7, 70.0, "4"),
        (7, 8, 200.0, "outlet venule"),
    ]:
        G.add_edge(u, v, length=length, diameter_um=5.0, branch_order=order)
    return G


def _by_id(rows) -> dict:
    return {int(row["Branch ID"]): row for row in rows}


def test_rows_follow_the_network_from_inlet_to_outlet():
    rows = connectivity_rows(_example(), [0], [8])
    assert [r["Branch ID"] for r in rows] == ["0", "1", "2", "3", "4", "5", "6"]
    assert [(r["From node ID"], r["To node ID"]) for r in rows] == [
        ("", "1"), ("1", "2"), ("2", "5"), ("2", "6"), ("5", "7"), ("6", "7"), ("7", ""),
    ]
    assert rows[0]["Notes"] == "Inlet" and rows[-1]["Notes"] == "Outlet"
    assert [r["Length (um)"] for r in rows] == ["300", "200", "100", "100", "100", "70", "200"]
    assert rows[0]["Diameter (um)"] == "5" and rows[0]["Branch order"] == "arteriole"


def test_a_split_lists_both_daughters_and_a_merge_both_parents():
    rows = _by_id(connectivity_rows(_example(), [0], [8]))
    assert rows[1]["Downstream branch IDs"] == "2, 3"   # 1 -> 2 diverges
    assert rows[6]["Upstream branch IDs"] == "4, 5"     # 5 -> 7 and 6 -> 7 converge
    assert rows[2]["Upstream branch IDs"] == "1"
    assert rows[0]["Upstream branch IDs"] == "" and rows[6]["Downstream branch IDs"] == ""


def test_ids_are_the_ones_the_pkl_and_vtk_exports_use():
    G = _example()
    G.add_edge(5, 7, length=100.0)  # a parallel vessel: key 1
    rows = _by_id(connectivity_rows(G, [0], [8]))
    for branch_id, (u, v, k) in enumerate(G.edges(keys=True)):
        row = rows[branch_id]
        assert (row["Edge u"], row["Edge v"], row["Edge key"]) == (str(u), str(v), str(k))
        assert G.has_edge(int(row["Edge u"]), int(row["Edge v"]), key=int(row["Edge key"]))
    parallel = [r for r in rows.values() if r["Edge key"] == "1"]
    assert len(parallel) == 1 and parallel[0]["From node ID"] == "5"


def test_orientation_follows_path_length_not_vessel_count():
    # Inlet 0 reaches 3 by one 100 um vessel or by two 10 um ones via 2:
    # 2 is nearer the inlet than 3, so the vessel between them runs 2 -> 3.
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=5.0)
    G.add_edge(1, 3, length=100.0)
    G.add_edge(1, 2, length=10.0)
    G.add_edge(3, 2, length=10.0)
    G.add_edge(3, 4, length=5.0)
    rows = connectivity_rows(G, [0], [4])
    between = next(r for r in rows if {r["Edge u"], r["Edge v"]} == {"2", "3"})
    assert (between["From node ID"], between["To node ID"]) == ("2", "3")


def test_dead_ends_unconnected_pieces_self_loops_and_roles_are_noted():
    G = _example()
    G.add_edge(2, 9, length=20.0)       # dead end off the split
    G.add_edge(20, 21, length=5.0)      # a piece no inlet reaches
    G.add_edge(7, 7, length=3.0)        # self-loop
    rows = connectivity_rows(G, [0], [8], arteriole_boundary_nodes=[0], venule_boundary_nodes=[8])
    find = {(r["Edge u"], r["Edge v"]): r for r in rows}
    assert find[("2", "9")]["Notes"] == "Dead end"
    assert find[("2", "9")]["To node ID"] == "9"
    assert "Not connected to an inlet" in find[("20", "21")]["Notes"]
    assert (find[("20", "21")]["From node ID"], find[("20", "21")]["To node ID"]) == ("20", "21")
    assert rows[-1]["Edge u"] == "20"  # unreached pieces come last
    assert find[("7", "7")]["Notes"] == "Self-loop"
    assert find[("0", "1")]["Notes"] == "Inlet; arteriole boundary"
    assert find[("7", "8")]["Notes"] == "Outlet; venule boundary"


def test_an_inlet_at_a_junction_keeps_its_node_id():
    # Inlet 1 sits where three vessels meet: it is no open tip, so it is written.
    G = nx.MultiGraph()
    for u, v in [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5)]:
        G.add_edge(u, v, length=10.0)
    rows = connectivity_rows(G, [1], [5])
    first = next(r for r in rows if r["Edge v"] == "2")
    assert first["From node ID"] == "1" and "Inlet" in first["Notes"]


def test_without_an_inlet_every_row_says_so_and_nothing_is_reoriented():
    rows = connectivity_rows(_example(), [], [8])
    assert all("Not connected to an inlet" in r["Notes"] for r in rows)
    assert [r["Branch ID"] for r in rows] == [str(i) for i in range(7)]


def test_write_connectivity_csv_round_trips(tmp_path):
    rows = connectivity_rows(_example(), [0], [8])
    path = write_connectivity_csv(tmp_path / "net_connectivity.csv", rows)
    with path.open(newline="", encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert tuple(read[0].keys()) == CONNECTIVITY_COLUMNS
    assert read == rows


@pytest.mark.parametrize("value, text", [(5.0, "5"), (4.25, "4.25"), (1 / 3, "0.333"), (None, "")])
def test_numbers_are_written_plainly(value, text):
    G = nx.MultiGraph()
    G.add_edge(0, 1, length=1.0, diameter_um=value)
    assert connectivity_rows(G, [0], [1])[0]["Diameter (um)"] == text


# --- only the vessels between an inlet and an outlet -------------------------


def _with_dead_ends() -> nx.MultiGraph:
    """The example plus a dead-end branch 2 -> 9 -> 10 with a loop 9-11-12-9
    hanging off it, a self-loop at 7 and a piece 20 - 21 no inlet reaches."""
    G = _example()
    for u, v in [(2, 9), (9, 10), (9, 11), (11, 12), (12, 9), (20, 21)]:
        G.add_edge(u, v, length=10.0, diameter_um=4.0)
    G.add_edge(7, 7, length=3.0)
    return G


def test_inlet_to_outlet_vessels_keeps_only_what_blood_can_cross():
    G = _with_dead_ends()
    kept = {frozenset(e[:2]) for e in inlet_to_outlet_vessels(G, [0], [8])}
    assert kept == {frozenset(p) for p in [(0, 1), (1, 2), (2, 5), (2, 6), (5, 7), (6, 7), (7, 8)]}
    assert inlet_to_outlet_vessels(G, [0], []) == set()


def test_parallel_vessels_between_inlet_and_outlet_are_both_kept():
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 2), (1, 2)])
    assert len(inlet_to_outlet_vessels(G, [0], [2])) == 3


def test_only_inlet_to_outlet_drops_dead_ends_and_keeps_the_ids():
    G = _with_dead_ends()
    every = _by_id(connectivity_rows(G, [0], [8]))
    rows = connectivity_rows(G, [0], [8], only_inlet_to_outlet=True)
    assert len(rows) == 7
    assert not any("Dead end" in r["Notes"] or "Self-loop" in r["Notes"] for r in rows)
    for row in rows:  # the same Branch IDs and edge keys as the full export
        same = every[int(row["Branch ID"])]
        assert (row["Edge u"], row["Edge v"], row["Edge key"]) == (
            same["Edge u"], same["Edge v"], same["Edge key"]
        )
    split = next(r for r in rows if (r["From node ID"], r["To node ID"]) == ("1", "2"))
    # The dead end 2 -> 9 is no longer one of the vessels leaving node 2.
    to_9 = next(b for b, r in every.items() if {r["Edge u"], r["Edge v"]} == {"2", "9"})
    assert str(to_9) not in split["Downstream branch IDs"].split(", ")
    assert len(split["Downstream branch IDs"].split(", ")) == 2


def test_an_outlet_left_as_a_tip_by_the_filter_is_written_blank():
    # Outlet 3 also has a dead end hanging off it: in the filtered table it is
    # the open tip of its vessel.
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (1, 3), (3, 4)])
    rows = connectivity_rows(G, [0], [3], only_inlet_to_outlet=True)
    last = next(r for r in rows if "Outlet" in r["Notes"])
    assert last["To node ID"] == ""
    full = connectivity_rows(G, [0], [3])
    assert next(r for r in full if "Outlet" in r["Notes"])["To node ID"] == "3"


# --- unsolved_pieces: what a solve leaves unsolved, piece by piece ----------


def test_unsolved_vessels_come_in_pieces_longest_first():
    G = _example()
    G.add_edge(2, 9, length=20.0)       # a dead end off the split
    G.add_edge(9, 10, length=15.0)      # ... and on, a tree
    G.add_edge(9, 11, length=5.0)
    G.add_edge(2, 12, length=4.0)       # a second dead end off the same node
    G.add_edge(20, 21, length=50.0)     # a piece no inlet reaches
    G.add_edge(7, 7, length=3.0)        # a self-loop on a solved node

    pieces = unsolved_pieces(G, [0], [8])

    assert [p.length_um for p in pieces] == [50.0, 40.0, 4.0, 3.0]
    island, tree, stub, loop = pieces
    assert island.disconnected and island.attached_at == () and set(island.nodes) == {20, 21}
    assert tree.attached_at == (2,) and len(tree.edges) == 3 and not tree.disconnected
    assert stub.attached_at == (2,) and len(stub.edges) == 1
    assert loop.attached_at == (7,) and loop.edges == ((7, 7, 0),)
    solved = inlet_to_outlet_vessels(G, [0], [8])
    assert not solved & {e for p in pieces for e in p.edges}
    assert len(solved) + sum(len(p.edges) for p in pieces) == G.number_of_edges()


def test_a_network_with_every_vessel_solved_has_no_pieces_and_one_without_an_outlet_is_all_one():
    assert unsolved_pieces(_example(), [0], [8]) == []
    (everything,) = unsolved_pieces(_example(), [0], [])
    assert len(everything.edges) == 7 and everything.disconnected
