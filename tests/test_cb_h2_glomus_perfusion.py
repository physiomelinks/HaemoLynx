"""The per-edge rows behind the glomus-perfusion table (shunting and transit time)."""
import networkx as nx
import pytest

from cb_h2_glomus_perfusion import edge_rows


def _solved_edge():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, flow_abs=2e-6, hematocrit=0.4, assigned_diameter_um=6.0,
               length=30.0)
    return G


def test_a_solved_edge_gives_one_row_of_its_quantities():
    rows = edge_rows(_solved_edge(), {(0, 1, 0): 0.5}, {0: 0.0, 1: 3.0})
    assert rows == [{"th_fraction": 0.5, "flow": 2e-6, "hematocrit": 0.4, "diameter": 6.0,
                     "length": 30.0, "transit": 3.0}]


@pytest.mark.parametrize("attr", ["flow_abs", "hematocrit", "assigned_diameter_um", "length"])
def test_an_edge_missing_a_solved_quantity_raises_naming_it(attr):
    """These used to read .get(attr, nan), and length .get("length", 0.0)."""
    G = _solved_edge()
    del G.edges[0, 1, 0][attr]
    with pytest.raises(ValueError, match=rf"edge \(0, 1, 0\) has no '{attr}'"):
        edge_rows(G, {(0, 1, 0): 0.5}, {0: 0.0, 1: 3.0})
