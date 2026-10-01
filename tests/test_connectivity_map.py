"""The 2D connectivity map drawn from a connectivity CSV."""
from __future__ import annotations

import networkx as nx

from haemolynx.graph import connectivity_rows, write_connectivity_csv
from haemolynx.visualization import (
    connectivity_map_figure,
    connectivity_map_layout,
    read_connectivity_csv,
    write_connectivity_map,
)


def _example_rows():
    """Inlet 0 -> 1 -> 2, splitting to 5 and 6, merging at 7, -> outlet 8;
    a dead end 2 -> 9; a separate piece: inlet 20 -> outlet 22."""
    G = nx.MultiGraph()
    for u, v in [(0, 1), (1, 2), (2, 5), (2, 6), (5, 7), (6, 7), (7, 8), (2, 9), (20, 21), (21, 22)]:
        G.add_edge(u, v, length=10.0, diameter_um=5.0, branch_order="B1" if u < 20 else "B2")
    return connectivity_rows(G, [0, 20], [8, 22])


def test_layout_runs_from_the_inlets_to_the_outlets():
    layout = connectivity_map_layout(_example_rows())
    pos = layout.positions
    # The blank inlet/outlet cells are filled back in from the edge keys.
    assert layout.inlets == {"0", "20"} and layout.outlets == {"8", "22"}
    assert layout.dead_ends == {"9"}
    assert pos["0"][0] == 0 and pos["1"][0] == 1 and pos["2"][0] == 2
    assert pos["5"][0] == pos["6"][0] == 3 and pos["7"][0] == 4 and pos["8"][0] == 5
    assert pos["5"][1] != pos["6"][1]  # the two daughters do not overlap


def test_separate_pieces_are_stacked_not_overlapping():
    pos = connectivity_map_layout(_example_rows()).positions
    main = [pos[n][1] for n in ("0", "1", "2", "5", "6", "7", "8", "9")]
    other = [pos[n][1] for n in ("20", "21", "22")]
    assert max(other) < min(main)  # the smaller piece sits below the larger


def test_figure_hover_names_every_vessel_by_its_branch_id():
    rows = _example_rows()
    figure = connectivity_map_figure(rows, title="example")
    hovers = [t for trace in figure.data for t in (trace.hovertext or ()) if "Branch" in t]
    assert len(hovers) == len(rows)
    assert {h.split("</b>")[0].split()[-1] for h in hovers} == {r["Branch ID"] for r in rows}
    assert "10 vessels, 2 inlets, 2 outlets" in figure.layout.title.text


def test_write_connectivity_map_reads_the_csv_and_writes_html(tmp_path):
    csv_path = write_connectivity_csv(tmp_path / "net_connectivity.csv", _example_rows())
    assert read_connectivity_csv(csv_path)[0]["Branch ID"] == "0"
    html_path = write_connectivity_map(csv_path)
    assert html_path == tmp_path / "net_connectivity_map.html"
    text = html_path.read_text(encoding="utf-8")
    assert "plotly" in text and "net_connectivity" in text
