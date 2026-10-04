"""The 2D connectivity map drawn from a connectivity CSV.

The page lays the map out in its own script (``connectivity_map.js``), so most
of these run that script under Node and are skipped where Node is missing.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from importlib.resources import files

import networkx as nx
import pytest

from haemolynx.graph import CONNECTIVITY_COLUMNS, connectivity_rows, write_connectivity_csv
from haemolynx.visualization import (
    connectivity_map_html,
    read_connectivity_csv,
    write_connectivity_map,
)
from haemolynx.visualization.connectivity_map import map_config

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="the map's script runs under Node")

_RUNNER = """
const map = require(process.argv[1]);
let input = "";
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const { csv, config, command, args } = JSON.parse(input);
  const rows = map.parseCsv(csv);
  let out;
  if (command === "rows") out = rows;
  else if (command === "inlet") {
    const layout = map.inletLayout(rows);
    out = {
      positions: Object.fromEntries(layout.positions),
      inlets: [...layout.inlets].sort(), outlets: [...layout.outlets].sort(),
      deadEnds: [...layout.deadEnds].sort(),
    };
  } else if (command === "branch") out = map.branchLayout(rows, config);
  else if (command === "view") out = map.buildView(rows, args.view, config);
  else if (command === "stats") out = map.orderStats(rows, config);
  else if (command === "stats_csv") out = map.statsCsv(map.orderStats(rows, config));
  else if (command === "hover") {
    const built = map.buildView(rows, args.view, config);
    // Data units straight to pixels at a zoom of args.scale pixels per unit.
    const toPixel = (x, y) => [x * args.scale, -y * args.scale];
    const item = map.nearestItem(
      built.items, args.x * args.scale, -args.y * args.scale, toPixel, () => true
    );
    out = item && item.text;
  }
  process.stdout.write(JSON.stringify(out));
});
"""


def _csv_text(rows) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(CONNECTIVITY_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _run(rows_or_text, command, **args):
    text = rows_or_text if isinstance(rows_or_text, str) else _csv_text(rows_or_text)
    script = str(files("haemolynx.visualization").joinpath("connectivity_map.js"))
    done = subprocess.run(
        [NODE, "-e", _RUNNER, script],
        input=json.dumps({"csv": text, "config": map_config(), "command": command, "args": args}),
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(done.stdout)


def _example_rows():
    """Inlet 0 -> 1 -> 2, splitting to 5 and 6, merging at 7, -> outlet 8;
    a dead end 2 -> 9; a separate piece: inlet 20 -> outlet 22."""
    G = nx.MultiGraph()
    for u, v in [(0, 1), (1, 2), (2, 5), (2, 6), (5, 7), (6, 7), (7, 8), (2, 9), (20, 21), (21, 22)]:
        G.add_edge(u, v, length=10.0, diameter_um=5.0, branch_order="B1" if u < 20 else "B2")
    return connectivity_rows(G, [0, 20], [8, 22])


def _ordered_rows():
    """Inlet 0 -Art1-> 1 -Art2-> 2, splitting to 3 and 4 (B01), merging at 5,
    -Ven2-> 6 -Ven1-> outlet 7."""
    G = nx.MultiGraph()
    for u, v, order in [
        (0, 1, "Art1"), (1, 2, "Art2"), (2, 3, "B01"), (2, 4, "B01"),
        (3, 5, "B02"), (4, 5, "B02"), (5, 6, "Ven2"), (6, 7, "Ven1"),
    ]:
        G.add_edge(u, v, length=10.0, diameter_um=5.0, branch_order=order)
    return connectivity_rows(G, [0], [7])


@needs_node
def test_script_reads_the_csv_as_python_does(tmp_path):
    csv_path = write_connectivity_csv(tmp_path / "net_connectivity.csv", _example_rows())
    text = csv_path.read_text(encoding="utf-8")
    assert _run(text, "rows") == read_connectivity_csv(csv_path)
    assert '"' in text  # a quoted "a, b" upstream/downstream cell was parsed


@needs_node
def test_script_refuses_a_csv_that_is_not_a_connectivity_csv():
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run("a,b\n1,2\n", "rows")
    assert "not a connectivity CSV" in error.value.stderr


@needs_node
def test_layout_runs_from_the_inlets_to_the_outlets():
    layout = _run(_example_rows(), "inlet")
    pos = layout["positions"]
    # The blank inlet/outlet cells are filled back in from the edge keys.
    assert layout["inlets"] == ["0", "20"] and layout["outlets"] == ["22", "8"]
    assert layout["deadEnds"] == ["9"]
    assert pos["0"][0] == 0 and pos["1"][0] == 1 and pos["2"][0] == 2
    assert pos["5"][0] == pos["6"][0] == 3 and pos["7"][0] == 4 and pos["8"][0] == 5
    assert pos["5"][1] != pos["6"][1]  # the two daughters do not overlap


@needs_node
def test_separate_pieces_are_stacked_not_overlapping():
    pos = _run(_example_rows(), "inlet")["positions"]
    main = [pos[n][1] for n in ("0", "1", "2", "5", "6", "7", "8", "9")]
    other = [pos[n][1] for n in ("20", "21", "22")]
    assert max(other) < min(main)  # the smaller piece sits below the larger


@needs_node
@pytest.mark.parametrize("view", ["inlet", "branch_order"])
def test_hover_names_every_vessel_by_its_branch_id(view):
    rows = _example_rows()
    built = _run(rows, "view", view=view)
    texts = [i["text"] for i in built["items"] if i["kind"] == "vessel"]
    assert len(texts) == len(rows)  # one hover item per vessel
    assert {t.split("</b>")[0].split()[-1] for t in texts} == {r["Branch ID"] for r in rows}
    assert built["counts"] == {"vessels": 10, "inlets": 2, "outlets": 2}


@needs_node
@pytest.mark.parametrize("scale", [40, 4000])
def test_hovering_anywhere_along_a_vessel_finds_it_at_any_zoom(scale):
    # Branch 0 runs from inlet 0 (x=0) to node 1 (x=1); a point a quarter of
    # the way along is far from both ends and from the vessel's midpoint once
    # zoomed in (scale = pixels per column).
    rows = _example_rows()
    pos = _run(rows, "inlet")["positions"]
    (x0, y0), (x1, y1) = pos["0"], pos["1"]
    text = _run(rows, "hover", view="inlet", scale=scale,
                x=x0 + 0.25 * (x1 - x0), y=y0 + 0.25 * (y1 - y0))
    assert text.startswith("<b>Branch 0</b>")


@needs_node
def test_hovering_empty_space_finds_nothing():
    assert _run(_example_rows(), "hover", view="inlet", scale=400, x=0.5, y=50.0) is None


@needs_node
def test_branch_order_columns_run_from_the_inlets_to_the_outlets():
    layout = _run(_ordered_rows(), "branch")
    # Ven counts up from the outlet, so Ven2 sits before Ven1.
    assert layout["columns"] == ["Art1", "Art2", "B01", "B02", "Ven2", "Ven1"]
    x_of = {s["row"]["Branch order"]: (s["x0"] + s["x1"]) / 2 for s in layout["segments"]}
    assert [x_of[o] for o in layout["columns"]] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@needs_node
def test_branch_order_view_keeps_vessels_of_one_order_apart_in_their_column():
    segments = _run(_ordered_rows(), "branch")["segments"]
    b01 = [s["y"] for s in segments if s["row"]["Branch order"] == "B01"]
    assert len(b01) == 2 and abs(b01[0] - b01[1]) >= 1.0


@needs_node
@pytest.mark.parametrize("view", ["inlet", "branch_order"])
def test_inlet_and_outlet_vessels_are_drawn_bolder(view):
    traces = _run(_ordered_rows(), "view", view=view)["traces"]
    widths = {}
    for t in traces:
        if t["mode"] == "lines" and "legendgroup" in t:
            widths.setdefault(t["legendgroup"], set()).add(t["line"]["width"])
    assert widths["Art1"] == {6} and widths["Ven1"] == {6}
    assert widths["B01"] == {2}


@needs_node
def test_branch_order_view_labels_its_columns():
    xaxis = _run(_ordered_rows(), "view", view="branch_order")["xaxis"]
    assert xaxis["ticktext"] == ["Art1", "Art2", "B01", "B02", "Ven2", "Ven1"]


@needs_node
def test_branch_order_stats_per_order_then_all():
    rows = _ordered_rows()
    # Lengths 10..80 and diameters 2..9 along the path, so the means differ per order.
    for i, row in enumerate(sorted(rows, key=lambda r: int(r["Branch ID"]))):
        row["Length (um)"] = str(10.0 * (i + 1))
        row["Diameter (um)"] = str(2.0 + i)
    stats = {s["order"]: s for s in _run(rows, "stats")}
    assert list(stats) == ["Art1", "Art2", "B01", "B02", "Ven2", "Ven1", "All"]
    by_order = {}
    for row in rows:
        by_order.setdefault(row["Branch order"], []).append(row)
    for order, group in by_order.items():
        s = stats[order]
        assert s["vessels"] == len(group)
        assert s["length"] == pytest.approx(sum(float(r["Length (um)"]) for r in group) / len(group))
        assert s["diameter"] == pytest.approx(sum(float(r["Diameter (um)"]) for r in group) / len(group))
    # The two B01 daughters each have one vessel in (Art2) and one out (a B02).
    assert stats["B01"]["upstream"] == 1 and stats["B01"]["downstream"] == 1
    # Ven2 is fed by both B02s and feeds Ven1: connections = (2 + 1) / 2.
    assert stats["Ven2"]["upstream"] == 2 and stats["Ven2"]["connections"] == 1.5
    # The inlet vessel has nothing upstream; the outlet vessel nothing downstream.
    assert stats["Art1"]["upstream"] == 0 and stats["Art1"]["inlets"] == 1
    assert stats["Ven1"]["downstream"] == 0 and stats["Ven1"]["outlets"] == 1
    assert stats["All"]["vessels"] == len(rows) == 8
    assert stats["All"]["length"] == pytest.approx(45.0)


@needs_node
def test_branch_order_stats_leave_blank_lengths_out_of_the_mean():
    rows = _ordered_rows()
    for row in rows:
        if row["Branch order"] == "B01":
            row["Length (um)"] = ""
            break
    b01 = next(s for s in _run(rows, "stats") if s["order"] == "B01")
    assert b01["vessels"] == 2 and b01["length"] == pytest.approx(10.0)


@needs_node
def test_branch_order_stats_save_as_csv():
    text = _run(_ordered_rows(), "stats_csv")
    table = list(csv.DictReader(io.StringIO(text)))
    assert [r["Branch order"] for r in table][-1] == "All"
    assert table[0]["Mean connections (in + out) / 2"] == "0.5000"  # Art1: (0 + 1) / 2


def test_write_connectivity_map_embeds_the_csv_and_the_script(tmp_path):
    csv_path = write_connectivity_csv(tmp_path / "net_connectivity.csv", _example_rows())
    html_path = write_connectivity_map(csv_path)
    assert html_path == tmp_path / "net_connectivity_map.html"
    text = html_path.read_text(encoding="utf-8")
    assert "cdn.plot.ly/plotly-" in text and "HaemoLynxConnectivityMap.start(" in text
    assert "Load CSV" in text  # the page can draw another CSV
    start = text.index('<script id="hl-data" type="application/json">') + len(
        '<script id="hl-data" type="application/json">'
    )
    data = json.loads(text[start:text.index("</script>", start)])
    assert data["csv"] == csv_path.read_text(encoding="utf-8")
    assert data["name"] == "net_connectivity" and data["view"] == "inlet"


def test_embedded_csv_cannot_close_its_script_element():
    page = connectivity_map_html("Branch ID,Notes\n0,</script><b>x</b>\n", name="odd")
    assert page.count("</script>") == 4  # plotly, data, map script, start


def test_unknown_view_is_refused():
    with pytest.raises(ValueError):
        connectivity_map_html("", view="sideways")


def test_the_module_is_valid_python_3_9():
    """A backslash inside an f-string's braces parses on 3.12 only; on 3.9 to
    3.11 it stopped the whole package importing."""
    import ast
    from pathlib import Path

    import haemolynx.visualization.connectivity_map as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 9))
