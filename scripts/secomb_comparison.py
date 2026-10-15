#!/usr/bin/env python3
"""HaemoLynx's flow solve against Tim Secomb's reference codes (the release
review's ``flow.8a``).

NetFlow (github.com/secomb/NetFlowV2) solves a network with the Pries 1994
in vivo viscosity law and phase separation at diverging bifurcations;
FlowEstimate (github.com/secomb/FlowEstimateV1) estimates unknown boundary
conditions as Fry et al. (2012) do. HaemoLynx re-implements both. This script
solves one network with HaemoLynx and with them, and reports how far apart
the flows, pressures and discharge haematocrits are.

Neither code has an open-source licence ("free to use at your own risk",
FlowEstimate "for non-commercial use"), so neither is vendored here: clone
them outside the repository and ``build`` compiles copies of them. The copies
change what is *written*, never what is computed:

- ``writeflow.cpp`` prints flows and haematocrits with ``%.9e`` instead of
  ``%f`` (NetFlow, six decimals in nl/min) or ``%7.3f`` (FlowEstimate, three),
  which leave a capillary's flow of 1e-3 nl/min with one significant figure;
- the same function also writes each node's pressure
  (``Current/NodePressures.out``), which neither prints;
- a second NetFlow is built with ``dishemversion = 0`` (``flow.cpp``), its own
  switch between ``dishem.cpp`` and the default ``dishem_generalized.cpp``.

One more FlowEstimate is built that *does* compute differently, as a
diagnostic and never as the reference: its bias factor kappa held at 1
(``flow.cpp``), to show how much of its difference from HaemoLynx is kappa's.

**The physics is matched before comparing** (``compare``):

- viscosity: Pries et al. 1994 in vivo, what NetFlow uses
  (``pries_viscosity_form=in_vivo_1994``); NetFlow's plasma viscosity is set
  to HaemoLynx's 1.2 mPa.s and its mean cell volume to 92 fl, so its
  ``(92/MCV)^(1/3)`` diameter scaling is 1;
- pressures are written with NetFlow's own mmHg (1333 dyn/cm^2, ``facfp``),
  so a pressure drop gives both programs the same flow;
- every vessel off an inlet-to-outlet path is removed first (NetFlow needs a
  complete set of boundary conditions, and a dead end's rounding-sized flow
  would be a "bifurcation" to either program), and zero-resistance bridges are
  solved as vessels of their own diameter and length in both, since a segment
  in Secomb's format takes its length from its nodes (see
  :mod:`haemolynx.visualization.secomb_io`);
- haematocrit fixed at 0.45, then distributed (``distributed_iterative``)
  with each junction rule against the NetFlow that matches it:
  ``sequential_bifurcations`` against NetFlow as shipped
  (``dishem_generalized``), ``no_separation`` against ``dishemversion = 0``;
  both iterate with NetFlow's relaxation schedule, to tight tolerances, and
  the file lists the vessels largest flow first so the generalised rule takes
  a junction's outflows in the order HaemoLynx does;
- NetFlow's linear solver stops at ``SOR_TOLERANCE_MMHG``, not its sample's
  1e-10 mmHg, which leaves near-stagnant vessels unresolved.

FlowEstimate is compared with ``open_end_model=estimated`` on a network whose
open ends (dead ends leaving the image) have no condition, every vessel
written as one straight segment (its length the distance between its nodes,
in both programs), so the estimate's pressure term sums over the same nodes
with the same weights. FlowEstimate is given HaemoLynx's own targets (pressure
and wall shear stress, from the closed solve) and a final shear weight equal
to HaemoLynx's, and keeps its own continuation in that weight, its own way of
settling flow directions and its own bias factor kappa. It solves by
conjugate gradients: its LU path assembles its matrix wrongly where two
segments join the same two nodes. A network without open ends is given to it
with every condition known instead, a check that it then solves as NetFlow.

Usage::

    python scripts/secomb_comparison.py build --netflow-src <NetFlowV2> \\
        --flowestimate-src <FlowEstimateV1> --out <build dir>
    python scripts/secomb_comparison.py compare --bin <build dir> --out outputs/secomb_comparison \\
        [--graph NAME <solved graph .pkl> INLET_PA OUTLET_PA [--image-shape Z Y X]]

``compare`` always runs the hand-built network of
``examples/simple_network_haemodynamics.py`` as it is and with unequal
capillaries (its capillaries are all 5 um, so no bifurcation separates
anything); ``--graph`` adds a solved run, its inlets and outlets read back
from the solve (the dead ends of the solved vessels held at INLET_PA /
OUTLET_PA).
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

#: NetFlow's mmHg: its flow factor ``facfp`` takes 1333 dyn/cm^2.
NETFLOW_PA_PER_MMHG = 133.3
HAEMATOCRIT = 0.45
PLASMA_VISCOSITY_CP = 1.2
#: Vessels carrying less than this share of the largest flow are left out of
#: the relative flow differences (a relative difference of nothing).
NO_FLOW_SHARE = 1e-9
#: NetFlow's SOR stops once no node's pressure moves more than this (mmHg).
#: On E14.5 with fixed haematocrit (2026-10-15): at its sample's 1e-10,
#: vessels carrying 1e-8 of the largest flow -- pressure drops of ~3e-10
#: mmHg -- came out up to 29% away from HaemoLynx's; at 5e-12 up to 1.5%, at
#: 2e-12 0.58% (5 s). Rounding stops the updates shrinking below ~1.5e-12
#: there: asked for 1e-14, it ran its 5e7 sweeps (29 min) and agreed within
#: 1.5e-4. The sweep cap bounds a network whose floor is higher.
SOR_TOLERANCE_MMHG = 2e-12
SOR_MAX_SWEEPS = 2_000_000
#: ``max_segment_um`` that writes a straight vessel as one segment.
STRAIGHT = 1e9
MAX_HAEMATOCRIT_PASSES = 400
HAEMATOCRIT_TOLERANCE = 1e-7
#: Synthetic capillary diameters (um) for the unequal variant.
UNEQUAL_CAPILLARIES = {(1, 2): 6.0, (1, 3): 4.5, (2, 4): 5.5, (3, 5): 4.0, (4, 5): 5.0, (4, 6): 6.5, (5, 6): 4.5}

VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")

# --- build ---------------------------------------------------------------------------------

_DUMP_PRESSURES = (
    "\t{\t/* HaemoLynx comparison: every node's pressure */\n"
    "\t\tFILE *pfp = fopen(\"Current/NodePressures.out\", \"w\");\n"
    "\t\tfor (i = 1; i <= nnod; i++) fprintf(pfp, \"%i %.12e\\n\", nodname[i], nodpress[i]);\n"
    "\t\tfclose(pfp);\n"
    "\t}\n"
)
PATCHES = {
    "netflow": {
        "writeflow.cpp": [
            ('fprintf(ofp, "%i %i %i %i %f %f %f",', 'fprintf(ofp, "%i %i %i %i %f %.9e %.9e",'),
            ("\textern int nseg,*segname,*segtyp,*nodname;",
             "\textern int nseg,nnod,*segname,*segtyp,*nodname;\n\textern double *nodpress;"),
            ("\tfclose(ifp);\n\tfclose(ofp);\n}", "\tfclose(ifp);\n\tfclose(ofp);\n" + _DUMP_PRESSURES + "}"),
        ],
    },
    "flowestimate": {
        "writeflow.cpp": [
            ('"%i %i %i %i %7.3f %7.3f %7.4f\\n"', '"%i %i %i %i %7.3f %.9e %.9e\\n"'),
            ("\tfclose(ifp);\n\tfclose(ofp);\n}", "\tfclose(ifp);\n\tfclose(ofp);\n" + _DUMP_PRESSURES + "}"),
        ],
    },
}
DISHEM_SWITCH = ("int dishemversion = 1;", "int dishemversion = 0;")
#: A diagnostic FlowEstimate, not a reference: its bias factor kappa (added
#: in 2019, not in Fry et al. 2012 or HaemoLynx) held at 1, to show how much
#: of a difference from HaemoLynx's estimate is kappa's.
KAPPA_HELD_AT_ONE = ("kappa = kappasum2 / SQR(kappasum1);", "kappa = 1.;")


def _patch(directory: Path, name: str, replacements) -> None:
    path = directory / name
    raw = path.read_bytes().decode("latin-1")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: expected one '{old}', found {text.count(old)}; has the code changed?")
        text = text.replace(old, new)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("latin-1"))


def _compile(directory: Path, exe: str, vcvars: Optional[Path]) -> Path:
    sources = sorted(p.name for p in directory.glob("*.cpp"))
    if shutil.which("g++"):
        command = ["g++", "-O2", "-std=c++17", "-o", exe, *sources]
        if sys.platform.startswith("linux"):
            command.append("-lstdc++fs")
        subprocess.run(command, cwd=directory, check=True)
        return directory / exe
    if vcvars is None or not vcvars.exists():
        raise SystemExit("No g++ on PATH and no MSVC vcvars64.bat (--vcvars) to compile with.")
    script = directory / "compile.bat"
    script.write_text(f'@echo off\ncall "{vcvars}" >nul\ncd /d "{directory}"\n'
                      f"cl /nologo /O2 /EHsc /w /Fe:{exe}.exe {' '.join(sources)}\n")
    done = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True)
    if done.returncode != 0 or not (directory / f"{exe}.exe").exists():
        raise SystemExit(f"Compiling {directory} failed:\n{done.stdout}\n{done.stderr}")
    return directory / f"{exe}.exe"


def build(args) -> int:
    out = Path(args.out)
    targets = [
        ("netflow", Path(args.netflow_src), PATCHES["netflow"], None),
        ("netflow_dishem", Path(args.netflow_src), PATCHES["netflow"], DISHEM_SWITCH),
        ("flowestimate", Path(args.flowestimate_src), PATCHES["flowestimate"], None),
        ("flowestimate_kappa1", Path(args.flowestimate_src), PATCHES["flowestimate"], KAPPA_HELD_AT_ONE),
    ]
    built = {}
    for name, source, patches, switch in targets:
        directory = out / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for file in [*source.glob("*.cpp"), *source.glob("*.h")]:
            shutil.copy2(file, directory / file.name)
        for file, replacements in patches.items():
            _patch(directory, file, replacements)
        if switch:
            _patch(directory, "flow.cpp", [switch])
        built[name] = str(_compile(directory, name, Path(args.vcvars) if args.vcvars else VCVARS))
        print(f"built {built[name]}")
    commits = {}
    for label, source in (("netflow", args.netflow_src), ("flowestimate", args.flowestimate_src)):
        try:
            commits[label] = subprocess.run(
                ["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commits[label] = None
    (out / "build.json").write_text(json.dumps({"executables": built, "source_commits": commits}, indent=2))
    return 0


# --- networks ------------------------------------------------------------------------------


@dataclass
class Network:
    name: str
    graph: nx.MultiGraph
    inlets: list
    outlets: list
    inlet_pa: float
    outlet_pa: float
    #: Dead ends leaving the image (FlowEstimate's unknown boundary nodes).
    open_ends: list
    #: Explicit Fry targets (Pa) when the closed solve gives none.
    targets: Optional[tuple] = None
    source: str = ""


def _example_module():
    spec = importlib.util.spec_from_file_location("simple_network_example", REPO / "examples" / "simple_network_haemodynamics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_a_third_along(G, u, v, new) -> None:
    """Vessel u-v as two vessels of its own diameter meeting at a degree-2
    node *new* a third of the way along: the same resistance and the same
    haematocrit, one node more. (At the midpoint, with the vessels written in
    10 um segments, its two halves are 10 segments each, as many as its mirror
    image 2-4 is in one: still symmetric to ``flowtest``.)"""
    data = G[u][v][0]
    a, b = np.asarray(G.nodes[u]["pos"], dtype=float), np.asarray(G.nodes[v]["pos"], dtype=float)
    middle = a + (b - a) / 3.0
    G.remove_edge(u, v, 0)
    G.add_node(new, pos=middle)
    for start, end in ((a, middle), (middle, b)):
        G.add_edge(u if start is a else new, new if start is a else v, **{
            **data, "length": float(np.linalg.norm(end - start)), "voxels": [start.tolist(), end.tolist()]})


def synthetic_networks() -> list[Network]:
    """The example's network as it is (all capillaries 5 um), and with the
    capillaries' diameters made unequal so the phase separation divides
    something. The example is mirror-symmetric: NetFlow's ``flowtest`` judges a
    vessel flowless from a solve with every conductance 1 and so removes 4-5,
    which carries flow once the diameters differ; the unequal variant splits
    3-5 a third of the way along, which changes no resistance and no haematocrit but
    breaks that symmetry."""
    from haemolynx import haemodynamics
    from haemolynx.parsers import load_config

    example = _example_module()
    settings = load_config(example.CONFIG_PATH, example.SCHEMA)
    networks = []
    for name, diameters in (("synthetic", {}), ("synthetic_unequal", UNEQUAL_CAPILLARIES)):
        G = example.build_example_network()
        if diameters:
            _split_a_third_along(G, 3, 5, 8)
        haemodynamics.apply_poiseuille_haemodynamics(G, diameter_by_branch_order=settings["diameter_by_branch_order"])
        for (u, v), d in diameters.items():
            for a, b in ((u, v), (u, 8), (8, v)) if (u, v) == (3, 5) else ((u, v),):
                if G.has_edge(a, b):
                    G[a][b][0]["diameter_um"] = d
        networks.append(Network(
            name, G, [0], [7], float(settings["inlet_pressure_pa"]), float(settings["outlet_pressure_pa"]), [],
            source="examples/simple_network_haemodynamics.py"
            + (" with unequal capillaries, 3-5 split a third of the way along" if diameters else ""),
        ))
    return networks


def graph_network(name: str, path: str, inlet_pa: float, outlet_pa: float, image_shape=None) -> Network:
    """A solved run's graph, its inlets and outlets read back from the solve."""
    from haemolynx import graph
    from haemolynx.graph import FLOW_SOLVED

    G = pickle.load(open(path, "rb"))
    if isinstance(G, dict):
        G = G["graph"]
    solved = G.edge_subgraph([e for e in G.edges(keys=True) if G.edges[e].get(FLOW_SOLVED) is True])
    ends = [n for n in solved if solved.degree(n) == 1]
    inlets = [n for n in ends if np.isclose(G.nodes[n].get("pressure", np.nan), inlet_pa)]
    outlets = [n for n in ends if np.isclose(G.nodes[n].get("pressure", np.nan), outlet_pa)]
    if image_shape and "image_extent_um" not in G.graph:
        voxel = G.graph.get("image_voxel_size_zyx", (1.0, 1.0, 1.0))
        graph.dead_ends.record_image_extent(G, image_shape, voxel)
    open_ends = [n for n in graph.dead_ends.open_end_nodes(G) if n not in inlets and n not in outlets]
    return Network(name, G, inlets, outlets, inlet_pa, outlet_pa, open_ends, source=str(path))


def prepared(network: Network, *, keep_open_ends: bool) -> Network:
    """The network both programs solve: only vessels with a conductance, only
    those on a path between two boundary nodes, bridges as vessels, and the
    physics reset to the matched settings."""
    from haemolynx import graph
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
    from haemolynx.haemodynamics.poiseuille import EDGE_DIAMETER_BASIS

    G = network.graph
    G = G.edge_subgraph([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                         if u != v and d.get("conductance")]).copy()
    ends = network.open_ends if keep_open_ends else []
    sources, sinks = [*network.inlets, *ends], [*network.outlets, *ends]
    if not network.outlets:  # one designated side: every boundary node is both
        sources = sinks = [*network.inlets, *ends]
    kept = graph.inlet_to_outlet_vessels(G, sources, sinks)
    G = G.edge_subgraph(kept).copy()
    for _u, _v, data in G.edges(data=True):
        data[IS_ZERO_RESISTANCE] = False
        data["branch_order"] = data.get("branch_order") or "bridge"
        data.pop(EDGE_DIAMETER_BASIS, None)
        data.pop("discharge_haematocrit", None)
    G.graph.pop("open_end_conditions", None)
    keep = set(G)
    return Network(network.name, G, [n for n in network.inlets if n in keep], [n for n in network.outlets if n in keep],
                   network.inlet_pa, network.outlet_pa, [n for n in ends if n in keep], network.targets, network.source)


# --- HaemoLynx side ------------------------------------------------------------------------


def _config():
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig

    # Every vessel carries its own diameter_um; the table (which the config
    # requires) is never read.
    return HaemodynamicsApplyConfig(diameters={
        "diameter_by_branch_order": {"unused": 10.0}, "custom_edges": [], "viscosity_law": "pries",
        "pries_viscosity_form": "in_vivo_1994", "diameter_basis": "endothelial", "haematocrit": HAEMATOCRIT,
    })


def solve_haemolynx(network: Network, *, distributed: bool, junction_rule: str) -> dict:
    """Matched-physics resistances, then the flow (and haematocrit) solve."""
    from haemolynx import haemodynamics as hd
    from haemolynx.haemodynamics.apply import apply_poiseuille_resistances
    from haemolynx.haemodynamics.haematocrit_distribution import iterate_flow_and_haematocrit

    G = network.graph
    config = _config()
    for _u, _v, data in G.edges(data=True):
        data.pop("discharge_haematocrit", None)
    apply_poiseuille_resistances(G, config)
    if distributed:
        result = iterate_flow_and_haematocrit(
            G, recompute_resistances=lambda: apply_poiseuille_resistances(G, config),
            inlet_haematocrit=HAEMATOCRIT, inlet_p_bc=network.inlet_pa, outlet_p_bc=network.outlet_pa,
            inlet_nodes=network.inlets, outlet_nodes=network.outlets, max_iterations=MAX_HAEMATOCRIT_PASSES,
            tolerance=HAEMATOCRIT_TOLERANCE, junction_rule=junction_rule, solver="sparse",
        )
        return {"converged": bool(result["converged"]), "iterations": int(result["iterations"]),
                "max_delta": float(result["max_delta"]), "still_changing": _still_changing(G, junction_rule)}
    conductance, nodes = hd.build_conductance_matrix_from_graph(G, solver="sparse")
    flow = hd.solve_flow_from_conductance_matrix(
        conductance, nodes, inlet_p_bc=network.inlet_pa, outlet_p_bc=network.outlet_pa,
        inlet_nodes=network.inlets, outlet_nodes=network.outlets,
    )
    hd.set_edge_flows(G, nodes, flow["pressure"])
    for _u, _v, data in G.edges(data=True):
        data["discharge_haematocrit"] = HAEMATOCRIT
    return {"converged": True, "iterations": 1, "max_delta": 0.0, "still_changing": ""}


def _still_changing(G, junction_rule: str) -> str:
    """The vessel whose haematocrit one more distribution over the final
    flows would move most, and by how much: where an unconverged loop stopped."""
    from haemolynx.haemodynamics.haematocrit_distribution import distribute_discharge_haematocrit

    H = copy.deepcopy(G)
    distribute_discharge_haematocrit(H, inlet_haematocrit=HAEMATOCRIT, junction_rule=junction_rule)
    change = {(u, v, k): abs(H.edges[u, v, k]["discharge_haematocrit"] - d["discharge_haematocrit"])
              for u, v, k, d in G.edges(keys=True, data=True)}
    (u, v, k), amount = max(change.items(), key=lambda item: item[1])
    return f"{u}-{v}-{k} ({amount:.2g})"


# --- running Secomb's codes ----------------------------------------------------------------


def _rheol_params(*, phase_separation: bool, qtol: float, flowestimate: bool) -> str:
    lines = [
        "0.964 6.98 -13.29\t\tbifpar (bifurcation law)",
        "0.80 -0.075 -11.0 12.0\t\tcpar",
        "6.0 -0.085 3.2 -2.44 -0.06 0.645\tviscpar",
    ]
    if flowestimate:
        lines.append("1.70 -0.35 -0.6 -0.01\t\tfahrpr")
    lines += [
        f"{SOR_MAX_SWEEPS} {SOR_TOLERANCE_MMHG:.0e} 1.95\t\tnitmax, tol, omega - linear solver",
        f"{MAX_HAEMATOCRIT_PASSES} {qtol:.3e} {HAEMATOCRIT_TOLERANCE:.1e}\t\tnitmax1, qtol, hdtol - nonlinear solver",
        "1.1 0.5\t\toptw, optlam",
        f"3. {PLASMA_VISCOSITY_CP} 92.\t\tconstvisc, vplas, mcv",
        f"{HAEMATOCRIT}\t\tconsthd",
        "1\t\tvaryviscosity",
        f"{int(phase_separation)}\t\tphaseseparation",
    ]
    return "\n".join(lines) + "\n"


def _contour_params(G) -> str:
    pos = np.asarray([G.nodes[n]["pos"] for n in G], dtype=float)
    z, y, x = (max(float(np.max(np.abs(pos[:, i]))), 1.0) for i in range(3))
    return f"0. 0. {z / 2:.1f} 5\tplane origin\n{x:.1f} 0. {z / 2:.1f} 51\tcorner\n0. {y:.1f} {z / 2:.1f} 51\tcorner\n"


def _run(exe: Path, run_dir: Path, timeout: int) -> str:
    completed = subprocess.run([str(exe)], cwd=run_dir, capture_output=True, text=True, timeout=timeout)
    (run_dir / "stdout.txt").write_text(completed.stdout + completed.stderr)
    if not (run_dir / "Current" / "NodePressures.out").exists():
        raise RuntimeError(f"{exe.name} wrote no results in {run_dir} (exit {completed.returncode}); see stdout.txt")
    return completed.stdout


def _node_pressures(run_dir: Path) -> dict[int, float]:
    out = {}
    for line in (run_dir / "Current" / "NodePressures.out").read_text().splitlines():
        name, value = line.split()
        out[int(name)] = float(value)
    return out


def _warnings(stdout: str) -> list[str]:
    return [line.strip() for line in stdout.splitlines() if "Warning" in line or "Error" in line]


# --- comparing -----------------------------------------------------------------------------


def _relative_flow_differences(G, export, theirs: dict) -> dict:
    """Per vessel, HaemoLynx's flow u->v against the first segment's of its
    chain (m^3/s), and the haematocrits."""
    from haemolynx.visualization.secomb_io import NL_PER_MIN_PER_M3_S

    rows = []
    for (u, v, key), names in export.edge_segments.items():
        data = G.edges[u, v, key]
        ours = float(data["conductance"]) * (float(G.nodes[u]["pressure"]) - float(G.nodes[v]["pressure"]))
        # FlowEstimate's flowtest drops segments a unit-conductance solve
        # leaves without flow (NetFlow's keeps them as type 0).
        segment = theirs.segments.get(names[0], (-1, 0, 0, 0.0, 0.0, float("nan")))
        along = [theirs.segments[n][4] if n in theirs.segments else 0.0 for n in names]
        rows.append({
            "edge": f"{u}-{v}-{key}", "u": u, "v": v, "diameter_um": float(data["diameter_um"]),
            "flow_haemolynx": ours, "flow_secomb": segment[4] / NL_PER_MIN_PER_M3_S,
            "chain_spread": (max(along) - min(along)) / NL_PER_MIN_PER_M3_S,
            "secomb_type": segment[0],
            "hd_haemolynx": float(data.get("discharge_haematocrit", HAEMATOCRIT)), "hd_secomb": segment[5],
        })
    return rows


def summarise(rows: list[dict], pressures: list[tuple], drop_pa: float, inflow: tuple) -> dict:
    flows = np.asarray([r["flow_haemolynx"] for r in rows])
    theirs = np.asarray([r["flow_secomb"] for r in rows])
    largest = float(np.max(np.abs(flows))) if len(flows) else 0.0
    flowing = np.abs(flows) > NO_FLOW_SHARE * largest
    rel = np.abs(theirs[flowing] - flows[flowing]) / np.abs(flows[flowing])
    hd_diff = np.abs(np.asarray([r["hd_secomb"] - r["hd_haemolynx"] for r in rows]))[flowing]
    p_diff = np.asarray([abs(a - b) for _n, a, b in pressures]) / drop_pa if pressures else np.zeros(1)
    worst = int(np.argmax(rel)) if len(rel) else None
    worst_row = [r for r, f in zip(rows, flowing) if f][worst] if worst is not None else {}
    return {
        "vessels": len(rows),
        "vessels_with_flow": int(flowing.sum()),
        "no_flow_vessels": int((~flowing).sum()),
        "no_flow_secomb_max_share": (float(np.max(np.abs(theirs[~flowing]))) / largest if (~flowing).any() and largest else 0.0),
        "flow_rel_diff_max": float(rel.max()) if len(rel) else None,
        "flow_rel_diff_median": float(np.median(rel)) if len(rel) else None,
        "flow_rel_diff_p99": float(np.percentile(rel, 99)) if len(rel) else None,
        "vessels_over_1pct": int(np.sum(rel > 0.01)),
        "worst_vessel": worst_row.get("edge"),
        "worst_vessel_flow_share": (abs(worst_row["flow_haemolynx"]) / largest) if worst_row else None,
        "hd_abs_diff_max": float(hd_diff.max()) if len(hd_diff) else None,
        "hd_abs_diff_median": float(np.median(hd_diff)) if len(hd_diff) else None,
        "pressure_diff_max_share_of_drop": float(p_diff.max()),
        "pressure_diff_median_share_of_drop": float(np.median(p_diff)),
        "inflow_haemolynx": inflow[0],
        "inflow_secomb": inflow[1],
        "inflow_rel_diff": abs(inflow[1] - inflow[0]) / abs(inflow[0]) if inflow[0] else None,
    }


def _export(network: Network, path: Path, *, order_by_flow: bool, unknown=(), max_segment_um=10.0):
    from haemolynx.visualization.secomb_io import write_secomb_network

    export = write_secomb_network(
        network.graph, path, inlet_nodes=network.inlets, outlet_nodes=network.outlets,
        inlet_pressure_pa=network.inlet_pa, outlet_pressure_pa=network.outlet_pa,
        boundary_haematocrit=HAEMATOCRIT, unknown_boundary_nodes=unknown, pa_per_mmhg=NETFLOW_PA_PER_MMHG,
        order_by_flow=order_by_flow, max_segment_um=max_segment_um, title=f"HaemoLynx {network.name}",
    )
    path.with_name(path.stem + "_segments.json").write_text(json.dumps(
        {f"{u}|{v}|{k}": names for (u, v, k), names in export.edge_segments.items()}))
    return export


def _match_lengths(network: Network, scratch: Path, unknown=(), max_segment_um=10.0) -> dict:
    """Give every vessel the length its chain will have in the file, so both
    programs solve one network; returns how far apart they were."""
    export = _export(network, scratch / "lengths.dat", order_by_flow=False, unknown=unknown,
                     max_segment_um=max_segment_um)
    moved = []
    for edge, length in export.edge_lengths_um.items():
        data = network.graph.edges[edge]
        if data.get("length"):
            moved.append(abs(length - float(data["length"])) / float(data["length"]))
        data["length"] = length
    return {"length_rel_change_max": float(max(moved, default=0.0)), "shorter_than_chord": export.shorter_than_chord,
            "bridges_as_vessels": export.bridges_as_vessels}


def compare_netflow(network: Network, exes: dict, out: Path, timeout: int) -> list[dict]:
    from haemolynx.visualization.secomb_io import read_secomb_network

    base = prepared(network, keep_open_ends=False)
    cases = [("fixed", "netflow", None), ("distributed", "netflow", "sequential_bifurcations"),
             ("distributed", "netflow_dishem", "no_separation")]
    results = []
    for haematocrit, program, rule in cases:
        label = f"{network.name}__{haematocrit}" + (f"__{rule}" if rule else "")
        run_dir = out / "runs" / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        net = copy.deepcopy(base)
        lengths = _match_lengths(net, run_dir)
        ours = solve_haemolynx(net, distributed=haematocrit == "distributed", junction_rule=rule or "no_separation")
        export = _export(net, run_dir / "Network.dat", order_by_flow=True)
        G = net.graph
        largest_nl = max(abs(float(d["flow_signed"])) for *_e, d in G.edges(data=True)) * 6e13
        (run_dir / "RheolParams.dat").write_text(_rheol_params(
            phase_separation=haematocrit == "distributed", qtol=1e-7 * largest_nl, flowestimate=False))
        (run_dir / "ContourParams.dat").write_text(_contour_params(G))
        stdout = _run(Path(exes[program]), run_dir, timeout)
        theirs = read_secomb_network(run_dir / "Current" / "NetworkNew.dat")
        pressure_mmhg = _node_pressures(run_dir)
        rows = _relative_flow_differences(G, export, theirs)
        pressures = [(n, float(G.nodes[n]["pressure"]), pressure_mmhg[name] * NETFLOW_PA_PER_MMHG)
                     for n, name in export.node_names.items()]
        summary = summarise(rows, pressures, net.inlet_pa - net.outlet_pa, _inflow(rows, net.inlets))
        iterations = re.findall(r"Flow: (\d+) iterations", stdout)
        unsettled = re.findall(r"h'crit error = ([\d.eE+-]+) at segment (\d+)", stdout)
        vessel_of = {name: f"{u}-{v}-{k}" for (u, v, k), names in export.edge_segments.items() for name in names}
        results.append({
            "network": network.name, "program": program, "haematocrit": haematocrit, "junction_rule": rule or "",
            "nodes": G.number_of_nodes(), "secomb_segments": len(theirs.segments),
            "junctions_4plus": sum(1 for n in G if G.degree(n) >= 4),
            **lengths,
            "haemolynx_converged": ours["converged"], "haemolynx_passes": ours["iterations"],
            "haemolynx_still_changing": ours["still_changing"],
            "secomb_passes": int(iterations[-1]) if iterations else None,
            "secomb_still_changing": (f"{vessel_of.get(int(unsettled[-1][1]), '?')} ({float(unsettled[-1][0]):.2g})"
                                      if unsettled else ""),
            "secomb_warnings": " | ".join(_warnings(stdout))[:400],
            **summary,
        })
        with open(run_dir / "vessels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{label}: max rel flow diff {summary['flow_rel_diff_max']:.3g}, "
              f"median {summary['flow_rel_diff_median']:.3g}, max hd diff {summary['hd_abs_diff_max']:.3g}")
    return results


def _inflow(rows: list[dict], inlets) -> tuple[float, float]:
    """Total flow into the network through *inlets* (m^3/s), HaemoLynx's and
    Secomb's, from the per-vessel rows (flows positive u -> v)."""
    inlets = set(inlets)
    ours = theirs = 0.0
    for row in rows:
        sign = (row["u"] in inlets) - (row["v"] in inlets)
        ours += sign * row["flow_haemolynx"]
        theirs += sign * row["flow_secomb"]
    return ours, theirs


# --- FlowEstimate --------------------------------------------------------------------------


def _flowestimate_params(*, pressure_target_mmhg: float, shear_target_dyn: float, final_ktau: float, steps: int) -> str:
    start = final_ktau / 2 ** (steps - 1)
    # Conjugate gradients, as FlowEstimate's own sample. Its LU path
    # (solvetype 1) assembles the matrix with '=' where two segments join the
    # same two nodes (Amatrix.cpp), so one overwrites the other: on the Nerve
    # network, whose straight export has one such pair, it broke flow
    # conservation, while CG and SOR agreed with each other and HaemoLynx.
    return "\n".join([
        "FlowEstParams",
        "2\t\tsolvetype (1 = LU, 2 = CG, 3 = SOR)",
        f"{pressure_target_mmhg:.9g}\ttarget pressure (mmHg)",
        "0\t\tvarytargetshear (0 or 1)",
        f"{shear_target_dyn:.9g}\ttarget shear stress for fixed target (dyn/cm2)",
        f"{start:.9g}\tktaustart, starting value of ktau",
        f"{steps}\t\tnumber of ktau steps, doubling each time",
        "5\t\tmax number of runs for each tau value",
        "1.e-10\teps, tolerance for iterative solvers",
        "2000000\tnitmax2",
        "25\t\tnitmax3",
        "1.95 1.0\tomega1, omega2",
        "8.0\t\tdiamcrit",
        "10.\t\tknown_flow_weight (um)",
        "11111\tseed",
    ]) + "\n"


def compare_flowestimate(network: Network, exes: dict, out: Path, timeout: int) -> list[dict]:
    """HaemoLynx's ``estimated`` open ends against FlowEstimate, on the
    network with its open ends left without a condition. A network without
    open ends is given to FlowEstimate with every condition known, which
    leaves its estimate nothing to choose: a check that it then solves the
    network as NetFlow and HaemoLynx do."""
    from haemolynx import haemodynamics as hd
    from haemolynx.haemodynamics.apply import apply_poiseuille_resistances
    from haemolynx.visualization.secomb_io import read_secomb_network

    net = prepared(network, keep_open_ends=True)
    if not net.outlets:
        return []
    case = "estimated" if net.open_ends else "complete_conditions"
    G = net.graph
    # Every vessel straight from node to node, one segment each, so the
    # pressure term sums over the same nodes with the same weights (half the
    # length of each vessel at a node) in both: a chain's inner nodes would
    # each count in FlowEstimate's and in none of HaemoLynx's.
    for u, v, data in G.edges(data=True):
        chord = math.dist(np.round(np.asarray(G.nodes[u]["pos"], float), 6), np.round(np.asarray(G.nodes[v]["pos"], float), 6))
        if chord >= 1e-3:
            data["length"] = chord
    scratch = out / "runs" / f"{network.name}__{case}"
    scratch.mkdir(parents=True, exist_ok=True)
    lengths = _match_lengths(net, scratch, unknown=net.open_ends, max_segment_um=STRAIGHT)
    if net.open_ends:
        apply_poiseuille_resistances(G, _config())
        conductance, nodes = hd.build_conductance_matrix_from_graph(G, solver="sparse")
        conditions = hd.open_ends.estimate_open_end_conditions(
            G, conductance, nodes, inlet_nodes=net.inlets, outlet_nodes=net.outlets, inlet_p_bc=net.inlet_pa,
            outlet_p_bc=net.outlet_pa, ends=net.open_ends,
        )
        matrix, names, index, extra_in, extra_out = hd.open_ends.augment(conductance, nodes, conditions)
        flow = hd.solve_flow_from_conductance_matrix(
            matrix, names, inlet_p_bc=net.inlet_pa, outlet_p_bc=net.outlet_pa,
            inlet_nodes=[*net.inlets, *extra_in], outlet_nodes=[*net.outlets, *extra_out], node_to_idx=index,
        )
        hd.set_edge_flows(G, nodes, flow["pressure"][: len(nodes)])
        info = conditions.info
    else:
        solve_haemolynx(net, distributed=False, junction_rule="no_separation")
        targets = _closed_targets(G)
        info = {"pressure_target_pa": targets["pressure"], "shear_target_pa": targets["shear"]}
    # HaemoLynx's objective, wp * sum a (p - p0)^2 + wt * sum L (tau - s tau0)^2,
    # in FlowEstimate's units (mmHg, dyn/cm^2, kpress = 1).
    p0, tau0 = info["pressure_target_pa"], info["shear_target_pa"]
    weights = _fry_weights(G, net, tau0)
    final_ktau = weights["wt"] * 0.01 / (weights["wp"] * NETFLOW_PA_PER_MMHG ** 2)
    drop = abs(net.inlet_pa - net.outlet_pa)
    results = []
    # With every condition known there is nothing for kappa to move.
    for program in ("flowestimate", "flowestimate_kappa1") if net.open_ends else ("flowestimate",):
        label = f"{network.name}__{case}__{program}"
        run_dir = out / "runs" / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        export = _export(net, run_dir / "Network.dat", order_by_flow=False, unknown=net.open_ends,
                         max_segment_um=STRAIGHT)
        (run_dir / "RheolParams.dat").write_text(_rheol_params(phase_separation=False, qtol=1e-3, flowestimate=True))
        (run_dir / "ContourParams.dat").write_text(_contour_params(G))
        (run_dir / "FlowEstParams.dat").write_text(_flowestimate_params(
            pressure_target_mmhg=p0 / NETFLOW_PA_PER_MMHG, shear_target_dyn=tau0 * 10.0, final_ktau=final_ktau,
            steps=9))
        stdout = _run(Path(exes[program]), run_dir, timeout)
        theirs = read_secomb_network(run_dir / "Current" / "NetworkNew.dat")
        pressure_mmhg = _node_pressures(run_dir)
        rows = _relative_flow_differences(G, export, theirs)
        pressures = [(n, float(G.nodes[n]["pressure"]), pressure_mmhg[name] * NETFLOW_PA_PER_MMHG)
                     for n, name in export.node_names.items() if name in pressure_mmhg]
        open_diff = [abs(float(G.nodes[n]["pressure"]) - pressure_mmhg[export.node_names[n]] * NETFLOW_PA_PER_MMHG) / drop
                     for n in net.open_ends if export.node_names[n] in pressure_mmhg]
        summary = summarise(rows, pressures, drop, _inflow(rows, net.inlets))
        flows = np.asarray([r["flow_haemolynx"] for r in rows])
        their_flows = np.asarray([r["flow_secomb"] for r in rows])
        big = np.abs(flows) > 1e-3 * np.max(np.abs(flows))
        kappa = re.findall(r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$",
                           (run_dir / "Current" / "Run_summary.out").read_text(), flags=re.M)
        result = {
            "network": network.name, "program": program, "haematocrit": "fixed", "junction_rule": "",
            "case": case, "nodes": G.number_of_nodes(), "secomb_segments": len(theirs.segments),
            "open_ends": len(net.open_ends), **lengths,
            "pressure_target_pa": p0, "shear_target_pa": tau0, "final_ktau_over_kpress": final_ktau,
            "haemolynx_estimate": json.dumps({k: v for k, v in info.items() if not isinstance(v, list)}),
            "secomb_final_kappa": float(kappa[-1][4]) if kappa else None,
            "direction_agreement_over_0.1pct_of_max": float(np.mean(np.sign(flows[big]) == np.sign(their_flows[big]))),
            "open_end_pressure_diff_max_share_of_drop": max(open_diff, default=None),
            "open_end_pressure_diff_median_share_of_drop": float(np.median(open_diff)) if open_diff else None,
            "secomb_warnings": " | ".join(_warnings(stdout))[:400],
            **summary,
        }
        results.append(result)
        with open(run_dir / "vessels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{label}: median rel flow diff {summary['flow_rel_diff_median']:.3g}, "
              f"directions agree {result['direction_agreement_over_0.1pct_of_max']:.3f}")
    return results


def _closed_targets(G) -> dict:
    """The length-weighted mean node pressure and length-weighted median
    |wall shear stress| (Pa) over the vessels that carry flow, as
    ``estimate_open_end_conditions`` takes them from its closed solve."""
    pressure_sum = weight_sum = 0.0
    shear, lengths = [], []
    for u, v, data in G.edges(data=True):
        tau = float(data["diameter_um"]) * (float(G.nodes[u]["pressure"]) - float(G.nodes[v]["pressure"])) / (4 * float(data["length"]))
        if abs(tau) <= 0:
            continue
        shear.append(abs(tau)), lengths.append(float(data["length"]))
        for n in (u, v):
            pressure_sum += float(G.nodes[n]["pressure"]) * float(data["length"]) / 2
            weight_sum += float(data["length"]) / 2
    order = np.argsort(shear)
    cum = np.cumsum(np.asarray(lengths)[order])
    return {"pressure": pressure_sum / weight_sum,
            "shear": float(np.asarray(shear)[order][int(np.searchsorted(cum, cum[-1] / 2))])}


def _fry_weights(G, net: Network, tau0: float) -> dict:
    """``wp`` and ``wt`` as ``estimate_open_end_conditions`` sets them: each
    term over its own scale, the node weights summed over the free nodes."""
    designated = set(net.inlets) | set(net.outlets)
    node_weight = {}
    total_length = 0.0
    for u, v, data in G.edges(data=True):
        length = float(data["length"])
        total_length += length
        for n in (u, v):
            node_weight[n] = node_weight.get(n, 0.0) + length / 2
    free = sum(w for n, w in node_weight.items() if n not in designated)
    drop = abs(net.inlet_pa - net.outlet_pa)
    return {"wp": 1.0 / (free * drop ** 2), "wt": 1.0 / (total_length * tau0 ** 2)}


# --- report --------------------------------------------------------------------------------


def report(results: list[dict], out: Path, header: str) -> None:
    keys = list(dict.fromkeys(k for row in results for k in row))
    with open(out / "secomb_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    shown = ["network", "program", "haematocrit", "junction_rule", "vessels_with_flow", "flow_rel_diff_max",
             "flow_rel_diff_median", "vessels_over_1pct", "hd_abs_diff_max", "hd_abs_diff_median",
             "pressure_diff_max_share_of_drop", "inflow_rel_diff", "haemolynx_passes", "secomb_passes"]

    def cell(value):
        if value is None or value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    lines = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    lines += ["| " + " | ".join(cell(r.get(k)) for k in shown) + " |" for r in results]
    (out / "REPORT.md").write_text(f"# HaemoLynx against Secomb's NetFlow and FlowEstimate\n\n{header}\n\n"
                                   + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def compare(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built = json.loads((Path(args.bin) / "build.json").read_text())
    exes = built["executables"]
    networks = synthetic_networks()
    for name, path, inlet_pa, outlet_pa in args.graph or []:
        networks.append(graph_network(name, path, float(inlet_pa), float(outlet_pa), args.image_shape))
    results = []
    for network in networks:
        if network.outlets:
            results += compare_netflow(network, exes, out, args.timeout)
        results += compare_flowestimate(network, exes, out, args.timeout)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
    header = (f"HaemoLynx at `{commit}`{' (src modified)' if dirty else ''}; NetFlowV2 at "
              f"`{(built['source_commits'].get('netflow') or '?')[:7]}`, FlowEstimateV1 at "
              f"`{(built['source_commits'].get('flowestimate') or '?')[:7]}`.\n\nNetworks: "
              + "; ".join(f"{n.name} ({n.source})" for n in networks))
    report(results, out, header)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="copy, patch (output only) and compile NetFlow and FlowEstimate")
    b.add_argument("--netflow-src", required=True)
    b.add_argument("--flowestimate-src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--vcvars", help="MSVC vcvars64.bat, when there is no g++")
    c = commands.add_parser("compare", help="solve each network with HaemoLynx and with Secomb's codes")
    c.add_argument("--bin", required=True, help="the build directory")
    c.add_argument("--out", required=True)
    c.add_argument("--graph", nargs=4, action="append", metavar=("NAME", "PKL", "INLET_PA", "OUTLET_PA"))
    c.add_argument("--image-shape", type=int, nargs=3, help="(z, y, x) voxels, for a graph that records no extent")
    c.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
