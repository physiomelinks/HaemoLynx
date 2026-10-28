#!/usr/bin/env python3
"""The cut-out harness of ``PLAN_synthetic_surround.md``: how close each
``open_end_model`` comes to the flow beyond an image's faces, on networks
whose beyond is known.

A synthetic network drawn to a template at ``--scale`` times its box along
``--axes``, closed, and solved, is the truth. A box of the template's own size
cut from it (``graph.crop_graph_to_box``) is a stand-in image: the box nearest
the centre whose faces cut enough arterial and venous vessels to feed it
(:func:`candidate_boxes`; the bed's centre alone holds none). Its widest arterial
and venous open ends, as many as the template has inlets and outlets, are its
inlets and outlets, as a person picks them from an image;
the pieces of it joined to neither are left out, as a run leaves them
unsolved. The crop is solved under each open-end model
(``open_end_sensitivity.MODELS``) and every vessel's flow scored against the
truth's:

* ``direction_agreement``: the share of vessels flowing the truth's way;
* ``flow_error_median`` / ``_p90``: ``|q - q_true| / |q_true|``;
* ``share_error_median``: the same on each flow as a share of the inflow
  through the designated inlets (the truth's through the same vessels), so a
  network fed at the wrong pressure is not marked down for that alone;
* ``flow_rank_correlation``: Spearman's, of the flows' sizes;

for the whole crop and beyond ``--margin-um`` of its faces, over vessels whose
true flow is above the statistics' carrying floor. At its open ends:
``open_end_direction_agreement`` (blood entering or leaving as in the truth;
an end the model gives no flow is wrong), apart for arterial and venous ends
(``_trees``, what decides a surround's trunk or tree) and capillaries; the
share of arterial ends feeding the image and of venous ends draining it in the
truth (``*_ends_upstream_in_truth``: where the surround's trunk is right); the
share of the inflow through the open ends, the model's and the truth's; and
``designated_inflow_ratio``, the designated inlets' inflow over the truth's.

Beside the models, the **surround** (phase 4): the crop inside a network
drawn round it to the template (``synthetic.surround_network``, at the
truth's ``--scale`` along ``--axes`` unless ``--surround-scale``, its own
seed), its outer inlets and outlets at the truth's pressures, any end it
leaves open under ``estimated``; the image's pressures are read back and
scored as a model's. It is drawn to the very template the truth was, where
an image's run would measure its own: a kind start.

The report ends with the plan's kill criterion (:func:`kill_criterion`: the
surround must beat ``estimated`` on median flow error and on direction
agreement, medians over the runs, and how many runs it beats it in), and
D5's line: over every case, at how many arteriole and venule ends the
surround's trunk is right, and "revisit D5" at half or fewer (Harvey,
2026-10-27: every such end runs on as a trunk, "revisit if the harness says
otherwise").

Usage::

    python scripts/surround_check.py --template fixture --seeds 1 2 3 4 5 \\
        --beds delaunay voronoi --out outputs/surround_check/fixture
    python scripts/surround_check.py --template <run>.haemorun [--source graph] ...
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import networkx as nx
import numpy as np

from open_end_sensitivity import MODELS, _solve

MARGIN_UM = 20.0
#: Where the image's box may sit in the larger network: every BOX_STEPS-th of
#: its extent, BOX_GAP of it off the outer faces.
BOX_STEPS = 8
BOX_GAP = 0.1
#: How many of the best boxes are cropped looking for one a run could solve.
BOX_TRIES = 20
#: The oracle's row, and how much stiffer than the network's stiffest vessel
#: its hold on each end is.
ORACLE = "oracle (true pressures at the ends)"
ORACLE_STIFFNESS = 1e8
#: The surround's row, and how far its shell's seed lies from the truth's,
#: so the two are not drawn from one stream.
SURROUND = "surround"
SURROUND_SEED_OFFSET = 1000
#: The templates a surround is drawn to, and each one's row: the template
#: the truth was drawn to (kind: the tissue it draws is the truth's kind),
#: or the one a run of the stand-in image would measure (:func:`image_template`).
TRUTH_TEMPLATE = "truth"
IMAGE_TEMPLATE = "image"
SURROUND_TEMPLATES = {TRUTH_TEMPLATE: SURROUND, IMAGE_TEMPLATE: "surround, image's own template"}
#: What the kill criterion compares the surround with, and on what: each
#: score's name and whether higher is better.
BASELINE = "estimated"
KILL_SCORES = (("flow_error_median", "median flow error", False), ("direction_agreement", "direction agreement", True))
SCORED = (
    "direction_agreement",
    "flow_error_median",
    "flow_error_p90",
    "share_error_median",
    "flow_rank_correlation",
)


# --- the scores (pure) ---------------------------------------------------------------------


def flow_scores(q_true: Sequence[float], q_model: Sequence[float], floor: float) -> dict[str, Optional[float]]:
    """:data:`SCORED` (but the share error) of *q_model* against *q_true*,
    signed flows along the same orientation, over the vessels whose true flow
    is above *floor*."""
    from scipy.stats import spearmanr

    t, m = np.asarray(q_true, dtype=float), np.asarray(q_model, dtype=float)
    keep = np.abs(t) > floor
    t, m = t[keep], m[keep]
    if not len(t):
        return {"vessels": 0, "direction_agreement": None, "flow_error_median": None, "flow_error_p90": None,
                "flow_rank_correlation": None}
    error = np.abs(m - t) / np.abs(t)
    rank = spearmanr(np.abs(t), np.abs(m))[0] if len(t) > 2 and np.ptp(np.abs(m)) > 0 else None
    return {
        "vessels": int(len(t)),
        "direction_agreement": float(np.mean(np.sign(m) == np.sign(t))),
        "flow_error_median": float(np.median(error)),
        "flow_error_p90": float(np.percentile(error, 90)),
        "flow_rank_correlation": None if rank is None or not np.isfinite(rank) else float(rank),
    }


def share_error_median(
    q_true: Sequence[float], q_model: Sequence[float], floor: float, inflow_true: float, inflow_model: float
) -> Optional[float]:
    """The median ``|q/Q - q_true/Q_true|`` over ``|q_true/Q_true|``, each
    flow a share of its own solve's designated inflow, over the vessels whose
    true flow is above *floor*; None without inflow on both sides."""
    if not (inflow_true > 0 and inflow_model > 0):
        return None
    t, m = np.asarray(q_true, dtype=float), np.asarray(q_model, dtype=float)
    keep = np.abs(t) > floor
    if not keep.any():
        return None
    st, sm = t[keep] / inflow_true, m[keep] / inflow_model
    return float(np.median(np.abs(sm - st) / np.abs(st)))


def direction_agreement_at_ends(inflow_true: Sequence[float], inflow_model: Sequence[float], floor: float) -> Optional[float]:
    """The share of open ends whose true flow is above *floor* that the model
    has blood entering by, or leaving by, as the truth does; a model flow of
    exactly 0 (a closed end) is wrong."""
    t, m = np.asarray(inflow_true, dtype=float), np.asarray(inflow_model, dtype=float)
    keep = np.abs(t) > floor
    if not keep.any():
        return None
    return float(np.mean(np.sign(m[keep]) == np.sign(t[keep])))


def inflow_share(inflow_through_open: Sequence[float], inflow_through_designated: Sequence[float]) -> Optional[float]:
    """Of the blood entering through the ends given, the share entering
    through the open ones (each end's inflow, negative where blood leaves)."""
    into_open = float(np.clip(np.asarray(inflow_through_open, dtype=float), 0.0, None).sum())
    into_all = into_open + float(np.clip(np.asarray(inflow_through_designated, dtype=float), 0.0, None).sum())
    return into_open / into_all if into_all > 0 else None


def widest_ends(ends: Sequence[Any], side: str, count: int) -> list:
    """The *count* widest open ends on *side* (``surround.ARTERIAL`` /
    ``VENOUS``): what a person picks as an image's inlets or outlets."""
    on_side = [e for e in ends if e.side == side]
    on_side.sort(key=lambda e: (-(e.diameter_um or 0.0), str(e.node)))
    return [e.node for e in on_side[:count]]


# --- the networks --------------------------------------------------------------------------


def set_resistances(G: nx.MultiGraph, run_basis: str = "plasma_column") -> None:
    """Poiseuille resistances under the default viscosity law, each vessel on
    its own diameter basis: the same on the truth and every crop."""
    from haemolynx.haemodynamics.poiseuille import PoiseuilleModel, edge_diameter_basis, set_edge_resistance

    model = PoiseuilleModel(40.0, 100.0)
    for *_e, data in G.edges(data=True):
        resistance = model.resistance_of_uniform_segment(
            float(data["length"]), float(data["diameter_um"]), diameter_basis=edge_diameter_basis(data, run_basis)
        )
        set_edge_resistance(data, resistance)


def truth_network(template, *, seed: int, scale: float, axes: str, bed_model: str, inlet_pa: float, outlet_pa: float):
    """The network drawn larger, closed, labelled and solved, and its roles."""
    from haemolynx.graph.branch_order import assign_vessel_branch_orders
    from haemolynx.haemodynamics.open_ends import OpenEndConditions
    from haemolynx.synthetic import GenerationOptions, boundary_nodes, generate_network

    G = generate_network(
        template, seed=seed, options=GenerationOptions(scale=scale, scale_axes=axes, bed_model=bed_model)
    )
    roles = boundary_nodes(G)
    assign_vessel_branch_orders(
        G, list(roles["inlet"]), outlet_nodes=list(roles["outlet"]),
        arteriole_boundary_nodes=list(roles["arteriole_boundary"]), venule_boundary_nodes=list(roles["venule_boundary"]),
        large_arteriole_boundary_nodes=list(roles["large_arteriole_boundary"]),
        large_venule_boundary_nodes=list(roles["large_venule_boundary"]),
    )
    set_resistances(G)
    _solve(G, OpenEndConditions(), list(roles["inlet"]), list(roles["outlet"]), inlet_pa, outlet_pa)
    return G, roles


def box_positions(big_um: Sequence[float], extent_um: Sequence[float]) -> np.ndarray:
    """The corners ``lo`` a box of *extent_um* may take in a box of *big_um*:
    every :data:`BOX_STEPS`-th of its extent along an axis the network was
    grown along, kept :data:`BOX_GAP` of it off the outer faces there, so
    each face it has there cuts vessels; at 0 along any other."""
    big, own = np.asarray(big_um, dtype=float), np.asarray(extent_um, dtype=float)
    per_axis = []
    for b, o in zip(big, own):
        room = b - o
        if room <= BOX_GAP * o * 2:
            per_axis.append(np.array([max(room / 2.0, 0.0)]))
            continue
        per_axis.append(np.arange(BOX_GAP * o, room - BOX_GAP * o + 1e-9, o / BOX_STEPS))
    return np.array(np.meshgrid(*per_axis, indexing="ij")).reshape(3, -1).T


def tree_vessels_cut(G: nx.MultiGraph, corners: np.ndarray, extent_um: Sequence[float]) -> np.ndarray:
    """For each box (``lo`` in *corners*, of *extent_um*), how many arterial
    and how many venous vessels its faces cut: ``(boxes, 2)``."""
    from haemolynx.synthetic.surround import ARTERIAL, VENOUS, vessel_side

    points, index, sides = [], [], []
    for i, (_u, _v, data) in enumerate(G.edges(data=True)):
        side = vessel_side(data)
        if side in (ARTERIAL, VENOUS):
            voxels = np.asarray(data["voxels"], dtype=float).reshape(-1, 3)
            points.append(voxels)
            index.append(np.full(len(voxels), len(sides)))
            sides.append(side == VENOUS)
    if not points:
        return np.zeros((len(corners), 2), dtype=int)
    points, index, venous = np.vstack(points), np.concatenate(index), np.asarray(sides)
    total = np.bincount(index, minlength=len(venous))
    own = np.asarray(extent_um, dtype=float)
    out = np.zeros((len(corners), 2), dtype=int)
    for b, lo in enumerate(corners):
        inside = np.all((points >= lo) & (points <= lo + own), axis=1)
        held = np.bincount(index, weights=inside, minlength=len(venous))
        cut = (held > 0) & (held < total)
        out[b] = (int(np.sum(cut & ~venous)), int(np.sum(cut & venous)))
    return out


def candidate_boxes(G: nx.MultiGraph, extent_um: Sequence[float], inlet_count: int, outlet_count: int) -> np.ndarray:
    """The ``lo`` of each box an image of *G* may be taken from, best first:
    of :func:`box_positions`, those whose faces cut at least *inlet_count*
    arterial and *outlet_count* venous vessels (an image is framed round
    vessels it can be fed through), nearest the centre first; then the
    others cutting at least one of each, by how much of the scarcer side they
    cut. A box cutting no vessel of a side is no candidate."""
    from haemolynx.graph.dead_ends import graph_image_extent_um

    big = graph_image_extent_um(G)
    corners = box_positions(big, extent_um)
    cut = tree_vessels_cut(G, corners, extent_um)
    centre = (big - np.asarray(extent_um, dtype=float)) / 2.0
    distance = np.linalg.norm(corners - centre, axis=1)
    enough = (cut[:, 0] >= inlet_count) & (cut[:, 1] >= outlet_count)
    scarcer = np.minimum(cut[:, 0] / max(inlet_count, 1), cut[:, 1] / max(outlet_count, 1))
    order = np.lexsort((distance, -scarcer, ~enough))
    return corners[[i for i in order if cut[i, 0] > 0 and cut[i, 1] > 0]]


def stand_in_image(crop: nx.MultiGraph, inlet_count: int, outlet_count: int):
    """``(image, inlets, outlets, open ends)``: the largest piece of *crop*
    cut with both an arterial and a venous vessel, its widest arterial and
    venous open ends its inlets and outlets; None for a crop with no such
    piece."""
    from haemolynx.synthetic.surround import ARTERIAL, VENOUS, read_open_ends

    ends = read_open_ends(crop)
    pieces = sorted(nx.connected_components(crop), key=lambda p: (-crop.subgraph(p).number_of_edges(), min(map(str, p))))
    for piece in pieces:
        own = [e for e in ends if e.node in piece]
        inlets = widest_ends(own, ARTERIAL, inlet_count)
        outlets = widest_ends(own, VENOUS, outlet_count)
        if inlets and outlets:
            return crop.subgraph(piece).copy(), inlets, outlets, own
    return None


def _flows(C: nx.MultiGraph, truth: nx.MultiGraph) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Each vessel of *C*: ``(edges, q_true, q_model, interior)``, both flows
    signed along the source vessel's ``u -> v``."""
    from haemolynx.graph.crop import CROP_FROM, CROP_SOURCE_EDGE
    from haemolynx.graph.dead_ends import graph_image_extent_um

    extent = graph_image_extent_um(C)
    edges, q_true, q_model, interior = [], [], [], []
    for a, b, key, data in C.edges(keys=True, data=True):
        u, v, k = data[CROP_SOURCE_EDGE]
        start = data[CROP_FROM]
        end = b if start == a else a
        source = truth.edges[u, v, k]
        q_true.append(float(source["conductance"]) * (truth.nodes[u]["pressure"] - truth.nodes[v]["pressure"]))
        q_model.append(float(data["conductance"]) * (C.nodes[start]["pressure"] - C.nodes[end]["pressure"]))
        edges.append((a, b, key))
        far = [float(np.min(np.minimum(C.nodes[n]["pos"], extent - C.nodes[n]["pos"]))) for n in (a, b)]
        interior.append(min(far) > MARGIN_UM)
    return edges, np.asarray(q_true), np.asarray(q_model), np.asarray(interior, dtype=bool)


def _inflow_at(C: nx.MultiGraph, node: Any, q_true: Mapping, q_model: Mapping) -> tuple[float, float]:
    """The flow entering the box through end *node*, true and modelled."""
    from haemolynx.graph.crop import CROP_FROM

    (a, b, key), = C.edges(node, keys=True)
    sign = 1.0 if C.edges[a, b, key][CROP_FROM] == node else -1.0
    edge = (a, b, key) if (a, b, key) in q_true else (b, a, key)
    return sign * q_true[edge], sign * q_model[edge]


def score_crop(C, truth, inlets, outlets, ends) -> dict[str, Any]:
    """Every score of one solved crop, read off its vessels' flows. See the
    module."""
    from haemolynx.statistics.flux_heterogeneity import CARRYING_FLOOR_FRACTION
    from haemolynx.synthetic.surround import ARTERIAL, CAPILLARY, VENOUS

    edges, q_true, q_model, interior = _flows(C, truth)
    floor = CARRYING_FLOOR_FRACTION * float(np.max(np.abs(q_true))) if len(q_true) else 0.0
    by_true = dict(zip(edges, q_true))
    by_model = dict(zip(edges, q_model))
    designated_in = np.asarray([_inflow_at(C, n, by_true, by_model) for n in inlets]).reshape(-1, 2)
    designated_out = np.asarray([_inflow_at(C, n, by_true, by_model) for n in outlets]).reshape(-1, 2)
    open_ends = [e for e in ends if e.node not in set(inlets) | set(outlets)]
    at_open = {e.node: _inflow_at(C, e.node, by_true, by_model) for e in open_ends}
    true_inflow, model_inflow = (float(v) for v in designated_in.sum(axis=0))
    row: dict[str, Any] = {
        "vessels_in_crop": C.number_of_edges(),
        "open_ends": len(open_ends),
        "designated_inflow_ratio": model_inflow / true_inflow if true_inflow > 0 else None,
        "designated_inlets_fed_in_truth": float(np.mean(designated_in[:, 0] > 0)) if len(designated_in) else None,
        "designated_outlets_drained_in_truth": float(np.mean(designated_out[:, 0] < 0)) if len(designated_out) else None,
    }
    for which, column in (("truth", 0), ("model", 1)):
        row[f"open_end_inflow_share_{which}"] = inflow_share(
            [flows[column] for flows in at_open.values()], [*designated_in[:, column], *designated_out[:, column]]
        )
    for label, keep in (("", np.ones(len(edges), dtype=bool)), (" [interior]", interior)):
        scores = flow_scores(q_true[keep], q_model[keep], floor)
        scores["share_error_median"] = share_error_median(q_true[keep], q_model[keep], floor, true_inflow, model_inflow)
        row.update({f"{name}{label}": value for name, value in scores.items()})
    # Whatever the model: how often a tree end runs on upstream of the image
    # (an arteriole feeding it, a venule draining it), so the surround's
    # trunk (D5) is right there; :func:`trunk_rule` pools the counts.
    for side, sign in ((ARTERIAL, 1.0), (VENOUS, -1.0)):
        true_in = [at_open[e.node][0] for e in open_ends if e.side == side and abs(at_open[e.node][0]) > floor]
        row[f"{side}_ends_upstream_in_truth"] = float(np.mean(np.sign(true_in) == sign)) if true_in else None
        row[f"{side}_ends_upstream"] = int(np.sum(np.sign(true_in) == sign))
        row[f"{side}_ends_judged"] = len(true_in)
    for label, sides in (("", (ARTERIAL, VENOUS, CAPILLARY)), ("_trees", (ARTERIAL, VENOUS)), ("_capillaries", (CAPILLARY,))):
        chosen = [at_open[e.node] for e in open_ends if e.side in sides]
        row[f"open_end_direction_agreement{label}"] = direction_agreement_at_ends(
            [t for t, _m in chosen], [m for _t, m in chosen], floor
        )
        row[f"open_ends{label}_scored"] = sum(1 for t, _m in chosen if abs(t) > floor)
    return row


def true_pressure_at(C: nx.MultiGraph, truth: nx.MultiGraph, node: Any) -> float:
    """The truth's pressure where end *node* of *C* lies: along its source
    vessel by the share of it the piece spans (pressure falls linearly along
    a vessel of one width, its resistance going with its length)."""
    from haemolynx.graph.crop import CROP_FROM, CROP_SOURCE_EDGE, CROP_SPAN

    (a, b, key), = C.edges(node, keys=True)
    data = C.edges[a, b, key]
    u, v, _k = data[CROP_SOURCE_EDGE]
    start, end = data[CROP_SPAN]
    share = start if data[CROP_FROM] == node else end
    p_u, p_v = float(truth.nodes[u]["pressure"]), float(truth.nodes[v]["pressure"])
    return p_u + (p_v - p_u) * share


def oracle_conditions(C: nx.MultiGraph, truth: nx.MultiGraph, ends: Sequence[Any], inlet_pa: float, outlet_pa: float):
    """Every end in *ends* held at the truth's pressure there: what a perfect
    surround would give. Each is joined to the inlet and the outlet level by
    conductances :data:`ORACLE_STIFFNESS` times the network's largest, split
    so the divider between them sits at its pressure."""
    from haemolynx.haemodynamics.open_ends import OpenEndConditions

    total = ORACLE_STIFFNESS * max(float(d["conductance"]) for *_e, d in C.edges(data=True))
    to_inlet, to_outlet = {}, {}
    for node in ends:
        share = float(np.clip((true_pressure_at(C, truth, node) - outlet_pa) / (inlet_pa - outlet_pa), 0.0, 1.0))
        to_inlet[node], to_outlet[node] = total * share, total * (1.0 - share)
    return OpenEndConditions(model="oracle", to_inlet=to_inlet, to_outlet=to_outlet)


def solve_surrounded(image, template, inlets, outlets, *, seed: int, scale: float, axes: str, bed_model: str,
                     inlet_pa: float, outlet_pa: float):
    """``(C, S)``: *image* solved inside a network drawn round it to
    *template* (*S*, solved, its outer inlets and outlets at *inlet_pa* and
    *outlet_pa*, an end it leaves open under ``estimated``), and a copy of
    *image* carrying *S*'s pressures and its own conductances, to score."""
    from haemolynx import haemodynamics as hd
    from haemolynx.synthetic import GenerationOptions, boundary_nodes, surround_network

    S = surround_network(
        image, template, inlets=inlets, outlets=outlets,
        options=GenerationOptions(scale=scale, scale_axes=axes, bed_model=bed_model), seed=seed,
    )
    set_resistances(S)
    roles = boundary_nodes(S)
    conductance, node_list = hd.build_conductance_matrix_from_graph(S, solver="sparse")
    conditions = hd.open_ends.open_end_conditions(
        "estimated", S, inlet_nodes=roles["inlet"], outlet_nodes=roles["outlet"], inlet_p_bc=inlet_pa,
        outlet_p_bc=outlet_pa, conductance=conductance, node_list=node_list,
    )
    _solve(S, conditions, roles["inlet"], roles["outlet"], inlet_pa, outlet_pa)
    C = copy.deepcopy(image)
    set_resistances(C)
    for node in C:
        C.nodes[node]["pressure"] = S.nodes[node]["pressure"]
    return C, S


def image_template(image, inlets, outlets, roles: Mapping[str, Sequence[Any]]):
    """The template a run of *image* would measure for itself: its
    statistics report's sections (the template source a run takes by
    default), its designated *inlets* and *outlets*, and the boundary nodes
    of the truth's *roles* that lie in it, as perfect vessel masks would
    give them; its box, the image's."""
    from haemolynx.synthetic.template import SOURCE_STATISTICS, measure_template

    inside = {role: [n for n in nodes if n in image] for role, nodes in roles.items()}
    return measure_template(
        image, inlet_nodes=inlets, outlet_nodes=outlets,
        arteriole_boundary_nodes=inside.get("arteriole_boundary", ()),
        venule_boundary_nodes=inside.get("venule_boundary", ()),
        large_arteriole_boundary_nodes=inside.get("large_arteriole_boundary", ()),
        large_venule_boundary_nodes=inside.get("large_venule_boundary", ()),
        samples=False, source=SOURCE_STATISTICS,
    )


def run_case(
    template, *, seed: int, scale: float, axes: str, bed_model: str, inlet_pa: float, outlet_pa: float,
    surround: bool = True, surround_scale: Optional[float] = None,
    surround_templates: Sequence[str] = (TRUTH_TEMPLATE,),
) -> list[dict]:
    """One truth network, its crop, every model's scores, and with
    *surround* a surround's for each of *surround_templates*
    (:data:`SURROUND_TEMPLATES`; :func:`solve_surrounded`, at
    *surround_scale*, the truth's *scale* without one)."""
    from haemolynx import haemodynamics as hd

    from haemolynx.graph import crop_graph_to_box

    truth, roles = truth_network(
        template, seed=seed, scale=scale, axes=axes, bed_model=bed_model, inlet_pa=inlet_pa, outlet_pa=outlet_pa
    )
    own = np.asarray(template.extent_um, dtype=float)
    found = None
    for lo in candidate_boxes(truth, own, template.inlet_count, template.outlet_count)[:BOX_TRIES]:
        found = stand_in_image(crop_graph_to_box(truth, lo, lo + own), template.inlet_count, template.outlet_count)
        if found is not None:
            break
    if found is None:
        raise ValueError(
            "no box of the template's size holds a piece cut with both an arterial and a venous vessel: "
            "the trees reach too little of the network drawn; draw it at a smaller --scale"
        )
    image, inlets, outlets, ends = found
    rows = []
    # The oracle first: every end, the designated ones too, at the truth's
    # pressure. It scores the harness itself (it must match the truth) and is
    # the most a surround could give.
    case = {
        "bed": bed_model, "seed": seed, "truth_vessels": truth.number_of_edges(),
        "box_lo_um": " ".join(f"{v:.0f}" for v in lo), "inlets": len(inlets), "outlets": len(outlets),
    }
    C = copy.deepcopy(image)
    set_resistances(C)
    _solve(C, oracle_conditions(C, truth, [e.node for e in ends], inlet_pa, outlet_pa), [], [], inlet_pa, outlet_pa)
    rows.append({"model": ORACLE, **case, **score_crop(C, truth, inlets, outlets, ends)})
    for model, extra in MODELS:
        C = copy.deepcopy(image)
        set_resistances(C)
        conductance = node_list = None
        if model == "estimated":
            conductance, node_list = hd.build_conductance_matrix_from_graph(C, solver="sparse")
        conditions = hd.open_ends.open_end_conditions(
            model, C, inlet_nodes=inlets, outlet_nodes=outlets, inlet_p_bc=inlet_pa, outlet_p_bc=outlet_pa,
            conductance=conductance, node_list=node_list, **extra,
        )
        _solve(C, conditions, inlets, outlets, inlet_pa, outlet_pa)
        label = model + (f" k={extra['terminal_resistance_multiple']:g}" if extra else "")
        rows.append({"model": label, **case, **score_crop(C, truth, inlets, outlets, ends)})
    for which in surround_templates if surround else ():
        from haemolynx.synthetic.surround import SURROUND_RECORD

        label = SURROUND_TEMPLATES[which]
        start = time.perf_counter()
        try:
            drawn_to = template if which == TRUTH_TEMPLATE else image_template(image, inlets, outlets, roles)
            C, S = solve_surrounded(
                image, drawn_to, inlets, outlets, seed=seed + SURROUND_SEED_OFFSET,
                scale=scale if surround_scale is None else surround_scale, axes=axes, bed_model=bed_model,
                inlet_pa=inlet_pa, outlet_pa=outlet_pa,
            )
        except ValueError as failure:
            # A surround that cannot be drawn (a shell too thin to hold a
            # join for each open end) is a run it loses, not one left out.
            rows.append({"model": label, **case, "surround_failed": str(failure)})
            continue
        record = S.graph[SURROUND_RECORD]
        drawn = {
            "surround_vessels": S.number_of_edges(), "surround_trunks": record["trunks"],
            "surround_bed_joins": record["bed_joins"], "surround_left_open": len(record["left_open"]),
            "surround_template_vessels": drawn_to.vessel_count, "surround_seconds": time.perf_counter() - start,
        }
        rows.append({"model": label, **case, **score_crop(C, truth, inlets, outlets, ends), **drawn})
    return rows


# --- the report ----------------------------------------------------------------------------


def summarise(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict]:
    """Per model, in the order first met: each numeric column's median over
    the runs (None where no run has it)."""
    out: dict[str, dict] = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        chosen = [r for r in rows if r["model"] == model]
        summary: dict[str, Any] = {"model": model, "runs": len(chosen)}
        for column in columns:
            values = [
                float(r[column]) for r in chosen
                if isinstance(r.get(column), (int, float, np.number)) and not isinstance(r.get(column), bool)
            ]
            summary[column] = float(np.median(values)) if values else None
        out[model] = summary
    return list(out.values())


def trunk_rule(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(right, judged)``: over every case (its oracle row; the counts are
    the truth's, the same in every model's), the arteriole and venule open
    ends upstream of the image, where the surround's trunk (D5) is right,
    and those whose true flow is above the floor."""
    cases = [r for r in rows if r["model"] == ORACLE]
    right = sum(int(r.get(f"{side}_ends_upstream") or 0) for r in cases for side in ("arterial", "venous"))
    judged = sum(int(r.get(f"{side}_ends_judged") or 0) for r in cases for side in ("arterial", "venous"))
    return right, judged


def trunk_rule_line(right: int, judged: int) -> str:
    """The report's line on D5: how often the trunk was right, and, at half
    or fewer, that D5 is to be revisited (Harvey, 2026-10-27)."""
    if not judged:
        return "D5 (every arteriole and venule end a trunk): no tree end above the floor to judge it on."
    line = f"D5 (every arteriole and venule end a trunk): right at {right} of {judged} tree ends ({right / judged:.0%})."
    if right * 2 <= judged:
        line += " **Half or fewer: revisit D5.**"
    return line


def kill_criterion(rows: Sequence[Mapping[str, Any]], label: str = SURROUND) -> str:
    """The plan's go / no-go line: the medians over the runs of the
    surround row *label* against :data:`BASELINE`'s on each of
    :data:`KILL_SCORES`, and in how many runs (one bed and seed) it does
    better; met only where it beats it on both medians and was drawn in
    every run."""
    summary = {r["model"]: r for r in summarise(rows, [score for score, _name, _higher in KILL_SCORES])}
    if label not in summary or BASELINE not in summary:
        return f"Kill criterion: no {label} or {BASELINE} row to judge it on."
    runs = {
        (r["bed"], r["seed"]): r for r in rows if r["model"] == label
    }
    paired = [(runs[(r["bed"], r["seed"])], r) for r in rows if r["model"] == BASELINE and (r["bed"], r["seed"]) in runs]
    failed = sum(1 for r in runs.values() if r.get("surround_failed"))
    parts, met = [], not failed
    if failed:
        parts.append(f"not drawn in {failed} of {len(runs)} runs")
    for score, name, higher in KILL_SCORES:
        ours, theirs = summary[label][score], summary[BASELINE][score]
        if ours is None or theirs is None:
            parts.append(f"{name}: not measured")
            met = False
            continue
        better = ours > theirs if higher else ours < theirs
        met = met and better
        wins = sum(
            1 for s, b in paired
            if s.get(score) is not None and b.get(score) is not None and (s[score] > b[score] if higher else s[score] < b[score])
        )
        parts.append(f"{name} {ours:.3g} against {theirs:.3g} (better in {wins} of {len(paired)} runs)")
    verdict = "**met**" if met else "**not met**: the surround ships as a sensitivity tool at most (D9)"
    return f"Kill criterion ({label} beats {BASELINE} on both): " + "; ".join(parts) + f". {verdict}."


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


REPORTED = (
    "direction_agreement",
    "direction_agreement [interior]",
    "flow_error_median",
    "flow_error_median [interior]",
    "share_error_median",
    "share_error_median [interior]",
    "flow_rank_correlation",
    "open_end_direction_agreement",
    "open_end_direction_agreement_trees",
    "open_end_direction_agreement_capillaries",
    "arterial_ends_upstream_in_truth",
    "venous_ends_upstream_in_truth",
    "open_end_inflow_share_model",
    "open_end_inflow_share_truth",
    "designated_inflow_ratio",
)


def _template(name: str, source: str):
    if name == "fixture":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from synthetic_fixtures import fixture_template

        return fixture_template()
    from haemolynx.synthetic import load_template

    return load_template(name, source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--template", required=True, help="'fixture', or a template run's .haemorun / .haemobatch")
    parser.add_argument("--source", default="statistics", choices=("statistics", "graph"))
    # The trees reach their template's distance from the outer faces, so a
    # truth much larger than its template holds no box fed from both sides.
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--axes", default="xy", choices=("xy", "xyz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--beds", nargs="+", default=["delaunay", "voronoi"], choices=("delaunay", "voronoi"))
    parser.add_argument("--inlet-pa", type=float, default=4500.0)
    parser.add_argument("--outlet-pa", type=float, default=1000.0)
    parser.add_argument("--no-surround", dest="surround", action="store_false", help="score the open-end models alone")
    parser.add_argument("--surround-scale", type=float, default=None, help="the surround's box (default: --scale)")
    parser.add_argument(
        "--surround-templates", nargs="+", default=list(SURROUND_TEMPLATES), choices=tuple(SURROUND_TEMPLATES),
        help="what each surround is drawn to: the truth's own template, the stand-in image's measured one, or both",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    template = _template(args.template, args.source)
    rows: list[dict] = []
    for bed in args.beds:
        for seed in args.seeds:
            start = time.perf_counter()
            try:
                case = run_case(
                    template, seed=seed, scale=args.scale, axes=args.axes, bed_model=bed,
                    inlet_pa=args.inlet_pa, outlet_pa=args.outlet_pa,
                    surround=args.surround, surround_scale=args.surround_scale,
                    surround_templates=args.surround_templates,
                )
            except ValueError as failure:
                print(f"{bed} seed {seed}: skipped, {failure}", flush=True)
                continue
            rows.extend(case)
            print(f"{bed} seed {seed}: {case[0]['truth_vessels']} vessels drawn, {case[0]['vessels_in_crop']} in the "
                  f"crop, {case[0]['open_ends']} open ends ({time.perf_counter() - start:.0f} s)", flush=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "surround_check_runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, [c for c in columns if c not in ("bed", "seed", "model")])
    shown = ["model", "runs", *REPORTED]
    d5 = trunk_rule_line(*trunk_rule(rows))
    lines = [d5]
    if args.surround:
        surround_scale = args.scale if args.surround_scale is None else args.surround_scale
        lines[:0] = [kill_criterion(rows, SURROUND_TEMPLATES[which]) for which in args.surround_templates]
        lines.append(
            f"Each surround is drawn at {surround_scale:g}x along {args.axes}: `{SURROUND}` to the template the truth "
            f"was drawn to (kind: it draws the truth's kind of tissue), `{SURROUND_TEMPLATES[IMAGE_TEMPLATE]}` to "
            "the template a run of the stand-in image measures for itself (its statistics, its inlets and outlets, "
            "the truth's boundary nodes within it as perfect vessel masks would give them)."
        )
    report = (
        f"# Cut-out harness: open-end models{' and the surround' if args.surround else ''}\n\n"
        f"Template `{args.template}` ({args.source}), truth at {args.scale:g}x along {args.axes}, closed; "
        f"beds {', '.join(args.beds)}, seeds {', '.join(map(str, args.seeds))}; inlet {args.inlet_pa:g} Pa, "
        f"outlet {args.outlet_pa:g} Pa. Medians over the runs; per-run rows in `surround_check_runs.csv`.\n\n"
        + markdown_table(summary, shown) + "\n\n" + "\n\n".join(lines) + "\n"
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(markdown_table(summary, shown))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
