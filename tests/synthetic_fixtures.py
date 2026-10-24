"""Fixtures for the synthetic-network tests: a tiny hand-built hierarchical
network every template field has a hand-computed value for, a hand-set
template, and a ``.haemorun`` writer that saves a run of either.

The tiny network (positions in (z, y, x) um, a 100 x 200 x 300 um box of
2 x 1 x 1 um voxels, so a z/x mix-up shows):

    inlet 0 --Large_Art1-- 1 --Art1-- 2 --Art2-- 4 --B01-- 6 --B02-- 7 --Ven1-- 8 --Large_Ven1-- 9 outlet
                            \\--Art1-- 3 --Art2-- 5 --B01--/   \\--B03-- 10 (blind)
                                                    4 --B01-- 11 (leaves the image at z = 0)

Node 1 is the large-arteriole boundary, 4 and 5 the arteriole boundaries,
7 the venule boundary, 8 the large-venule boundary. One loop (1-2-4-6-5-3),
four dead ends: 0, 9 and 11 on faces, 10 inside.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from haemolynx.graph._helpers import calculate_path_length
from haemolynx.graph.dead_ends import (
    BLIND,
    DEAD_END_KIND,
    DEAD_END_KIND_SOURCE,
    LEAVES_IMAGE,
    SOURCE_CLASSIFIER,
    record_image_extent,
)
from haemolynx.gui.results import StageLayers
from haemolynx.gui.run_snapshot import RunSnapshot, write_run_snapshot
from haemolynx.gui.stage_checkpoints import StageCheckpoint
from haemolynx.synthetic.template import (
    STATISTICS_REPORT_KEY,
    ClassTarget,
    DeadEndTarget,
    Endpoint,
    LoopTarget,
    NetworkTemplate,
    OrderTarget,
    Provenance,
)

TINY_SHAPE = (51, 201, 301)
TINY_VOXEL_ZYX = (2.0, 1.0, 1.0)
TINY_EXTENT_UM = (100.0, 200.0, 300.0)

TINY_POSITIONS: dict[int, tuple[float, float, float]] = {
    0: (50.0, 100.0, 0.0),
    1: (50.0, 100.0, 60.0),
    2: (50.0, 70.0, 100.0),
    3: (50.0, 130.0, 100.0),
    4: (50.0, 70.0, 150.0),
    5: (50.0, 130.0, 150.0),
    6: (50.0, 100.0, 200.0),
    7: (50.0, 100.0, 240.0),
    8: (50.0, 100.0, 270.0),
    9: (50.0, 100.0, 300.0),
    10: (50.0, 120.0, 260.0),
    11: (0.0, 70.0, 160.0),
}

#: (u, v, label, diameter, via): ``via`` adds interior polyline points.
TINY_EDGES: tuple[tuple, ...] = (
    (0, 1, "Large_Art1", 20.0, ()),
    (1, 2, "Art1", 12.0, ()),
    (1, 3, "Art1", 10.0, ()),
    (2, 4, "Art2", 8.0, ()),
    (3, 5, "Art2", 7.0, ()),
    (4, 6, "B01", 5.0, ()),
    (5, 6, "B01", 5.0, ()),
    (4, 11, "B01", 4.0, ()),
    (6, 7, "B02", 5.0, ((50.0, 112.0, 220.0),)),
    (7, 10, "B03", 4.0, ()),
    (8, 7, "Ven1", 9.0, ()),
    (9, 8, "Large_Ven1", 18.0, ()),
)

TINY_ROLES: dict[str, tuple[int, ...]] = {
    "inlet": (0,),
    "outlet": (9,),
    "arteriole_boundary": (4, 5),
    "venule_boundary": (7,),
    "large_arteriole_boundary": (1,),
    "large_venule_boundary": (8,),
}

TINY_DEAD_END_KINDS = {0: LEAVES_IMAGE, 9: LEAVES_IMAGE, 11: LEAVES_IMAGE, 10: BLIND}


def _polyline(u: int, v: int, via: Sequence[tuple]) -> list[list[float]]:
    points = [TINY_POSITIONS[u], *via, TINY_POSITIONS[v]]
    dense: list[list[float]] = []
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.5)) + 1)
        for t in np.linspace(0.0, 1.0, steps)[:-1]:
            dense.append((a + t * (b - a)).tolist())
    dense.append(list(points[-1]))
    return dense


def tiny_hierarchical_graph(*, with_report: bool = False) -> nx.MultiGraph:
    """The network above, labelled and measured as a finished run leaves it."""
    G = nx.MultiGraph()
    for node, pos in TINY_POSITIONS.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v, label, diameter, via in TINY_EDGES:
        voxels = _polyline(u, v, via)
        G.add_edge(
            u,
            v,
            voxels=voxels,
            length=calculate_path_length(voxels),
            branch_order=label,
            diameter_um=diameter,
            diameter_source="measured",
        )
    for node, kind in TINY_DEAD_END_KINDS.items():
        G.nodes[node][DEAD_END_KIND] = kind
        G.nodes[node][DEAD_END_KIND_SOURCE] = SOURCE_CLASSIFIER
    G.graph["voxel_size"] = TINY_VOXEL_ZYX
    G.graph["image_voxel_size_zyx"] = TINY_VOXEL_ZYX
    G.graph["image_voxel_size_xyz"] = tuple(reversed(TINY_VOXEL_ZYX))
    record_image_extent(G, TINY_SHAPE, TINY_VOXEL_ZYX)
    if with_report:
        G.graph[STATISTICS_REPORT_KEY] = {"mode": "fast", "csv": None, "sections": tiny_sections(G)}
    return G


def tiny_sections(G: nx.MultiGraph | None = None) -> dict:
    """The statistics report sections a run of the tiny network leaves, as
    the statistics modules compute them (the same functions
    ``export_results`` calls)."""
    from haemolynx.statistics import (
        compute_branch_order_statistics,
        compute_vessel_statistics_by_measure,
    )
    from haemolynx.synthetic.template import (
        BRANCH_ORDERS_SECTION,
        OPTIONAL_SECTIONS,
        REQUIRED_SECTIONS,
    )

    G = tiny_hierarchical_graph() if G is None else G
    positions = {n: G.nodes[n]["pos"] for n in G}
    enabled = frozenset(n for n in REQUIRED_SECTIONS + OPTIONAL_SECTIONS if n != BRANCH_ORDERS_SECTION)
    sections = dict(
        compute_vessel_statistics_by_measure(
            G,
            node_positions=positions,
            voxel_size=TINY_VOXEL_ZYX,
            image_dimensions=TINY_SHAPE,
            enabled_measures=enabled,
            inlet_nodes=list(TINY_ROLES["inlet"]),
            outlet_nodes=list(TINY_ROLES["outlet"]),
            arteriole_boundary_nodes=list(TINY_ROLES["arteriole_boundary"]),
            venule_boundary_nodes=list(TINY_ROLES["venule_boundary"]),
        )
    )
    sections[BRANCH_ORDERS_SECTION] = compute_branch_order_statistics(G, node_positions=positions)
    return sections


def tiny_expected() -> dict:
    """Hand-computed values of the tiny network, for the reader tests."""
    G = tiny_hierarchical_graph()
    lengths = {(u, v): d["length"] for u, v, d in G.edges(data=True)}
    by_label: dict[str, list[float]] = {}
    for u, v, d in G.edges(data=True):
        by_label.setdefault(d["branch_order"], []).append(d["length"])
    return {
        "total_length_um": sum(lengths.values()),
        "edge_count": 12,
        "node_count": 12,
        "cyclomatic": 1,
        "dead_ends": {"total": 4, "leaves_image": 3, "interior": 1, "artefacts": 0},
        "order_counts": {"Large_Art1": 1, "Art1": 2, "Art2": 2, "BO1": 3, "BO2": 1, "BO3": 1, "Ven1": 1, "Large_Ven1": 1},
        "class_counts": {"large_arteriole": 1, "arteriole": 4, "capillary": 5, "venule": 1, "large_venule": 1},
        "capillary_total_um": sum(by_label["B01"]) + sum(by_label["B02"]) + sum(by_label["B03"]),
        "b02_tortuosity": lengths[(6, 7)] / 40.0,
        "inlet_face": "x0",
        "outlet_face": "x1",
        "blind_chain_um": lengths[(7, 10)],
    }


def fixture_template(*, notes: Sequence[str] = ()) -> NetworkTemplate:
    """A hand-set template of a small network the generator is tested on:
    a 120 x 200 x 200 um box, two inlets on x0, two outlets on x1, three
    arteriole and two venule orders, 160 capillaries, 51 loops, 16 dead
    ends (8 at the faces -- the four inlets and outlets among them -- and 8
    inside). The counts are consistent with room to spare: with the 10
    boundary leaves taking two bed edges each, the bed's 106 vertices
    (160 - 51 - 4 + 1) hold the 12 dead ends that are not roots, 88
    junctions of three and 6 of four (handshake: 12 + 264 + 24 = 300 =
    320 - 20), and a tree leaf or two cut by a face still fits."""

    def order(label: str, category: str, index: int, count: int, length: float, diameter: float, cv: float = 0.2) -> OrderTarget:
        return OrderTarget(
            label=label,
            category=category,
            index=index,
            edge_count=count,
            mean_length_um=length,
            mean_tortuosity=1.1,
            mean_emergence_angle_deg=40.0,
            mean_diameter_um=diameter,
            diameter_cv=cv,
        )

    orders = (
        order("Art1", "arteriole", 1, 2, 40.0, 12.0),
        order("Art2", "arteriole", 2, 4, 35.0, 9.5),
        order("Art3", "arteriole", 3, 6, 30.0, 7.5),
        order("Ven1", "venule", 1, 2, 40.0, 14.0),
        order("Ven2", "venule", 2, 4, 32.0, 11.0),
    )
    classes = {
        "arteriole": ClassTarget("arteriole", 12, 12 * 34.0, 34.0, 32.0, 9.0, 8.8, 1.1, 1.08),
        "capillary": ClassTarget("capillary", 160, 160 * 50.0, 50.0, 46.0, 5.0, 4.8, 1.15, 1.12),
        "venule": ClassTarget("venule", 6, 6 * 35.0, 35.0, 33.0, 12.0, 11.8, 1.1, 1.08),
    }
    return NetworkTemplate(
        extent_um=(120.0, 200.0, 200.0),
        voxel_size_zyx=(1.0, 0.98, 0.98),
        image_shape=(121, 205, 205),
        inlets=(Endpoint("x0", (0.4, 0.3, 0.0)), Endpoint("x0", (0.6, 0.7, 0.0))),
        outlets=(Endpoint("x1", (0.4, 0.7, 1.0)), Endpoint("x1", (0.6, 0.3, 1.0))),
        arteriole_boundary_count=6,
        venule_boundary_count=4,
        large_arteriole_boundary_count=0,
        large_venule_boundary_count=0,
        scheme="hierarchical",
        classes=classes,
        orders=orders,
        total_length_um=12 * 34.0 + 160 * 50.0 + 6 * 35.0,
        node_count=None,
        edge_count=178,
        average_degree=None,
        branching_points=None,
        mean_tortuosity=1.14,
        daughter_angle_deg=80.0,
        daughter_angle_sd_deg=None,
        murray_exponent=3.0,
        murray_iqr=(2.7, 3.3),
        # Half the bed's vertex spacing ((V / 106)^(1/3) = 36 um): the distance
        # from a capillary to the nearest one it does not meet.
        intercapillary_distance_um=18.0,
        intercapillary_distance_median_um=17.0,
        # Per the image volume (shape x voxel), as the loop statistic divides.
        loops=LoopTarget(independent_loops=51, independent_loops_per_mm3=51 / (121 * 205 * 205 * 0.98 * 0.98 / 1e9)),
        dead_ends=DeadEndTarget(
            total=16, leaves_image=8, interior=8, artefacts=0, island_ends=0,
            interior_per_mm=None, chain_length_median_um=30.0, chain_length_p10_um=18.0, chain_length_p90_um=50.0,
        ),
        tissue_volume_um3=None,
        provenance=Provenance(None, "statistics", None, None, None, None, tuple(notes)),
    )


def _group(stage: str, title: str = "") -> StageLayers:
    return StageLayers(stage=stage, title=title or stage, note=f"done {stage}")


def write_fixture_run(
    tmp_path: Path,
    *,
    graph: nx.MultiGraph | None = None,
    roles: Mapping[str, Sequence[Any]] | None = None,
    sections: Mapping[str, Any] | None = None,
    with_report: bool = True,
    drop_sections: Sequence[str] = (),
    settings: Mapping[str, Any] | None = None,
    graph_attributes: Mapping[str, Any] | None = None,
    name: str = "template.haemorun",
) -> Path:
    """Write a ``.haemorun`` of a run of *graph* (the tiny network by
    default): a Boundaries checkpoint carrying the role lists and an Export
    checkpoint carrying the graph with its statistics report."""
    G = tiny_hierarchical_graph() if graph is None else graph
    roles = dict(TINY_ROLES if roles is None else roles)
    if with_report:
        report_sections = dict(tiny_sections(G) if sections is None else sections)
        for key in drop_sections:
            report_sections.pop(key, None)
        G.graph[STATISTICS_REPORT_KEY] = {"mode": "fast", "csv": None, "sections": report_sections}
    else:
        G.graph.pop(STATISTICS_REPORT_KEY, None)
    for key, value in (graph_attributes or {}).items():
        G.graph[key] = value
    voxel = tuple(G.graph.get("image_voxel_size_zyx") or TINY_VOXEL_ZYX)
    run_settings = {
        "input_path": tmp_path / "stack.tif",
        "vtk_output_prefix": tmp_path / "out" / "stack",
        "statistics": True,
        "network_source": "segmented_image",
        "segmentation_source": "plasma_column",
        **dict(settings or {}),
    }
    role_lists = {f"{role}_nodes": tuple(roles.get(role, ())) for role in roles}
    checkpoints = (
        StageCheckpoint(stage="build_network", title="3. Graph", group=_group("build_network"), graph=G.copy(), voxel_size_zyx=voxel),
        StageCheckpoint(stage="assign_boundaries", title="4. Boundaries", group=_group("assign_boundaries"), graph=G.copy(), voxel_size_zyx=voxel, **role_lists),
        StageCheckpoint(stage="export_results", title="10. Export", group=_group("export_results"), graph=G, voxel_size_zyx=voxel, **role_lists),
    )
    snapshot = RunSnapshot(
        settings=dict(run_settings),
        results_state={"graph": G, "settings": dict(run_settings), "voxel_size_zyx": voxel},
        checkpoints=checkpoints,
    )
    return write_run_snapshot(tmp_path / name, snapshot)


def write_generated_run(
    tmp_path: Path,
    *,
    seed: int = 11,
    name: str = "e14_5.haemorun",
    settings: Mapping[str, Any] | None = None,
) -> Path:
    """A ``.haemorun`` of a network drawn to :func:`fixture_template`, saved as
    if it had come from an image: labelled, measured, its generator marks
    taken off. What the stage tests use as their template, so a run is
    matched to a network of a few hundred vessels rather than the tiny one."""
    from haemolynx.graph.branch_order import assign_vessel_branch_orders
    from haemolynx.synthetic.generate import GRAPH_NETWORK_SOURCE, SYNTHETIC_CLASS, SYNTHETIC_ROLES, boundary_nodes, generate_network
    from haemolynx.synthetic.template import GRAPH_TEMPLATE_RECORD

    G = generate_network(fixture_template(), seed=seed)
    roles = boundary_nodes(G)
    assign_vessel_branch_orders(
        G,
        roles["inlet"],
        outlet_nodes=roles["outlet"],
        arteriole_boundary_nodes=roles["arteriole_boundary"],
        venule_boundary_nodes=roles["venule_boundary"],
    )
    G.graph.pop(GRAPH_NETWORK_SOURCE, None)
    G.graph.pop(GRAPH_TEMPLATE_RECORD, None)
    for node in G:
        G.nodes[node].pop(SYNTHETIC_ROLES, None)
    for *_e, data in G.edges(data=True):
        data.pop(SYNTHETIC_CLASS, None)
        data["diameter_source"] = "measured"
    extent = np.asarray(G.graph["image_extent_um"], dtype=float)
    voxel = np.asarray(G.graph["image_voxel_size_zyx"], dtype=float)
    shape = tuple(int(round(e / v)) + 1 for e, v in zip(extent, voxel))
    positions = {n: G.nodes[n]["pos"] for n in G}
    from haemolynx.statistics import compute_branch_order_statistics, compute_vessel_statistics_by_measure
    from haemolynx.synthetic.template import BRANCH_ORDERS_SECTION, OPTIONAL_SECTIONS, REQUIRED_SECTIONS

    enabled = frozenset(n for n in REQUIRED_SECTIONS + OPTIONAL_SECTIONS if n != BRANCH_ORDERS_SECTION)
    sections = dict(
        compute_vessel_statistics_by_measure(
            G, node_positions=positions, voxel_size=tuple(voxel), image_dimensions=shape,
            enabled_measures=enabled, inlet_nodes=roles["inlet"], outlet_nodes=roles["outlet"],
            arteriole_boundary_nodes=roles["arteriole_boundary"], venule_boundary_nodes=roles["venule_boundary"],
        )
    )
    sections[BRANCH_ORDERS_SECTION] = compute_branch_order_statistics(G, node_positions=positions)
    return write_fixture_run(tmp_path, graph=G, roles=roles, sections=sections, name=name, settings=settings)


__all__ = [
    "TINY_DEAD_END_KINDS",
    "TINY_EDGES",
    "TINY_EXTENT_UM",
    "TINY_POSITIONS",
    "TINY_ROLES",
    "TINY_SHAPE",
    "TINY_VOXEL_ZYX",
    "fixture_template",
    "tiny_expected",
    "tiny_hierarchical_graph",
    "tiny_sections",
    "write_fixture_run",
    "write_generated_run",
]
