"""The "capillary blocks by mask" perturbation: which vessels a mask's blobs
block, and that blocking them is the capillary block's own occlusion -- on a
network small enough to check by hand.

The network (all at z = 4, 1 um voxels), two routes from an inlet to an
outlet, every vessel a tube of radius 2 um in the segmentation:

    n0 --Art1-- n1 ----------- top ----------- n2
                 |                              |
                left                          right
                 |                              |
                n3 ---------- bottom ---------- n4 --Ven1-- n5
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.haemodynamics import (
    CAPILLARY_BLOCK_TYPES,
    PERTURBATION_TYPES,
    blocks_from_mask,
    is_capillary_block,
    load_capillary_block_mask,
    perturbation_problems,
    settings_for_perturbation_type,
)
from haemolynx.haemodynamics.capillary_block import BLOCK_EDGE_ATTRIBUTE
from haemolynx.haemodynamics.poiseuille import set_edge_resistance
from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, default_schema, run_perturbations

SCHEMA = default_schema()

SHAPE = (8, 32, 54)
Z = 4
NODES = {
    0: (Z, 4, 0),
    1: (Z, 4, 10),
    2: (Z, 4, 40),
    3: (Z, 24, 10),
    4: (Z, 24, 40),
    5: (Z, 24, 50),
}
VESSELS = {
    "art": (0, 1, "Art1", 6.0),
    "top": (1, 2, "BO2", 4.0),
    "left": (1, 3, "BO2", 4.0),
    "right": (2, 4, "BO2", 4.0),
    "bottom": (3, 4, "BO2", 4.0),
    "ven": (4, 5, "Ven1", 6.0),
}
TUBE_RADIUS_UM = 2.0

#: One blob of each kind, as voxel (z, y, x) lists.
#: Wholly inside the top vessel, halfway along it.
ON_TOP = [(z, y, x) for z in (3, 4, 5) for y in (3, 4, 5) for x in (24, 25, 26)]
#: In the tissue between the two routes, 10 um from both.
IN_TISSUE = [(Z, 14, x) for x in (24, 25, 26)]
#: Across the bottom vessel's wall: y 25, 26 inside (1 and 2 um from its
#: centreline), 27-30 outside -- a third of it inside.
ACROSS_BOTTOM_WALL = [(Z, y, 20) for y in range(25, 31)]
#: At the n4 junction: five voxels along the bottom vessel, one as near the
#: right vessel as the bottom one, two along the right vessel.
AT_JUNCTION = [(Z, 24, x) for x in range(35, 40)] + [(Z, 23, 39), (Z, 23, 40), (Z, 22, 40)]
#: Inside segmented vessel the graph has no centreline through.
OFF_GRAPH_SEGMENT = [(Z, 14, x) for x in range(46, 53)]
OFF_GRAPH = [(Z, 14, 49)]


def _distance_to_segment(points: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _segmentation(extra: list = ()) -> np.ndarray:
    """The vessels as tubes of :data:`TUBE_RADIUS_UM`, 0/1."""
    grid = np.indices(SHAPE).reshape(3, -1).T.astype(float)
    inside = np.zeros(len(grid), dtype=bool)
    for u, v, _order, _diameter in VESSELS.values():
        inside |= _distance_to_segment(grid, NODES[u], NODES[v]) <= TUBE_RADIUS_UM
    seg = inside.reshape(SHAPE).astype(np.uint8)
    for voxel in extra:
        seg[voxel] = 1
    return seg


def _blobs(*groups) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    for group in groups:
        for voxel in group:
            mask[voxel] = True
    return mask


def _graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in NODES.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for u, v, order, diameter in VESSELS.values():
        G.add_edge(
            u,
            v,
            branch_order=order,
            diameter_um=diameter,
            length=float(np.linalg.norm(np.subtract(NODES[v], NODES[u]))),
            voxels=[list(map(float, NODES[u])), list(map(float, NODES[v]))],
        )
    for _u, _v, data in G.edges(data=True):
        set_edge_resistance(data, data["length"] / data["diameter_um"] ** 4)
    return G


def _ids(G: nx.MultiGraph) -> dict[str, int]:
    """Each vessel's branchID (its position in the graph's edge order)."""
    order = {(u, v): index for index, (u, v, _k) in enumerate(G.edges(keys=True))}
    return {
        name: order.get((u, v), order.get((v, u)))
        for name, (u, v, _o, _d) in VESSELS.items()
    }


def _by_number(blocks):
    return {block.number: block for block in blocks}


# --- which vessels the blobs block ----------------------------------------------------


def test_a_blob_inside_a_vessel_blocks_that_vessel_and_one_in_the_tissue_blocks_nothing():
    G = _graph()
    ids = _ids(G)
    blocked, summary, blocks = blocks_from_mask(
        G, _blobs(ON_TOP, IN_TISSUE), (1.0, 1.0, 1.0), vessel_mask=_segmentation()
    )

    assert blocked == [list(G.edges(keys=True))[ids["top"]]]
    on_top = next(b for b in blocks if b.voxels == len(ON_TOP))
    in_tissue = next(b for b in blocks if b.voxels == len(IN_TISSUE))
    assert on_top.overlap_fraction == 1.0
    assert on_top.blocked_branch_ids == [ids["top"]]
    assert on_top.centroid_um == pytest.approx((4.0, 4.0, 25.0))
    assert on_top.volume_um3 == pytest.approx(27.0)
    assert in_tissue.overlap_fraction == 0.0
    assert in_tissue.blocked_branch_ids == [] and in_tissue.reason == "below_overlap"
    assert summary["mask_blocks"] == 2
    assert summary["mask_blocks_on_vessels"] == 1
    assert summary["mask_blocks_below_overlap"] == 1
    assert summary["blocked_vessels"] == 1
    assert summary["vessel_mask_source"] == "segmentation"


@pytest.mark.parametrize("fraction, blocks_it", [(0.5, False), (1 / 3, True), (0.34, False), (0.0, True)])
def test_the_overlap_fraction_decides_whether_a_blob_part_in_a_vessel_blocks_it(fraction, blocks_it):
    G = _graph()
    ids = _ids(G)
    blocked, _summary, (block,) = blocks_from_mask(
        G,
        _blobs(ACROSS_BOTTOM_WALL),
        (1.0, 1.0, 1.0),
        vessel_mask=_segmentation(),
        min_overlap_fraction=fraction,
    )

    assert block.voxels_in_vessels == 2
    assert block.overlap_fraction == pytest.approx(2 / 6)
    assert block.vessel_voxels == {ids["bottom"]: 2}
    assert block.blocked_branch_ids == ([ids["bottom"]] if blocks_it else [])
    assert len(blocked) == int(blocks_it)


def test_a_blob_wholly_off_the_vessels_never_blocks_even_at_a_zero_fraction():
    G = _graph()
    blocked, _summary, (block,) = blocks_from_mask(
        G, _blobs(IN_TISSUE), (1.0, 1.0, 1.0), vessel_mask=_segmentation(), min_overlap_fraction=0.0
    )
    assert blocked == [] and block.reason == "below_overlap"


def test_a_blob_at_a_junction_blocks_the_vessel_holding_most_of_it_or_every_one():
    G = _graph()
    ids = _ids(G)
    edges = list(G.edges(keys=True))
    mask, seg = _blobs(AT_JUNCTION), _segmentation()

    largest, _s, (block,) = blocks_from_mask(G, mask, (1.0, 1.0, 1.0), vessel_mask=seg)
    every, summary, _b = blocks_from_mask(
        G, mask, (1.0, 1.0, 1.0), vessel_mask=seg, vessels_per_block="every_overlapped_vessel"
    )

    assert set(block.vessel_voxels) == {ids["bottom"], ids["right"]}
    assert block.vessel_voxels[ids["bottom"]] >= 5 and block.vessel_voxels[ids["right"]] >= 2
    assert sum(block.vessel_voxels.values()) == len(AT_JUNCTION)
    assert largest == [edges[ids["bottom"]]]
    assert set(every) == {edges[ids["bottom"]], edges[ids["right"]]}
    assert summary["vessels_per_block"] == "every_overlapped_vessel"


def test_segmentation_the_graph_has_no_vessel_through_blocks_nothing():
    G = _graph()
    blocked, summary, (block,) = blocks_from_mask(
        G, _blobs(OFF_GRAPH), (1.0, 1.0, 1.0), vessel_mask=_segmentation(OFF_GRAPH_SEGMENT)
    )
    assert block.overlap_fraction == 1.0
    assert block.reason == "no_vessel" and blocked == []
    assert summary["mask_blocks_off_graph"] == 1


def test_a_thick_vessel_bridge_is_never_blocked():
    """A zero-resistance bridge is no vessel to plug: a blob on one is off
    every vessel of the graph."""
    from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE

    G = _graph()
    u, v, _o, _d = VESSELS["top"]
    G[u][v][0][IS_ZERO_RESISTANCE] = True
    blocked, _summary, (block,) = blocks_from_mask(
        G, _blobs(ON_TOP), (1.0, 1.0, 1.0), vessel_mask=_segmentation()
    )
    assert blocked == [] and block.reason == "no_vessel"


def test_an_ilastik_labelled_segmentation_reads_as_the_binary_one():
    """ilastik exports 1 (vessel, the minority) and 2 (background)."""
    G = _graph()
    seg = _segmentation()
    labelled = np.where(seg > 0, 1, 2).astype(np.uint8)
    mask = _blobs(ON_TOP, IN_TISSUE, ACROSS_BOTTOM_WALL, AT_JUNCTION)

    binary = blocks_from_mask(G, mask, (1.0, 1.0, 1.0), vessel_mask=seg, min_overlap_fraction=0.3)
    ilastik = blocks_from_mask(G, mask, (1.0, 1.0, 1.0), vessel_mask=labelled, min_overlap_fraction=0.3)

    assert ilastik[0] == binary[0]
    assert [b.overlap_fraction for b in ilastik[2]] == [b.overlap_fraction for b in binary[2]]


def test_without_a_segmentation_the_graphs_lumens_stand_in_for_it():
    G = _graph()
    mask = _blobs(ON_TOP, IN_TISSUE, ACROSS_BOTTOM_WALL, AT_JUNCTION)

    with_seg = blocks_from_mask(G, mask, (1.0, 1.0, 1.0), vessel_mask=_segmentation(), min_overlap_fraction=0.3)
    with_lumens = blocks_from_mask(G, mask, (1.0, 1.0, 1.0), vessel_mask=None, min_overlap_fraction=0.3)

    assert with_lumens[1]["vessel_mask_source"] == "graph_lumens"
    assert with_lumens[0] == with_seg[0]
    assert [b.voxels_in_vessels for b in with_lumens[2]] == [b.voxels_in_vessels for b in with_seg[2]]


def test_blocks_are_placed_by_the_voxel_size():
    """A 2 um z step: a blob at voxel z = 2 is at 4 um, on the top vessel; read
    at 1 um it would be 2 um off it, in the tissue under it."""
    G = _graph()
    shape = (4,) + SHAPE[1:]
    seg = np.zeros(shape, dtype=np.uint8)
    seg[2, 3:6, 10:41] = 1
    mask = np.zeros(shape, dtype=bool)
    mask[2, 4, 24:27] = True

    blocked, _s, _b = blocks_from_mask(G, mask, (2.0, 1.0, 1.0), vessel_mask=seg)

    assert blocked == [list(G.edges(keys=True))[_ids(G)["top"]]]


def test_a_mask_not_on_the_segmentations_grid_is_refused():
    with pytest.raises(ValueError, match="same grid"):
        blocks_from_mask(
            _graph(), np.zeros((2, 2, 2), dtype=bool), (1.0, 1.0, 1.0), vessel_mask=_segmentation()
        )


def test_an_unknown_vessels_per_block_choice_is_refused():
    with pytest.raises(ValueError, match="vessels_per_block"):
        blocks_from_mask(_graph(), _blobs(ON_TOP), (1.0, 1.0, 1.0), vessels_per_block="nearest")


def test_an_empty_mask_blocks_nothing():
    blocked, summary, blocks = blocks_from_mask(
        _graph(), np.zeros(SHAPE, dtype=bool), (1.0, 1.0, 1.0), vessel_mask=_segmentation()
    )
    assert blocked == [] and blocks == [] and summary["mask_blocks"] == 0


# --- loading the mask -------------------------------------------------------------------


def test_the_mask_loads_as_a_boolean_volume(tmp_path):
    path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(path), _blobs(ON_TOP).astype(np.uint8) * 255)

    mask, voxel_size = load_capillary_block_mask(path, _graph())

    assert mask.dtype == bool and mask.shape == SHAPE
    assert int(mask.sum()) == len(ON_TOP)
    assert voxel_size == (1.0, 1.0, 1.0)


def test_a_mask_whose_voxel_size_is_not_the_images_is_refused(tmp_path):
    path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(path), _blobs(ON_TOP).astype(np.uint8))
    G = _graph()
    G.graph["image_voxel_size_zyx"] = (2.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="voxel size"):
        load_capillary_block_mask(path, G)


# --- the perturbation ----------------------------------------------------------------


def _settings(tmp_path: Path, entries, **extra):
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "run_haemodynamics": True,
            "run_perturbations": True,
            "perturbations": entries,
            "diameter_by_branch_order": {"Art1": 6.0, "BO2": 4.0, "Ven1": 6.0},
            "measure_diameters_from_raw_image": False,
            "viscosity_law": "constant",
            "inlet_p_bc": 100.0,
            "outlet_p_bc": 0.0,
        }
    )
    values.update(extra)
    return values


def _boundaries():
    return BoundaryNodes(inlet_nodes=[0], outlet_nodes=[5], resistance_node_pair=(0, 5))


def _network(seg):
    return SimpleNamespace(volume=SimpleNamespace(image=seg, voxel_size_zyx=(1.0, 1.0, 1.0)))


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _entry(mask_path: Path, **overrides):
    return {
        "name": "stall",
        "type": "capillary_block_mask",
        "overrides": {"capillary_block_mask_path": str(mask_path), **overrides},
    }


def test_capillary_block_mask_is_a_perturbation_type_with_its_own_options():
    assert "capillary_block_mask" in PERTURBATION_TYPES
    assert CAPILLARY_BLOCK_TYPES == {"capillary_block", "capillary_block_mask"}
    assert is_capillary_block("capillary_block_mask") and not is_capillary_block("pressure_sweep")
    options = settings_for_perturbation_type("capillary_block_mask")
    assert {
        "capillary_block_mask_path",
        "capillary_block_mask_min_overlap_fraction",
        "capillary_block_mask_vessels_per_block",
        "capillary_block_resistance_factor",
        "capillary_block_hypoperfusion_fraction",
    } <= set(options)
    # It chooses vessels by its mask alone.
    assert "capillary_block_selection" not in options
    assert all(name in SCHEMA for name in options)


def test_the_perturbation_blocks_the_vessels_under_the_masks_blobs(tmp_path):
    mask_path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(mask_path), _blobs(ON_TOP, IN_TISSUE).astype(np.uint8))
    model = HaemodynamicModel(graph=_graph())
    ids = _ids(model.graph)
    before = {(u, v, k): d["resistance"] for u, v, k, d in model.graph.edges(keys=True, data=True)}

    run = run_perturbations(
        _settings(tmp_path, [_entry(mask_path)]),
        model,
        _boundaries(),
        SCHEMA,
        network=_network(_segmentation()),
    )

    (result,) = run.results
    assert result.error is None, result.error
    written = {path.name for path in result.outputs}
    assert {"stall_mask_blocks.csv", "stall_blocked_vessels.csv", "stall_block_comparison.csv",
            "stall_flow_by_branch_order.csv", "stall_summary.csv"} <= written
    assert result.output_dir.name == "stall_capillary_block_mask"

    blobs = _read(result.output_dir / "stall_mask_blocks.csv")
    assert sorted((row["voxels"], row["blocks"], row["reason"]) for row in blobs) == [
        ("27", "True", ""),
        ("3", "False", "below_overlap"),
    ]

    (vessel,) = _read(result.output_dir / "stall_blocked_vessels.csv")
    assert int(vessel["branch_id"]) == ids["top"]
    assert float(vessel["blocked_resistance"]) == pytest.approx(1e6 * float(vessel["baseline_resistance"]))

    comparison = {row["metric"]: row["value"] for row in _read(result.output_dir / "stall_block_comparison.csv")}
    assert comparison["selection"] == "mask"
    assert comparison["mask_blocks"] == "2"
    assert comparison["mask_blocks_on_vessels"] == "1"
    assert comparison["vessel_mask_source"] == "segmentation"
    assert comparison["blocked_vessels"] == "1"
    assert float(comparison["total_inflow_percent_change"]) < 0
    # The top route's other vessel is fed only through the block.
    assert int(comparison["stopped_vessels"]) == 2

    u, v, _o, _d = VESSELS["top"]
    w, x, _o, _d = VESSELS["right"]
    G = result.graph
    assert G[u][v][0][BLOCK_EDGE_ATTRIBUTE] == "blocked"
    assert G[u][v][0]["flow_abs"] == 0.0 and G[w][x][0]["flow_abs"] == 0.0
    a, b, _o, _d = VESSELS["bottom"]
    assert G[a][b][0][BLOCK_EDGE_ATTRIBUTE] == "open" and G[a][b][0]["flow_abs"] > 0.0
    # The baseline the run goes on to export is as it arrived.
    assert {(u, v, k): d["resistance"] for u, v, k, d in model.graph.edges(keys=True, data=True)} == before
    assert all(BLOCK_EDGE_ATTRIBUTE not in d for *_e, d in model.graph.edges(keys=True, data=True))


def test_the_perturbations_overlap_and_vessel_choice_reach_the_blocks(tmp_path):
    mask_path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(mask_path), _blobs(ACROSS_BOTTOM_WALL, AT_JUNCTION).astype(np.uint8))
    model = HaemodynamicModel(graph=_graph())
    ids = _ids(model.graph)
    entry = _entry(
        mask_path,
        capillary_block_mask_min_overlap_fraction=0.3,
        capillary_block_mask_vessels_per_block="every_overlapped_vessel",
    )

    run = run_perturbations(
        _settings(tmp_path, [entry]), model, _boundaries(), SCHEMA, network=_network(_segmentation())
    )

    (result,) = run.results
    assert result.error is None, result.error
    blocked = {int(row["branch_id"]) for row in _read(result.output_dir / "stall_blocked_vessels.csv")}
    assert blocked == {ids["bottom"], ids["right"]}


def test_without_the_runs_segmentation_the_perturbation_reads_the_graphs_lumens(tmp_path, caplog):
    mask_path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(mask_path), _blobs(ON_TOP).astype(np.uint8))

    with caplog.at_level("WARNING"):
        run = run_perturbations(
            _settings(tmp_path, [_entry(mask_path)]), HaemodynamicModel(graph=_graph()), _boundaries(), SCHEMA
        )

    (result,) = run.results
    assert result.error is None, result.error
    comparison = {row["metric"]: row["value"] for row in _read(result.output_dir / "stall_block_comparison.csv")}
    assert comparison["vessel_mask_source"] == "graph_lumens"
    assert comparison["blocked_vessels"] == "1"
    assert "graph's own lumens" in caplog.text


def test_a_mask_with_no_blob_on_a_vessel_leaves_the_flows_as_the_baseline(tmp_path):
    mask_path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(mask_path), _blobs(IN_TISSUE).astype(np.uint8))

    run = run_perturbations(
        _settings(tmp_path, [_entry(mask_path)]),
        HaemodynamicModel(graph=_graph()),
        _boundaries(),
        SCHEMA,
        network=_network(_segmentation()),
    )

    (result,) = run.results
    assert result.error is None, result.error
    comparison = {row["metric"]: row["value"] for row in _read(result.output_dir / "stall_block_comparison.csv")}
    assert comparison["blocked_vessels"] == "0"
    assert math.isclose(float(comparison["total_inflow_percent_change"]), 0.0, abs_tol=1e-9)


# --- checks ----------------------------------------------------------------------------


def _values(entries):
    values = {setting.name: setting.default for setting in SCHEMA}
    values["run_perturbations"] = True
    values["perturbations"] = entries
    return values


def test_preflight_reports_a_mask_block_that_names_no_mask():
    problems = perturbation_problems(
        _values([{"name": "stall", "type": "capillary_block_mask", "overrides": {}}]), SCHEMA
    )
    assert any("names no mask" in line for line in problems), problems


def test_preflight_is_quiet_about_a_mask_block_naming_one(tmp_path):
    path = tmp_path / "blocks.tif"
    tifffile.imwrite(str(path), _blobs(ON_TOP).astype(np.uint8))
    assert perturbation_problems(_values([_entry(path)]), SCHEMA) == ()


def test_preflight_reports_a_mask_that_is_not_there(tmp_path):
    from haemolynx.pipeline.checks import check_perturbations

    report = check_perturbations(_values([_entry(tmp_path / "missing.tif")]), SCHEMA)

    assert any("capillary_block_mask_path" in line for line in report.errors), report.errors


def test_the_unset_mask_setting_is_not_demanded_of_every_run():
    from haemolynx.parsers import check_settings

    report = check_settings(SCHEMA, _values([]))
    assert not any("capillary_block_mask_path" in line for line in report.errors)


def test_a_batch_refuses_one_mask_of_blocks_for_every_image(tmp_path):
    from haemolynx.pipeline.batch import check_batch

    report = check_batch(_values([_entry(tmp_path / "blocks.tif")]), SCHEMA)
    assert any("one mask of blocks" in line for line in report.errors), report.errors


def test_the_red_cell_citation_counts_a_mask_block():
    from haemolynx.pipeline.citations import _used_red_cell_transport

    values = _values([{"name": "stall", "type": "capillary_block_mask"}])
    values["run_haemodynamics"] = True
    assert _used_red_cell_transport(values)


def test_the_panel_offers_the_type_and_its_rows():
    from haemolynx.gui.form import H5_DATASET_SETTINGS
    from haemolynx.gui.perturbation_editing import (
        display_label_for_setting,
        display_name_for_type,
        rows_for_type,
    )

    assert display_name_for_type("capillary_block_mask") == "capillary blocks by mask"
    assert rows_for_type("capillary_block_mask")[0] == "capillary_block_mask_path"
    assert display_label_for_setting("capillary_block_mask_path") == "Mask of blocks"
    assert H5_DATASET_SETTINGS["capillary_block_mask_h5_dataset_name"] == "capillary_block_mask_path"
