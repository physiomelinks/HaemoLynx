"""Where a perturbation's mask objects lie (haemodynamics.mask_distribution):
their density by branch order and per mm of capillary, and their depth in the
image -- on a network small enough to count by hand.

The network (1 um voxels, every vessel at y = 5):

    n0 --Art1-- n1 -------- top, B01 (z = 10) -------- n2
                 |                                       |
           left, B02 (x = 20)                    right, B01 (x = 60)
                 |                                       |
                n3 ------- bottom, B02 (z = 30) ------- n4 --Ven1-- n5

B01 and B02 are 60 um each, 120 um of capillary. The pericyte mask holds
seven pericytes: four placed (one on top, one on left, two on bottom) and
three that are not (one too far from any capillary, one too small, one beside
the arteriole alone).

Branch angles (each daughter's emergence from its unique lower-order parent):
at n1 the arteriole is the parent, so top carries straight on (0 degrees) and
left turns off it (90); at n4 right (B01) is the parent of bottom (B02) and
the venule, both at 90. Right and the arteriole are no junction's daughter.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import tifffile

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics import (
    PoiseuilleModel,
    kernel_density_mode,
    mask_distribution,
    pericyte_mask_distribution,
)
from haemolynx.haemodynamics.mask_distribution import DEPTH_STATISTICS, VESSEL_MEASURES, VESSEL_STATISTICS
from haemolynx.haemodynamics.pericyte_mask import (
    load_pericyte_mask,
    place_pericytes,
    set_poiseuille_resistances_with_pericyte_mask,
)
from haemolynx.pipeline import BoundaryNodes, HaemodynamicModel, default_schema, run_perturbations
from haemolynx.pipeline.export_dictionary import undocumented
from haemolynx.statistics.bifurcation import (
    branch_order_label,
    compute_emergence_angles_by_branch_order,
    emergence_angle_by_edge,
)

SCHEMA = default_schema()

SHAPE = (40, 20, 72)
NODES = {
    0: (10, 5, 0),
    1: (10, 5, 20),
    2: (10, 5, 60),
    3: (30, 5, 20),
    4: (30, 5, 60),
    5: (30, 5, 70),
}
VESSELS = {
    "art": (0, 1, "Art1"),
    "top": (1, 2, "B01"),
    "right": (2, 4, "B01"),
    "left": (1, 3, "B02"),
    "bottom": (3, 4, "B02"),
    "ven": (4, 5, "Ven1"),
}
DIAMETERS = {"Art1": 10.0, "B01": 5.0, "B02": 5.0, "Ven1": 10.0}
#: Each vessel's own measured diameter, so the widths the pericytes sit on differ.
VESSEL_DIAMETERS = {"art": 10.0, "top": 4.0, "right": 5.0, "left": 6.0, "bottom": 7.0, "ven": 10.0}
BRANCH_ANGLES = {"top": 0.0, "left": 90.0, "bottom": 90.0, "ven": 90.0}

#: Each pericyte as (centre, half width): a cube of 2 * half + 1 voxels a side.
#: A 5-voxel cube is 6.2 um across as a sphere of its volume, a pericyte's size.
PLACED = {
    "on_top": ((10, 7, 40), 2),
    "on_left": ((20, 7, 20), 2),
    "on_bottom_1": ((30, 7, 40), 2),
    "on_bottom_2": ((30, 7, 50), 2),
}
NOT_PLACED = {
    # 14 um from the nearest capillary, past the 3 um limit.
    "far": ((20, 15, 40), 2),
    # 2.6 um from top, but 2.5 um across: not a pericyte's size.
    "small": ((10, 7, 30), None),
    # Beside the arteriole: 12 um from the nearest capillary.
    "by_arteriole": ((10, 8, 8), 2),
}
PLACED_DEPTHS = [10.0, 20.0, 30.0, 30.0]
ALL_DEPTHS = PLACED_DEPTHS + [20.0, 10.5, 10.0]


def _graph() -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in NODES.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    for name, (u, v, order) in VESSELS.items():
        a, b = np.asarray(NODES[u], dtype=float), np.asarray(NODES[v], dtype=float)
        length = float(np.linalg.norm(b - a))
        steps = np.linspace(0.0, 1.0, int(length) + 1)[:, None]
        G.add_edge(
            u, v, branch_order=order, length=length, diameter_um=VESSEL_DIAMETERS[name],
            voxels=(a + steps * (b - a)).tolist(),
        )
    return G


def _at(*depths: float) -> list[tuple[float, float, float]]:
    """Centroids at these depths (z), all at y = x = 0."""
    return [(depth, 0.0, 0.0) for depth in depths]


def _pericyte_mask() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    for (z, y, x), half in {**PLACED, **NOT_PLACED}.values():
        if half is None:
            mask[z:z + 2, y:y + 2, x:x + 2] = True
        else:
            mask[z - half:z + half + 1, y - half:y + half + 1, x - half:x + half + 1] = True
    return mask


def _edge(G: nx.MultiGraph, name: str):
    u, v, _order = VESSELS[name]
    return next(edge for edge in G.edges(keys=True) if set(edge[:2]) == {u, v})


def _orders(distribution) -> dict[str, tuple]:
    return {
        row["branch_order"]: (row["vessels"], row["length_um"], row["objects"], row["objects_per_mm"])
        for row in distribution.by_branch_order
    }


# --- the kernel density's mode --------------------------------------------------------


def test_one_value_or_values_all_alike_are_their_own_mode():
    assert kernel_density_mode([12.5]) == (12.5, 0.0)
    assert kernel_density_mode([7.0, 7.0, 7.0]) == (7.0, 0.0)
    mode, bandwidth = kernel_density_mode([])
    assert math.isnan(mode) and math.isnan(bandwidth)
    assert kernel_density_mode([np.nan, 3.0]) == (3.0, 0.0)


def test_the_bandwidth_is_silvermans_rule():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    # SD sqrt(2.5), IQR 4 - 2 = 2: the smaller is IQR / 1.34.
    expected = 0.9 * min(math.sqrt(2.5), 2.0 / 1.34) * 5 ** -0.2

    mode, bandwidth = kernel_density_mode(values)

    assert bandwidth == pytest.approx(expected)
    assert mode == pytest.approx(3.0)


def test_an_iqr_of_zero_falls_back_to_the_sd():
    values = [10.0] * 10 + [50.0] * 3
    expected = 0.9 * float(np.std(values, ddof=1)) * 13 ** -0.2

    mode, bandwidth = kernel_density_mode(values)

    assert bandwidth == pytest.approx(expected)
    assert mode == 10.0


@pytest.mark.parametrize(
    "values",
    [
        [10.0, 20.0, 30.0, 30.0],
        [4.0, 4.5, 5.0, 9.0, 9.2, 9.4, 9.6, 30.0],
        list(np.random.default_rng(3).normal(50.0, 12.0, 400)) + [5.0, 95.0],
    ],
)
def test_the_mode_is_the_densitys_peak(values):
    """Within the search grid's step (a tenth of the bandwidth) of the peak a
    much finer grid finds."""
    mode, bandwidth = kernel_density_mode(values)

    data = np.asarray(values)
    fine = np.linspace(data.min(), data.max(), 200_001)
    density = np.exp(-0.5 * ((fine[:, None] - data[None, :]) / bandwidth) ** 2).sum(axis=1)
    assert abs(mode - fine[np.argmax(density)]) <= 0.1 * bandwidth


def test_the_mode_is_the_larger_clusters():
    mode, _bandwidth = kernel_density_mode([5.0, 5.2, 4.8, 5.1, 40.0, 40.3])
    assert mode == pytest.approx(5.0, abs=0.3)


# --- density and depth, from objects already placed ----------------------------------


def test_objects_are_counted_once_per_vessel_order_and_per_mm():
    G = _graph()
    top, left, bottom, art = (_edge(G, name) for name in ("top", "left", "bottom", "art"))

    distribution = mask_distribution(
        G,
        [top, left, bottom, bottom, art, None],
        _at(10.0, 20.0, 30.0, 30.0, 10.0, 25.0),
        image_depth_um=39.0,
        mask="test",
    )

    assert _orders(distribution) == {
        "Art1": (1, 20.0, 1, pytest.approx(50.0)),
        "BO1": (2, 60.0, 1, pytest.approx(1000.0 / 60.0)),
        "BO2": (2, 60.0, 3, pytest.approx(50.0)),
        "Ven1": (1, 10.0, 0, 0.0),
    }
    assert [row["branch_order"] for row in distribution.by_branch_order] == ["Art1", "BO1", "BO2", "Ven1"]
    summary = distribution.summary
    assert summary["mask"] == "test"
    assert summary["objects"] == 6
    assert summary["objects_on_vessels"] == 5
    assert summary["objects_on_capillaries"] == 4
    assert summary["capillary_vessels"] == 4
    assert summary["capillary_length_um"] == 120.0
    assert summary["objects_per_mm_capillary"] == pytest.approx(4 / 120.0 * 1000.0)
    assert summary["image_depth_um"] == 39.0
    assert summary["depth_min_um_all"] == 10.0
    assert summary["depth_max_um_all"] == 30.0
    assert summary["depth_range_um_all"] == 20.0
    assert summary["depth_mean_um_all"] == pytest.approx(125.0 / 6)
    # The object on no vessel is in the mask's depths, not the placed ones'.
    assert summary["depth_mean_um_on_vessels"] == pytest.approx(100.0 / 5)
    assert summary["depth_mode_um_on_vessels"] == kernel_density_mode([10.0, 20.0, 30.0, 30.0, 10.0])[0]
    assert summary["depth_mode_bandwidth_um_all"] == pytest.approx(
        kernel_density_mode([10.0, 20.0, 30.0, 30.0, 10.0, 25.0])[1]
    )


def test_a_bridge_is_no_vessel_and_an_unordered_vessel_is_listed_unassigned():
    G = _graph()
    G.add_node(6, pos=np.array([30.0, 5.0, 66.0]))
    G.add_edge(4, 6, length=6.0, voxels=[[30.0, 5.0, 60.0], [30.0, 5.0, 66.0]], **{IS_ZERO_RESISTANCE: True})
    G.add_node(7, pos=np.array([30.0, 5.0, 80.0]))
    G.add_edge(5, 7, length=10.0, voxels=[[30.0, 5.0, 70.0], [30.0, 5.0, 80.0]])

    distribution = mask_distribution(G, [], [], image_depth_um=39.0)

    orders = _orders(distribution)
    assert orders["unassigned"] == (1, 10.0, 0, 0.0)
    assert sum(row["vessels"] for row in distribution.by_branch_order) == 7
    assert distribution.summary["capillary_length_um"] == 120.0


def test_capillaries_only_lists_the_capillary_orders():
    distribution = mask_distribution(_graph(), [], [], image_depth_um=39.0, capillaries_only=True)
    assert list(_orders(distribution)) == ["BO1", "BO2"]


def test_an_empty_mask_has_no_depth_and_no_objects_per_mm():
    summary = mask_distribution(_graph(), [], [], image_depth_um=39.0).summary
    assert summary["objects"] == 0 and summary["objects_per_mm_capillary"] == 0.0
    for group in ("all", "on_vessels"):
        for name in DEPTH_STATISTICS:
            assert math.isnan(summary[f"depth_{name}_um_{group}"])


def test_a_network_without_capillaries_has_no_capillary_density():
    G = nx.MultiGraph()
    G.add_edge(0, 1, branch_order="Art1", length=20.0)
    summary = mask_distribution(G, [(0, 1, 0)], _at(5.0), image_depth_um=10.0).summary
    assert summary["capillary_length_um"] == 0.0
    assert math.isnan(summary["objects_per_mm_capillary"])


def test_a_placement_and_a_volume_per_centroid_are_required():
    with pytest.raises(ValueError, match="one each per object"):
        mask_distribution(_graph(), [None], [], image_depth_um=1.0)
    with pytest.raises(ValueError, match="one each per object"):
        mask_distribution(_graph(), [None], _at(1.0), image_depth_um=1.0, volumes_um3=[])


# --- the vessel each object sits on ---------------------------------------------------


def test_the_branch_angle_is_each_vessels_emergence_angle_as_the_branch_statistics_average_it():
    G = _graph()

    angles = emergence_angle_by_edge(G)
    by_order = compute_emergence_angles_by_branch_order(G)

    expected = {(frozenset(_edge(G, name)[:2]), 0): angle for name, angle in BRANCH_ANGLES.items()}
    assert angles == pytest.approx(expected)
    for tag, record in by_order.items():
        mine = [angle for name, angle in BRANCH_ANGLES.items()
                if branch_order_label(VESSELS[name][2]) == tag]
        assert record["Emergence Angle Sample Count"] == len(mine)
        assert record["Mean Emergence Angle (degrees)"] == pytest.approx(sum(mine) / len(mine))


def test_each_object_is_listed_with_its_vessels_length_width_and_angle():
    G = _graph()
    top, left, bottom = (_edge(G, name) for name in ("top", "left", "bottom"))

    distribution = mask_distribution(
        G, [top, left, bottom, bottom, None], _at(10.0, 20.0, 30.0, 30.0, 25.0),
        image_depth_um=39.0, volumes_um3=[125.0, 125.0, 125.0, 125.0, 8.0],
    )

    rows = distribution.objects
    assert [row["object"] for row in rows] == [1, 2, 3, 4, 5]
    edges = list(G.edges(keys=True))
    assert [
        (row["branch_id"], row["branch_order"], row["capillary"], row["vessel_length_um"],
         row["vessel_diameter_um"], row["vessel_branch_angle_deg"])
        for row in rows[:4]
    ] == [
        (edges.index(top), "BO1", True, 40.0, 4.0, pytest.approx(0.0)),
        (edges.index(left), "BO2", True, 20.0, 6.0, pytest.approx(90.0)),
        (edges.index(bottom), "BO2", True, 40.0, 7.0, pytest.approx(90.0)),
        (edges.index(bottom), "BO2", True, 40.0, 7.0, pytest.approx(90.0)),
    ]
    assert rows[0]["centroid_z_um"] == 10.0 and rows[4]["volume_um3"] == 8.0
    assert rows[4]["on_vessel"] is False and "branch_id" not in rows[4]
    summary = distribution.summary
    # Lengths 40, 20, 40, 40; diameters 4, 6, 7, 7; angles 0, 90, 90, 90.
    assert summary["vessel_length_n"] == 4
    assert summary["vessel_length_mean_um"] == 35.0
    assert summary["vessel_length_sd_um"] == pytest.approx(10.0)
    assert summary["vessel_length_median_um"] == 40.0
    assert (summary["vessel_length_min_um"], summary["vessel_length_max_um"]) == (20.0, 40.0)
    assert summary["vessel_diameter_mean_um"] == 6.0
    assert summary["vessel_diameter_sd_um"] == pytest.approx(math.sqrt(2.0))
    assert summary["vessel_diameter_median_um"] == 6.5
    assert summary["vessel_branch_angle_n"] == 4
    assert summary["vessel_branch_angle_mean_deg"] == pytest.approx(67.5)
    assert summary["vessel_branch_angle_sd_deg"] == pytest.approx(45.0)
    assert summary["vessel_branch_angle_median_deg"] == pytest.approx(90.0)


def test_a_vessel_with_no_angle_or_width_leaves_it_blank_and_out_of_the_summary():
    """Right is no junction's daughter; a vessel with no diameter has no width."""
    G = _graph()
    right, top = _edge(G, "right"), _edge(G, "top")
    del G.edges[top]["diameter_um"]

    distribution = mask_distribution(G, [right, top], _at(10.0, 20.0), image_depth_um=39.0)

    first, second = distribution.objects
    assert first["vessel_branch_angle_deg"] is None and first["vessel_diameter_um"] == 5.0
    assert second["vessel_diameter_um"] is None
    summary = distribution.summary
    assert summary["vessel_branch_angle_n"] == 1 and summary["vessel_diameter_n"] == 1
    assert summary["vessel_diameter_mean_um"] == 5.0
    # One value has no SD.
    assert math.isnan(summary["vessel_diameter_sd_um"])
    assert summary["vessel_length_n"] == 2


def test_no_object_on_a_vessel_has_no_vessel_statistics():
    summary = mask_distribution(_graph(), [None], _at(5.0), image_depth_um=39.0).summary
    for measure, unit in VESSEL_MEASURES:
        assert summary[f"vessel_{measure}_n"] == 0
        for name in VESSEL_STATISTICS:
            assert math.isnan(summary[f"vessel_{measure}_{name}_{unit}"])


# --- a pericyte mask ----------------------------------------------------------------


def test_a_pericyte_masks_density_counts_the_pericytes_placed_on_capillaries():
    G = _graph()

    placement = place_pericytes(G, _pericyte_mask(), (1.0, 1.0, 1.0))
    distribution = pericyte_mask_distribution(G, placement, image_shape=SHAPE, voxel_size_zyx=(1.0, 1.0, 1.0))

    assert len(placement.eligible) == 4
    assert {placement.projections[i][0] for i in placement.eligible} == {
        _edge(G, "top"), _edge(G, "left"), _edge(G, "bottom")
    }
    # Only the capillary orders: no pericyte is ever placed on another.
    assert _orders(distribution) == {
        "BO1": (2, 60.0, 1, pytest.approx(1000.0 / 60.0)),
        "BO2": (2, 60.0, 3, pytest.approx(50.0)),
    }
    summary = distribution.summary
    assert summary["mask"] == "pericyte_mask"
    assert (summary["objects"], summary["objects_on_vessels"], summary["objects_on_capillaries"]) == (7, 4, 4)
    assert summary["objects_per_mm_capillary"] == pytest.approx(4 / 120.0 * 1000.0)
    assert summary["image_depth_um"] == 39.0
    assert summary["depth_min_um_all"] == 10.0
    assert summary["depth_max_um_all"] == 30.0
    assert summary["depth_range_um_all"] == 20.0
    assert summary["depth_mean_um_all"] == pytest.approx(sum(ALL_DEPTHS) / 7)
    assert summary["depth_mean_um_on_vessels"] == pytest.approx(22.5)
    # As kernel_density_mode finds them, to its grid step: the pericytes come
    # in component order, and the sums in another order round differently.
    for group, depths in (("all", ALL_DEPTHS), ("on_vessels", PLACED_DEPTHS)):
        mode, bandwidth = kernel_density_mode(depths)
        assert summary[f"depth_mode_bandwidth_um_{group}"] == pytest.approx(bandwidth)
        assert summary[f"depth_mode_um_{group}"] == pytest.approx(mode, abs=0.1 * bandwidth)
    # Two of the four placed lie at 30 um: the peak is there.
    assert summary["depth_mode_um_on_vessels"] > 25.0
    # Each pericyte with its vessel: top, left and bottom twice.
    assert len(distribution.objects) == 7
    placed = sorted(
        (row["centroid_z_um"], row["branch_order"], row["vessel_length_um"], row["vessel_diameter_um"],
         row["vessel_branch_angle_deg"])
        for row in distribution.objects if row["on_vessel"]
    )
    assert placed == [
        (10.0, "BO1", 40.0, 4.0, pytest.approx(0.0)),
        (20.0, "BO2", 20.0, 6.0, pytest.approx(90.0)),
        (30.0, "BO2", 40.0, 7.0, pytest.approx(90.0)),
        (30.0, "BO2", 40.0, 7.0, pytest.approx(90.0)),
    ]
    cubes = [row["volume_um3"] for row in distribution.objects if row["on_vessel"]]
    assert cubes == pytest.approx([125.0] * 4)
    assert summary["vessel_length_mean_um"] == 35.0
    assert summary["vessel_diameter_mean_um"] == 6.0
    assert summary["vessel_branch_angle_mean_deg"] == pytest.approx(67.5)


def test_the_strategy_places_the_pericytes_the_density_counts(tmp_path):
    """One placement: the constriction strategy's own count of the pericytes
    it placed is the density's."""
    path = tmp_path / "pericytes.tif"
    tifffile.imwrite(str(path), _pericyte_mask().astype(np.uint8))
    G = _graph()

    _G, results = set_poiseuille_resistances_with_pericyte_mask(
        G,
        diameter_by_branch_order=DIAMETERS,
        constriction_factor_by_branch_order={order: 0.8 for order in DIAMETERS},
        pericyte_mask_path=path,
        constriction_length=8.0,
    )

    assert results["pericyte_count"] == 7
    assert results["eligible_pericyte_count"] == 4
    assert results["edges_with_pericytes"] == 3


def test_a_pericytes_depth_is_read_at_the_images_z_spacing(tmp_path):
    """A 2 um z step: a pericyte at plane 10 lies 20 um deep in an image of 16
    planes, 30 um deep."""
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (2.0, 1.0, 1.0)
    G.add_edge(0, 1, branch_order="B01", length=40.0,
               voxels=[[20.0, 5.0, float(x)] for x in range(0, 41)])
    mask = np.zeros((16, 12, 44), dtype=np.uint8)
    mask[9:12, 5:10, 18:23] = 1
    path = tmp_path / "pericytes.tif"
    tifffile.imwrite(str(path), mask)

    loaded, voxel_size_xyz, voxel_size_zyx = load_pericyte_mask(
        path, G, voxel_size_override_xyz=(1.0, 1.0, 2.0), voxel_size_policy="override"
    )
    placement = place_pericytes(G, loaded, voxel_size_zyx)
    summary = pericyte_mask_distribution(
        G, placement, image_shape=loaded.shape, voxel_size_zyx=voxel_size_zyx
    ).summary

    assert voxel_size_xyz == (1.0, 1.0, 2.0) and voxel_size_zyx == (2.0, 1.0, 1.0)
    assert summary["objects_on_vessels"] == 1
    assert summary["depth_mean_um_on_vessels"] == 20.0
    assert summary["image_depth_um"] == 30.0
    assert summary["objects_per_mm_capillary"] == pytest.approx(25.0)


# --- the perturbation ----------------------------------------------------------------


def _model() -> HaemodynamicModel:
    graph, _results = PoiseuilleModel(
        constriction_length=40.0, constriction_spacing=100.0, viscosity_law="constant"
    ).set_poiseuille_resistances(_graph(), DIAMETERS)
    return HaemodynamicModel(graph=graph)


def _boundaries() -> BoundaryNodes:
    return BoundaryNodes(inlet_nodes=[0], outlet_nodes=[5], resistance_node_pair=(0, 5))


def _settings(tmp_path: Path, entries) -> dict:
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "input_path": tmp_path / "input.tif",
            "vtk_output_prefix": tmp_path / "out" / "run",
            "plot_dir": tmp_path / "plots",
            "run_haemodynamics": True,
            "run_perturbations": True,
            "perturbations": entries,
            "diameter_by_branch_order": dict(DIAMETERS),
            "measure_diameters_from_raw_image": False,
            "viscosity_law": "constant",
            "pericyte_dilation_min_percent": 1,
            "pericyte_dilation_max_percent": 2,
            "pericyte_dilation_step_percent": 1,
        }
    )
    return values


def _entry(perturbation_type: str, mask_path: Path | None) -> dict:
    overrides = {"pericyte_constriction_factor": 0.8}
    if mask_path is not None:
        overrides.update(use_pericyte_mask_constriction=True, pericyte_mask_path=str(mask_path))
    return {"name": "tone", "type": perturbation_type, "overrides": overrides}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run(tmp_path: Path, entry: dict):
    run = run_perturbations(_settings(tmp_path, [entry]), _model(), _boundaries(), SCHEMA)
    (result,) = run.results
    assert result.error is None, result.error
    return result


def _mask_path(tmp_path: Path) -> Path:
    path = tmp_path / "pericytes.tif"
    tifffile.imwrite(str(path), _pericyte_mask().astype(np.uint8))
    return path


def test_a_pericyte_perturbation_with_a_mask_writes_where_its_pericytes_lie(tmp_path):
    mask_path = _mask_path(tmp_path)
    model = _model()

    run = run_perturbations(
        _settings(tmp_path, [_entry("pericyte_diameter_change", mask_path)]), model, _boundaries(), SCHEMA
    )

    (result,) = run.results
    assert result.error is None, result.error
    written = {path.name for path in result.outputs}
    assert {
        "tone_mask_distribution.csv", "tone_mask_density_by_branch_order.csv", "tone_mask_objects.csv"
    } <= written
    objects = _read(result.output_dir / "tone_mask_objects.csv")
    assert len(objects) == 7
    placed = [row for row in objects if row["on_vessel"] == "True"]
    assert len(placed) == 4
    off = next(row for row in objects if row["on_vessel"] == "False")
    assert off["branch_id"] == "" and off["vessel_length_um"] == ""
    edges = list(model.graph.edges(keys=True, data=True))
    for row in placed:
        u, v, key, data = edges[int(row["branch_id"])]
        # The baseline's vessel: perturbing the pericytes' tone moves no vessel.
        assert (row["u"], row["v"], row["key"]) == (str(u), str(v), str(key))
        assert float(row["vessel_length_um"]) == data["length"]
        assert float(row["vessel_diameter_um"]) == data["diameter_um"]
    assert sorted(float(row["vessel_branch_angle_deg"]) for row in placed) == pytest.approx([0.0, 90.0, 90.0, 90.0])
    rows = {row["metric"]: row["value"] for row in _read(result.output_dir / "tone_mask_distribution.csv")}
    assert undocumented("{name}_mask_distribution.csv", list(rows)) == []
    assert rows["mask"] == "pericyte_mask"
    assert rows["mask_path"] == mask_path.as_posix()
    assert (rows["objects"], rows["objects_on_vessels"], rows["objects_on_capillaries"]) == ("7", "4", "4")
    assert float(rows["capillary_length_um"]) == 120.0
    assert float(rows["objects_per_mm_capillary"]) == pytest.approx(4 / 120.0 * 1000.0)
    assert float(rows["image_depth_um"]) == 39.0
    assert float(rows["depth_range_um_all"]) == 20.0
    assert float(rows["depth_mean_um_on_vessels"]) == pytest.approx(22.5)
    orders = _read(result.output_dir / "tone_mask_density_by_branch_order.csv")
    assert [(row["branch_order"], row["vessels"], row["objects"]) for row in orders] == [
        ("BO1", "2", "1"), ("BO2", "2", "3")
    ]
    assert float(orders[1]["objects_per_mm"]) == pytest.approx(50.0)
    # The strategy placed the pericytes the density counts.
    assert result.summary["eligible_pericyte_count"] == result.summary["mask_distribution"]["objects_on_vessels"]


def test_a_pericyte_sweep_with_a_mask_writes_them_after_its_grid(tmp_path):
    result = _run(tmp_path, _entry("pericyte_dilation_sweep", _mask_path(tmp_path)))

    names = [path.name for path in result.outputs]
    assert names[0].endswith("sweep.csv")
    assert {
        "tone_mask_distribution.csv", "tone_mask_density_by_branch_order.csv", "tone_mask_objects.csv"
    } <= set(names)
    assert result.summary["mask_distribution"]["objects_on_vessels"] == 4


def test_a_pericyte_perturbation_without_a_mask_writes_no_mask_distribution(tmp_path):
    result = _run(tmp_path, _entry("pericyte_diameter_change", None))

    assert not list(result.output_dir.glob("*mask_*.csv"))
    assert "mask_distribution" not in result.summary
