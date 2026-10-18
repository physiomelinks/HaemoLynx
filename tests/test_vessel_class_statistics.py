"""The statistics' "Vessel Classes" section: the same measures for each class
of vessel -- large and small arterioles, capillaries, small and large venules,
or with global vessel masks large vessels, small vessels and capillaries."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from haemolynx import statistics
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.pipeline import default_schema, stages
from haemolynx.statistics import (
    compute_vessel_class_statistics,
    compute_vessel_density,
    compute_vessel_statistics_by_measure,
)
from haemolynx.statistics.bifurcation import branch_order_category
from haemolynx.statistics.loops import compute_loop_hierarchy
from haemolynx.statistics.vessel_classes import VESSEL_CLASSES

#: Node positions, (z, y, x) microns: a path bending through all three axes,
#: so the box the nodes span has a volume (20 x 20 x 10).
POSITIONS = {
    0: (0.0, 0.0, 0.0),
    1: (10.0, 0.0, 0.0),
    2: (10.0, 10.0, 0.0),
    3: (10.0, 10.0, 10.0),
    4: (20.0, 10.0, 10.0),
    5: (20.0, 20.0, 10.0),
}
BOX_UM3 = 20.0 * 20.0 * 10.0
#: (u, v, branch order, length um, diameter um, pressure drop Pa, |flow| m^3/s)
VESSELS = (
    (0, 1, "Large_Art1", 10.0, 20.0, 2.0, 4e-14),
    (1, 2, "Art1", 12.0, 8.0, 10.0, 4e-14),
    (2, 3, "B01", 10.0, 4.0, 5.0, 1e-15),
    (2, 3, "B02", 14.0, 6.0, 5.0, 3e-15),
    (3, 4, "Ven1", 10.0, 10.0, 4.0, 4e-14),
    (4, 5, "Large_Ven1", 10.0, 25.0, 1.0, 4e-14),
)
HIERARCHICAL_TITLES = [
    "Large Arterioles",
    "Small Arterioles",
    "Capillaries",
    "Small Venules",
    "Large Venules",
]


def _graph(vessels=VESSELS, *, solved: bool = True, relabel=None) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for node, pos in POSITIONS.items():
        G.add_node(node, pos=np.asarray(pos))
    pressure = 100.0
    for u, v, label, length, diameter, drop, flow in vessels:
        data = {
            "length": length,
            "branch_order": relabel.get(label, label) if relabel else label,
            "diameter_um": diameter,
            "voxels": [POSITIONS[u], POSITIONS[v]],
        }
        if solved:
            data.update(
                flow_abs=flow,
                pressure_drop=drop,
                pressure_u=pressure,
                pressure_v=pressure - drop,
                flow_state="solved",
            )
        G.add_edge(u, v, **data)
    return G


def _classes(G, **kwargs) -> dict:
    return compute_vessel_class_statistics(G, POSITIONS, **kwargs)[VESSEL_CLASSES]


def _titles(rows: dict) -> list[str]:
    return [name for name, value in rows.items() if isinstance(value, dict)]


def test_hierarchical_classes_are_listed_in_the_order_blood_meets_them():
    rows = _classes(_graph())

    assert _titles(rows) == HIERARCHICAL_TITLES
    assert rows["Vessel Classes From"].startswith("branch orders")
    assert rows["Small Arterioles"]["Branch Orders"] == "Art1"
    assert rows["Capillaries"]["Branch Orders"] == "B01, B02"


def test_each_class_counts_its_own_vessels_and_their_length():
    rows = _classes(_graph())

    assert [rows[t]["Vessel Count"] for t in HIERARCHICAL_TITLES] == [1, 1, 2, 1, 1]
    capillaries = rows["Capillaries"]
    assert capillaries["Total Length (microns)"] == pytest.approx(24.0)
    assert capillaries["Mean Length (microns)"] == pytest.approx(12.0)
    assert capillaries["Median Length (microns)"] == pytest.approx(12.0)
    assert capillaries["Share of Network Vessel Length (fraction)"] == pytest.approx(24.0 / 66.0)
    assert sum(
        rows[t]["Share of Network Vessel Length (fraction)"] for t in HIERARCHICAL_TITLES
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("tissue_volume_um3", [None, 1.0e5])
def test_class_length_densities_add_up_to_the_network_density(tissue_volume_um3):
    """Each class divides by the volume "vessel_density" divides by: the
    measured tissue when there is one, else the box the nodes span."""
    G = _graph()
    rows = _classes(G, tissue_volume_um3=tissue_volume_um3)
    network = compute_vessel_density(
        G, POSITIONS, None, None, True, tissue_volume_um3=tissue_volume_um3
    )

    volume = BOX_UM3 if tissue_volume_um3 is None else tissue_volume_um3
    assert rows["Vessel Class Density Volume (micron³)"] == pytest.approx(volume)
    assert rows["Capillaries"]["Length Density in Tissue (microns/micron³)"] == pytest.approx(
        24.0 / volume
    )
    assert sum(
        rows[t]["Length Density in Tissue (microns/micron³)"] for t in HIERARCHICAL_TITLES
    ) == pytest.approx(network["Vessel Density in Tissue (microns/micron³)"])


def test_whole_image_density_divides_by_the_image_in_microns():
    rows = _classes(_graph(), voxel_size=(2.0, 1.0, 0.5), image_dimensions=(10, 40, 80))
    image_um3 = (10 * 2.0) * (40 * 1.0) * (80 * 0.5)

    assert rows["Capillaries"]["Length Density in Whole Image (microns/micron³)"] == pytest.approx(
        24.0 / image_um3
    )


def test_without_image_dimensions_there_is_no_whole_image_density():
    rows = _classes(_graph())
    assert "Length Density in Whole Image (microns/micron³)" not in rows["Capillaries"]


def test_volume_surface_diameter_and_tortuosity_are_each_classes_own():
    rows = _classes(_graph())
    capillaries = rows["Capillaries"]

    # A cylinder per vessel: pi r^2 L and pi d L.
    assert capillaries["Vessel Volume (micron³)"] == pytest.approx(math.pi * (4 * 10 + 9 * 14))
    assert capillaries["Vessel Surface Area (micron²)"] == pytest.approx(math.pi * (4 * 10 + 6 * 14))
    assert capillaries["Volume Density in Tissue (micron³/micron³)"] == pytest.approx(
        math.pi * (4 * 10 + 9 * 14) / BOX_UM3
    )
    assert capillaries["Surface Density in Tissue (micron²/micron³)"] == pytest.approx(
        math.pi * (4 * 10 + 6 * 14) / BOX_UM3
    )
    assert capillaries["Mean Diameter (microns)"] == pytest.approx(5.0)
    assert rows["Large Venules"]["Median Diameter (microns)"] == pytest.approx(25.0)
    # Art1 runs 12 um between nodes 10 um apart.
    assert rows["Small Arterioles"]["Mean Tortuosity Index"] == pytest.approx(1.2)
    assert rows["Small Arterioles"]["Median Tortuosity Index"] == pytest.approx(1.2)
    assert rows["Capillaries"]["Mean Tortuosity Index"] == pytest.approx((1.0 + 1.4) / 2)


def test_without_positions_tortuosity_says_why():
    rows = compute_vessel_class_statistics(_graph(), None)[VESSEL_CLASSES]
    assert rows["Capillaries"]["Mean Tortuosity Index"] == "N/A (no position data)"
    assert rows["Vessel Class Density Volume (micron³)"].startswith("N/A")
    assert rows["Capillaries"]["Length Density in Tissue (microns/micron³)"].startswith("N/A")


def test_haemodynamics_per_class_read_only_the_vessels_carrying_flow():
    """A vessel behind a dead end carries exactly no flow: counted as vessel,
    left out of the velocities it would only pull towards 0."""
    blind = (2, 3, "B03", 10.0, 4.0, 0.0, 0.0)
    G = _graph(VESSELS + (blind,))
    for _u, _v, data in G.edges(data=True):
        if data["branch_order"] == "B03":
            data["flow_state"] = "blind_end"
    capillaries = _classes(G)["Capillaries"]

    assert capillaries["Vessel Count"] == 3
    assert capillaries["Vessels Carrying Flow"] == 2
    velocities = [1e-15 / (math.pi * 2e-6**2) * 1e3, 3e-15 / (math.pi * 3e-6**2) * 1e3]
    assert capillaries["Mean Velocity (mm/s)"] == pytest.approx(np.mean(velocities))
    assert capillaries["Median Velocity (mm/s)"] == pytest.approx(np.median(velocities))
    # tau = dP d / (4 L): 5 * 4 / 40 and 5 * 6 / 56.
    shear = [5.0 * 4.0 / 40.0, 5.0 * 6.0 / 56.0]
    assert capillaries["Mean Wall Shear Stress (Pa)"] == pytest.approx(np.mean(shear))
    assert capillaries["Mean Flow (nL/min)"] == pytest.approx(np.mean([1e-15, 3e-15]) * 6e13)
    assert capillaries["Mean Pressure (Pa)"] == pytest.approx(100.0 - 2.5)


def test_pressure_falls_from_the_arterial_to_the_venous_classes():
    G = _graph()
    # Each vessel's own pressures, falling along the path.
    pressure = 100.0
    for u, v, data in sorted(G.edges(data=True), key=lambda e: (e[0], e[1])):
        data["pressure_u"], data["pressure_v"] = pressure, pressure - data["pressure_drop"]
        if data["branch_order"] != "B01":
            pressure -= data["pressure_drop"]
    rows = _classes(G)
    means = [rows[t]["Mean Pressure (Pa)"] for t in HIERARCHICAL_TITLES]
    assert means == sorted(means, reverse=True)


def test_without_a_solved_flow_there_are_no_haemodynamic_rows():
    rows = _classes(_graph(solved=False))

    assert rows["Vessel Class Haemodynamics"] == "N/A (no flow solved)"
    for title in HIERARCHICAL_TITLES:
        assert "Mean Velocity (mm/s)" not in rows[title]
        assert "Vessels Carrying Flow" not in rows[title]


def test_global_masks_report_large_vessels_small_vessels_and_capillaries():
    relabel = {
        "Large_Art1": "Unknown_Large",
        "Large_Ven1": "Unknown_Large",
        "Art1": "Unknown_Small",
        "Ven1": "Unknown_Small",
        "B01": "Unknown_Cap",
        "B02": "Unknown_Cap",
    }
    rows = _classes(_graph(relabel=relabel))

    assert _titles(rows) == ["Large Vessels", "Small Vessels", "Capillaries"]
    assert rows["Vessel Classes From"].startswith("global vessel masks")
    assert rows["Large Vessels"]["Vessel Count"] == 2
    assert rows["Large Vessels"]["Total Length (microns)"] == pytest.approx(20.0)
    assert rows["Large Vessels"]["Mean Diameter (microns)"] == pytest.approx(22.5)
    assert rows["Small Vessels"]["Mean Tortuosity Index"] == pytest.approx((1.2 + 1.0) / 2)
    assert rows["Capillaries"]["Branch Orders"] == "Unknown_Cap"


def test_a_global_run_with_only_capillaries_still_lists_its_three_classes():
    rows = _classes(_graph(relabel={label: "Unknown_Cap" for *_e, label, _l, _d, _p, _f in VESSELS}))

    assert _titles(rows) == ["Large Vessels", "Small Vessels", "Capillaries"]
    assert rows["Large Vessels"]["Vessel Count"] == 0
    assert rows["Capillaries"]["Vessel Count"] == len(VESSELS)


def test_a_class_with_no_vessels_still_has_its_rows():
    """Two runs labelled alike report the same rows, so a run with no
    large-vessel tier says it has no large arterioles."""
    rows = _classes(_graph(VESSELS[1:5]))
    large = rows["Large Arterioles"]

    assert _titles(rows) == HIERARCHICAL_TITLES
    assert large["Vessel Count"] == 0
    assert large["Total Length (microns)"] == 0.0
    assert large["Length Density in Tissue (microns/micron³)"] == 0.0
    assert large["Mean Length (microns)"] == "N/A (no vessels in this class)"
    assert large["Mean Velocity (mm/s)"] == "N/A (no vessel of this class carries flow)"
    assert large["Branch Orders"] == "none"


def test_orders_counted_from_the_inlets_alone_are_all_capillaries():
    rows = _classes(_graph(relabel={"Large_Art1": "B01", "Art1": "B02", "Ven1": "B04", "Large_Ven1": "B05"}))

    assert _titles(rows) == ["Capillaries"]
    assert rows["Vessel Classes From"].startswith("capillary branch orders")
    assert rows["Capillaries"]["Vessel Count"] == len(VESSELS)


def test_unlabelled_vessels_are_listed_last():
    G = _graph()
    for _u, _v, data in G.edges(data=True):
        if data["branch_order"] == "Large_Ven1":
            del data["branch_order"]
    rows = _classes(G)

    assert _titles(rows) == HIERARCHICAL_TITLES + ["Unassigned Vessels"]
    assert rows["Unassigned Vessels"]["Vessel Count"] == 1
    assert rows["Large Venules"]["Vessel Count"] == 0


def test_no_labels_at_all_is_one_unassigned_class():
    G = _graph()
    for _u, _v, data in G.edges(data=True):
        del data["branch_order"]
    rows = _classes(G)

    assert rows["Vessel Classes From"] == "no branch orders assigned"
    assert _titles(rows) == ["Unassigned Vessels"]


def test_a_zero_resistance_bridge_is_no_vessel():
    G = _graph()
    G.add_edge(1, 3, length=5.0, diameter_um=20.0, **{IS_ZERO_RESISTANCE: True})
    rows = _classes(G)

    assert sum(rows[t]["Vessel Count"] for t in HIERARCHICAL_TITLES) == len(VESSELS)
    assert "Unassigned Vessels" not in rows


def test_a_class_with_some_diameters_missing_has_no_volume():
    G = _graph()
    for _u, _v, data in G.edges(data=True):
        if data["branch_order"] == "B02":
            del data["diameter_um"]
    capillaries = _classes(G)["Capillaries"]

    assert capillaries["Vessel Volume (micron³)"] == "N/A (1 of 2 vessels have no diameter)"
    assert capillaries["Mean Diameter (microns)"] == pytest.approx(4.0)


def test_the_measure_is_a_report_section_after_vessel_density():
    sections = compute_vessel_statistics_by_measure(
        _graph(), node_positions=POSITIONS,
        enabled_measures=frozenset({"vessel_classes", "vessel_density", "basic"}),
    )

    assert list(sections) == ["basic", "vessel_density", "vessel_classes"]
    assert sections["vessel_classes"] == compute_vessel_class_statistics(_graph(), POSITIONS)
    assert "vessel_classes" in statistics.CONNECTED_NETWORK_MEASURES
    assert "vessel_classes" not in statistics.NETWORK_ANALYSIS_MEASURES


def test_global_classes_are_their_own_vessel_types_in_every_breakdown():
    """Large and small vessels of a global run are compared apart, not lumped
    together as unassigned, wherever a statistic breaks down by vessel type."""
    assert branch_order_category("Unknown_Large") == "large_vessel"
    assert branch_order_category("Unknown_Small") == "small_vessel"
    assert branch_order_category("Unknown_Cap") == "capillary"

    G = nx.MultiGraph()
    for node, pos in {0: (0, 0, 0), 1: (10, 0, 0), 2: (0, 10, 0)}.items():
        G.add_node(node, pos=np.asarray(pos, float))
    for u, v, label in ((0, 1, "Unknown_Large"), (1, 2, "Unknown_Small"), (2, 0, "Unknown_Cap")):
        G.add_edge(u, v, length=10.0, branch_order=label)
    by_type = compute_loop_hierarchy(G, statistics_mode="full")["Median Loop Length By Vessel Type"]
    assert set(by_type) == {"large_vessel", "small_vessel", "capillary"}


# --- the export stage ----------------------------------------------------------


def _export(tmp_path, **overrides) -> nx.MultiGraph:
    settings = default_schema().defaults()
    settings.update(
        input_path=tmp_path / "classes.tif",
        statistics=True,
        statistics_mode="fast",
        run_haemodynamics=False,
        measurement_3d_to_cell_mask=False,
        vtk_export=False,
        visualize_vtk=False,
        visualize_results=False,
    )
    settings.update(overrides)
    shape = (30, 30, 20)
    volume = stages.SkeletonisedVolume(
        image=np.zeros(shape, dtype=np.uint8),
        skeleton=np.zeros(shape, dtype=bool),
        voxel_size_xyz=(1.0, 1.0, 1.0),
        voxel_size_zyx=(1.0, 1.0, 1.0),
        output_dir=tmp_path,
    )
    G = _graph(solved=False)
    stages.export_results(
        settings,
        stages.VesselNetwork(graph=G, volume=volume),
        stages.HaemodynamicModel(graph=G),
        stages.Solution(),
    )
    return G


def test_export_writes_each_class_to_the_statistics_csv_and_the_statistics_tab(tmp_path):
    from haemolynx.gui.statistics_view import statistics_report, statistics_sections

    G = _export(tmp_path)
    csv_text = (tmp_path / "classes_statistics.csv").read_text(encoding="utf-8")

    for title in HIERARCHICAL_TITLES:
        assert f"Vessel Classes > {title},Vessel Count," in csv_text
    assert "Vessel Classes > Capillaries,Total Length,24,microns" in csv_text
    report = statistics_report(G)
    assert "vessel_classes" in report["sections"]
    section = next(s for s in statistics_sections(report) if s.key == "vessel_classes")
    assert section.title == "Vessel classes"
    assert any(row.heading and row.name == "Small Arterioles" for row in section.rows)


def test_the_setting_turns_the_section_off(tmp_path):
    _export(tmp_path, statistics_vessel_classes=False)
    csv_text = (tmp_path / "classes_statistics.csv").read_text(encoding="utf-8")
    assert "Vessel Classes" not in csv_text
