"""The template a synthetic network is matched to (``synthetic/template.py``)."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.gui.run_snapshot import RunSnapshotError
from haemolynx.pipeline import default_schema
from haemolynx.pipeline.stages import BRANCH_ORDER_SECTION, STATISTICS_REPORT
from haemolynx.statistics.comprehensive import STATISTIC_MEASURES
from haemolynx.synthetic import template as tpl
from haemolynx.synthetic.template import (
    BRANCH_ORDERS_SECTION,
    DEFAULT_DAUGHTER_ANGLE_DEG,
    DEFAULT_MURRAY_EXPONENT,
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
    SECTION_TOGGLES,
    STATISTICS_REPORT_KEY,
    Endpoint,
    NetworkTemplate,
    Samples,
    TemplateError,
    endpoints_from_nodes,
    load_template,
    lognormal_from_mean_cv,
    lognormal_from_mean_median,
    lognormal_from_median_p10_p90,
    required_statistics_for,
    template_from_report,
)
from synthetic_fixtures import (
    TINY_EXTENT_UM,
    TINY_ROLES,
    TINY_VOXEL_ZYX,
    fixture_template,
    tiny_expected,
    tiny_hierarchical_graph,
    tiny_sections,
    write_fixture_run,
)

SCHEMA = default_schema()


@pytest.fixture(autouse=True)
def _fresh_cache():
    tpl.clear_template_cache()
    yield
    tpl.clear_template_cache()


# -- the constants the stages and the schema also hold --------------------


def test_report_keys_are_the_stages_constants():
    assert STATISTICS_REPORT_KEY == STATISTICS_REPORT
    assert BRANCH_ORDERS_SECTION == BRANCH_ORDER_SECTION


def test_every_section_toggle_is_a_schema_setting():
    for section, toggles in SECTION_TOGGLES.items():
        for toggle in toggles:
            assert toggle in SCHEMA, f"{section}: {toggle}"
    for section in REQUIRED_SECTIONS + OPTIONAL_SECTIONS:
        assert section in SECTION_TOGGLES
        if section != BRANCH_ORDERS_SECTION:
            assert section in STATISTIC_MEASURES


def test_required_statistics_for_lists_each_toggle_once_in_order():
    assert required_statistics_for(["basic", "loop_hierarchy", "dead_ends", "cyclomatic_number"]) == (
        "statistics",
        "statistics_network_analysis",
        "statistics_loop_hierarchy",
        "statistics_dead_ends",
        "statistics_cyclomatic_number",
    )


# -- the two sources ------------------------------------------------------


def _without_samples(template: NetworkTemplate) -> dict:
    data = template.to_dict()
    for item in data["classes"].values():
        item.pop("samples")
    for item in data["orders"]:
        item.pop("samples")
    data["loops"].pop("loop_length_samples_um")
    data["dead_ends"].pop("chain_length_samples_um")
    data["provenance"].pop("source")
    return data


def test_both_sources_agree_on_the_tiny_run(tmp_path):
    path = write_fixture_run(tmp_path)
    by_statistics = load_template(path, "statistics")
    by_graph = load_template(path, "graph")
    assert by_statistics.provenance.source == "statistics"
    assert by_graph.provenance.source == "graph"
    assert _without_samples(by_statistics) == _without_samples(by_graph)
    # Only the graph source has the per-vessel values.
    assert by_statistics.classes["capillary"].samples is None
    assert len(by_graph.classes["capillary"].samples.lengths_um) == 5


def test_statistics_source_reads_the_tiny_runs_hand_computed_values(tmp_path):
    expected = tiny_expected()
    template = load_template(write_fixture_run(tmp_path), "statistics")
    assert template.extent_um == TINY_EXTENT_UM
    assert template.voxel_size_zyx == TINY_VOXEL_ZYX
    assert template.image_shape == (51, 201, 301)
    assert template.scheme == "hierarchical"
    assert template.total_length_um == pytest.approx(expected["total_length_um"])
    assert (template.node_count, template.edge_count) == (12, 12)
    assert template.branching_points == 4
    assert {o.label: o.edge_count for o in template.orders} == expected["order_counts"]
    assert {k: v.count for k, v in template.classes.items()} == expected["class_counts"]
    assert template.classes["capillary"].total_length_um == pytest.approx(expected["capillary_total_um"])
    assert template.classes["large_arteriole"].mean_diameter_um == 20.0
    assert template.loops.independent_loops == expected["cyclomatic"]
    assert template.dead_ends.total == 4
    assert template.dead_ends.leaves_image == 3
    assert template.dead_ends.interior == 1
    assert template.dead_ends.artefacts == 0
    assert template.dead_ends.chain_length_median_um == pytest.approx(expected["blind_chain_um"])
    assert [e.face for e in template.inlets] == [expected["inlet_face"]]
    assert [e.face for e in template.outlets] == [expected["outlet_face"]]
    assert template.inlets[0].position_rel == pytest.approx((0.5, 0.5, 0.0))
    assert (template.arteriole_boundary_count, template.venule_boundary_count) == (2, 1)
    assert (template.large_arteriole_boundary_count, template.large_venule_boundary_count) == (1, 1)
    assert template.diameter_basis == "plasma_column"  # the run's segmentation_source
    assert template.provenance.notes == ()


def test_graph_source_samples_carry_each_vessels_own_values(tmp_path):
    expected = tiny_expected()
    template = load_template(write_fixture_run(tmp_path), "graph")
    capillaries = template.classes["capillary"].samples
    assert sorted(capillaries.diameters_um) == [4.0, 4.0, 5.0, 5.0, 5.0]
    assert max(capillaries.tortuosities) == pytest.approx(expected["b02_tortuosity"])
    assert template.dead_ends.chain_length_samples_um == pytest.approx((expected["blind_chain_um"],))
    assert len(template.loops.loop_length_samples_um) == 6  # the six vessels on the one loop
    art1 = next(o for o in template.orders if o.label == "Art1")
    assert sorted(art1.samples.diameters_um) == [10.0, 12.0]


def test_orders_are_listed_along_the_flow_path(tmp_path):
    template = load_template(write_fixture_run(tmp_path), "statistics")
    assert [o.label for o in template.orders] == [
        "Large_Art1", "Art1", "Art2", "BO1", "BO2", "BO3", "Ven1", "Large_Ven1",
    ]
    large, small = template.orders_on("arterial")
    assert [o.label for o in large] == ["Large_Art1"]
    assert [o.label for o in small] == ["Art1", "Art2"]
    large, small = template.orders_on("venous")
    assert [o.label for o in large] == ["Large_Ven1"]
    assert [o.label for o in small] == ["Ven1"]


# -- the reader, row by row -------------------------------------------------


def _hand_sections() -> dict:
    sections = tiny_sections()
    sections["murray_law"]["Median Fitted Murray Exponent"] = 2.9
    sections["murray_law"]["Fitted Murray Exponent IQR Low"] = 2.6
    sections["murray_law"]["Fitted Murray Exponent IQR High"] = 3.2
    sections["daughter_angles"]["Mean Daughter-Daughter Angle (degrees)"] = 80.0
    sections["intercapillary_distance"]["Mean Intercapillary Distance (microns)"] = 25.0
    sections["intercapillary_distance"]["Median Intercapillary Distance (microns)"] = 24.0
    sections["tortuosity"]["Average Tortuosity Index"] = 1.21
    sections["loop_hierarchy"]["Loop Length Median (um)"] = 150.0
    sections["loop_hierarchy"]["Independent Loops Per mm^3"] = 1000.0
    sections["loop_hierarchy"]["Vessels On No Loop Fraction"] = 0.25
    sections["cyclomatic_number"]["Cyclomatic Number"] = 7
    sections["dead_ends"]["Dead Ends"]["Dead Ends That Are Artefacts"] = 2
    sections["dead_ends"]["Dead Ends"]["Interior Dead Ends, Blind or Broken (unknown)"] = 3
    sections["dead_ends"]["Dead Ends"]["Interior Dead Ends per mm of Vessel (1/mm)"] = 4.5
    sections["vessel_density"]["Tissue Volume (micron³)"] = 5.0e6
    return sections


def test_reader_maps_each_row_to_its_field():
    G = tiny_hierarchical_graph()
    template = template_from_report(_hand_sections(), G, boundaries=TINY_ROLES)
    assert template.murray_exponent == 2.9
    assert template.murray_iqr == (2.6, 3.2)
    assert template.daughter_angle_deg == 80.0
    assert template.intercapillary_distance_um == 25.0
    assert template.intercapillary_distance_median_um == 24.0
    assert template.mean_tortuosity == 1.21
    assert template.loops.length_percentiles_um["Median"] == 150.0
    assert template.loops.independent_loops_per_mm3 == 1000.0
    assert template.loops.vessels_on_no_loop_fraction == 0.25
    assert template.loops.independent_loops == 7  # the cyclomatic number wins over the density
    assert template.dead_ends.artefacts == 2
    assert template.dead_ends.interior == 1 + 3  # blind + unknown
    assert template.dead_ends.interior_per_mm == 4.5
    assert template.tissue_volume_um3 == 5.0e6


def test_loop_count_comes_from_the_density_without_a_cyclomatic_section():
    sections = _hand_sections()
    sections.pop("cyclomatic_number")
    template = template_from_report(sections, tiny_hierarchical_graph(), boundaries=TINY_ROLES)
    expected = round(1000.0 * template.image_volume_um3 / 1e9)
    assert template.loops.independent_loops == expected


def test_not_available_values_fall_back_with_a_note():
    sections = _hand_sections()
    note = "N/A (no junction of two or more narrower measured daughters)"
    for row in ("Median Fitted Murray Exponent", "Fitted Murray Exponent IQR Low", "Fitted Murray Exponent IQR High"):
        sections["murray_law"][row] = note
    sections["daughter_angles"]["Mean Daughter-Daughter Angle (degrees)"] = "N/A (no bifurcation)"
    template = template_from_report(sections, tiny_hierarchical_graph(), boundaries=TINY_ROLES)
    assert template.murray_exponent is None
    assert template.murray_iqr is None
    assert template.daughter_angle_deg is None
    assert any(f"{DEFAULT_MURRAY_EXPONENT:g} is used" in n for n in template.provenance.notes)
    assert any(f"{DEFAULT_DAUGHTER_ANGLE_DEG:g} degrees" in n for n in template.provenance.notes)


def test_a_report_volume_unlike_the_extent_is_noted_not_trusted():
    sections = _hand_sections()
    sections["vessel_density"]["Total Image Volume (micron³)"] = 1.0
    template = template_from_report(sections, tiny_hierarchical_graph(), boundaries=TINY_ROLES)
    assert template.image_volume_um3 == pytest.approx(51 * 2.0 * 201 * 301)
    assert any("image volume" in n for n in template.provenance.notes)


# -- what a missing section says --------------------------------------------


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_a_missing_required_section_names_its_toggles(tmp_path, section):
    path = write_fixture_run(tmp_path, drop_sections=(section,))
    with pytest.raises(TemplateError) as caught:
        load_template(path, "statistics")
    error = caught.value
    assert error.reason == "no_statistics"
    assert error.missing_statistics == SECTION_TOGGLES[section]
    assert f"'{section}'" in str(error)
    for toggle in SECTION_TOGGLES[section]:
        assert toggle in str(error)
    assert "Additional measurements" in str(error)


def test_every_missing_section_is_named_in_one_error(tmp_path):
    path = write_fixture_run(tmp_path, drop_sections=("dead_ends", "loop_hierarchy"))
    with pytest.raises(TemplateError) as caught:
        load_template(path, "statistics")
    assert caught.value.missing_statistics == (
        "statistics_dead_ends", "statistics_network_analysis", "statistics_loop_hierarchy",
    )


def test_optional_sections_are_never_required(tmp_path):
    path = write_fixture_run(tmp_path, drop_sections=OPTIONAL_SECTIONS)
    template = load_template(path, "statistics")
    assert template.branching_points is None
    assert template.loops.independent_loops is not None  # from the density


def test_a_run_saved_with_statistics_off_names_the_master_switch(tmp_path):
    path = write_fixture_run(tmp_path, with_report=False)
    with pytest.raises(TemplateError) as caught:
        load_template(path, "statistics")
    assert caught.value.reason == "no_statistics"
    assert caught.value.missing_statistics[0] == "statistics"
    assert "Statistics off" in str(caught.value)


def test_the_graph_source_needs_no_report(tmp_path):
    path = write_fixture_run(tmp_path, with_report=False)
    template = load_template(path, "graph")
    assert template.vessel_count == 12


# -- what cannot be a template ---------------------------------------------


@pytest.mark.parametrize("source", ["statistics", "graph"])
def test_a_synthetic_run_is_refused_by_its_graph_marker(tmp_path, source):
    path = write_fixture_run(tmp_path, graph_attributes={"network_source": "synthetic"})
    with pytest.raises(TemplateError) as caught:
        load_template(path, source)
    assert caught.value.reason == "synthetic_run"
    assert "microscopy-image run" in str(caught.value)


def test_a_synthetic_run_is_refused_by_its_settings(tmp_path):
    path = write_fixture_run(tmp_path, settings={"network_source": "synthetic"})
    with pytest.raises(TemplateError, match="itself a synthetic network"):
        load_template(path, "statistics")


def test_global_mask_classes_are_refused(tmp_path):
    G = tiny_hierarchical_graph()
    for *_e, d in G.edges(data=True):
        d["branch_order"] = "Unknown_Cap" if d["branch_order"].startswith("B") else "Unknown_Small"
    path = write_fixture_run(tmp_path, graph=G)
    with pytest.raises(TemplateError) as caught:
        load_template(path, "graph")
    assert caught.value.reason == "global_classes"
    assert "arteriole and venule" in str(caught.value)


def test_a_run_without_branch_orders_is_refused(tmp_path):
    G = tiny_hierarchical_graph()
    for *_e, d in G.edges(data=True):
        d.pop("branch_order")
    path = write_fixture_run(tmp_path, graph=G, with_report=False)
    with pytest.raises(TemplateError) as caught:
        load_template(path, "graph")
    assert caught.value.reason in {"not_a_run", "no_branch_orders"}
    assert "Boundaries stage" in str(caught.value)


def test_a_run_without_inlets_or_outlets_is_refused(tmp_path):
    roles = dict(TINY_ROLES, outlet=())
    path = write_fixture_run(tmp_path, roles=roles, settings={"outlet_nodes": []})
    with pytest.raises(TemplateError) as caught:
        load_template(path, "statistics")
    assert caught.value.reason == "no_boundaries"


def test_a_run_without_diameters_is_refused(tmp_path):
    G = tiny_hierarchical_graph()
    for *_e, d in G.edges(data=True):
        d.pop("diameter_um")
    path = write_fixture_run(tmp_path, graph=G)
    with pytest.raises(TemplateError) as caught:
        load_template(path, "statistics")
    assert caught.value.reason == "no_diameters"


def test_a_file_that_is_not_a_run_raises_the_snapshot_error(tmp_path):
    bad = tmp_path / "not_a_run.haemorun"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(RunSnapshotError):
        load_template(bad, "statistics")


def test_an_unknown_source_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="source must be one of"):
        load_template(write_fixture_run(tmp_path), "guess")


# -- boundaries, bridges, the file's graphs -----------------------------------


def test_endpoint_faces_read_the_box_and_its_tolerance():
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([50.0, 100.0, 0.5]))     # within 2 voxels of x0
    G.add_node(1, pos=np.array([50.0, 100.0, 150.0]))   # inside
    G.add_node(2, pos=np.array([99.0, 100.0, 150.0]))   # within 1 % of z1 (2 um)
    G.add_node(3, pos=np.array([50.0, 2.5, 150.0]))     # 2.5 um from y0: not within 2 um nor 1 % of 200
    ends = endpoints_from_nodes(G, [0, 1, 2, 3], TINY_EXTENT_UM, TINY_VOXEL_ZYX)
    assert [e.face for e in ends] == ["x0", "interior", "z1", "interior"]
    assert ends[1].position_rel == pytest.approx((0.5, 0.5, 0.5))


def test_boundary_lists_fall_back_to_the_runs_settings(tmp_path):
    settings = {f"{role}_nodes": list(nodes) for role, nodes in TINY_ROLES.items()}
    path = write_fixture_run(tmp_path, roles={role: () for role in TINY_ROLES}, settings=settings)
    template = load_template(path, "statistics")
    assert template.inlet_count == 1 and template.outlet_count == 1
    assert template.arteriole_boundary_count == 2


def test_the_graph_source_contracts_zero_resistance_bridges(tmp_path):
    G = tiny_hierarchical_graph()
    G.add_node(20, pos=np.array([50.0, 100.0, 30.0]))
    # A bridge is no vessel: it must not count as one, nor break the counts.
    G.add_edge(1, 20, voxels=[[50.0, 100.0, 60.0], [50.0, 100.0, 30.0]], length=30.0, **{IS_ZERO_RESISTANCE: True})
    path = write_fixture_run(tmp_path, graph=G)
    template = load_template(path, "graph")
    assert {k: v.count for k, v in template.classes.items()} == tiny_expected()["class_counts"]


def test_the_export_checkpoints_graph_is_preferred(tmp_path):
    marked = tiny_hierarchical_graph()
    marked.graph["image_voxel_size_zyx"] = (4.0, 1.0, 1.0)
    marked.graph["voxel_size"] = (4.0, 1.0, 1.0)
    marked.graph["image_extent_um"] = (200.0, 200.0, 300.0)
    # The results graph says 2 um voxels, the export checkpoint 4 um: the checkpoint wins.
    path = write_fixture_run(tmp_path, graph=marked)
    from haemolynx.gui.run_snapshot import read_run_snapshot

    snapshot = read_run_snapshot(path)
    other = tiny_hierarchical_graph(with_report=True)
    snapshot = replace(snapshot, results_state={**snapshot.results_state, "graph": other})
    template = tpl.template_from_statistics(snapshot)
    assert template.voxel_size_zyx == (4.0, 1.0, 1.0)


def test_the_capillary_scheme_has_no_tree_orders(tmp_path):
    G = tiny_hierarchical_graph()
    for *_e, d in G.edges(data=True):
        d["branch_order"] = "B01"
    roles = dict(TINY_ROLES, arteriole_boundary=(), venule_boundary=(), large_arteriole_boundary=(), large_venule_boundary=())
    path = write_fixture_run(tmp_path, graph=G, roles=roles)
    template = load_template(path, "graph")
    assert template.scheme == "capillary"
    assert [o.category for o in template.orders] == ["capillary"]
    assert template.orders_on("arterial") == ((), ())
    assert set(template.classes) == {"capillary"}


# -- provenance and the loader ------------------------------------------------


def test_provenance_reads_the_runs_manifest_when_it_is_there(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stack_run_manifest.json").write_text(
        json.dumps({"haemolynx_version": "9.9.9", "git": {"commit": "abc123"}}), encoding="utf-8"
    )
    template = load_template(write_fixture_run(tmp_path), "statistics")
    assert template.provenance.run_version == "9.9.9"
    assert template.provenance.run_commit == "abc123"
    assert template.provenance.template_path == str(tmp_path / "template.haemorun")
    assert template.provenance.statistics_mode == "fast"
    assert template.provenance.haemolynx_version is not None


def test_load_template_remembers_a_file_until_it_changes(tmp_path):
    path = write_fixture_run(tmp_path)
    first = load_template(path, "statistics")
    assert load_template(path, "statistics") is first
    assert load_template(path, "graph") is not first
    import os
    import time

    write_fixture_run(tmp_path)
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert load_template(path, "statistics") is not first


def test_summary_names_what_a_log_line_needs(tmp_path):
    summary = load_template(write_fixture_run(tmp_path), "statistics").summary()
    assert summary["vessels"] == 12
    assert summary["inlets"] == 1 and summary["outlets"] == 1
    assert summary["orders"][0] == "Large_Art1"
    assert summary["source"] == "statistics"


def test_the_graph_source_reads_a_measured_tissue_volume_from_the_saved_report(tmp_path):
    sections = tiny_sections()
    sections["vessel_density"] = {**sections["vessel_density"], "Tissue Volume (micron³)": 4.5e6}
    path = write_fixture_run(tmp_path, sections=sections)
    assert load_template(path, "graph").tissue_volume_um3 == 4.5e6
    assert load_template(path, "statistics").tissue_volume_um3 == 4.5e6


# -- the diameter basis ---------------------------------------------------------


@pytest.mark.parametrize("source", ["statistics", "graph"])
def test_the_template_is_on_the_basis_its_run_segmented(tmp_path, source):
    path = write_fixture_run(tmp_path, settings={"segmentation_source": "endothelial"})
    template = load_template(path, source)
    assert template.diameter_basis == "endothelial"
    assert template.summary()["diameter_basis"] == "endothelial"
    assert not any("basis" in note for note in template.provenance.notes)


def test_a_vessels_own_basis_counts_and_most_vessels_decide(tmp_path):
    G = tiny_hierarchical_graph()
    edges = list(G.edges(keys=True, data=True))
    for _u, _v, _k, data in edges[:7]:
        data["diameter_basis"] = "endothelial"  # read inside the wall ring, as stamping marks it
    template = load_template(write_fixture_run(tmp_path, graph=G), "statistics")
    assert template.diameter_basis == "endothelial"
    assert any("two bases (7 endothelial, 5 plasma_column)" in note for note in template.provenance.notes)


def test_a_tie_goes_to_the_runs_own_basis():
    G = tiny_hierarchical_graph()
    for i, (*_e, data) in enumerate(G.edges(data=True)):
        data["diameter_basis"] = "endothelial" if i % 2 else "plasma_column"
    notes: list = []
    assert tpl.template_diameter_basis(G, "endothelial", notes) == "endothelial"
    assert tpl.template_diameter_basis(G, "plasma_column", []) == "plasma_column"
    assert notes and "6 endothelial, 6 plasma_column" in notes[0]


def test_the_graphs_record_wins_over_the_settings_and_old_names_read():
    G = tiny_hierarchical_graph()
    assert tpl.run_diameter_basis_of(G, {"segmentation_source": "endothelial"}) == "endothelial"
    assert tpl.run_diameter_basis_of(G, {"diameter_basis": "anatomical"}) == "endothelial"
    G.graph["diameter_basis"] = "plasma_column"  # what the run's haemodynamics recorded
    assert tpl.run_diameter_basis_of(G, {"segmentation_source": "endothelial"}) == "plasma_column"
    assert tpl.run_diameter_basis_of(tiny_hierarchical_graph(), {}) is None


def test_a_template_whose_run_names_no_basis_has_none_and_says_so():
    G = tiny_hierarchical_graph()
    template = tpl.measure_template(G, inlet_nodes=TINY_ROLES["inlet"], outlet_nodes=TINY_ROLES["outlet"])
    assert template.diameter_basis is None
    assert any("records no diameter basis" in note for note in template.provenance.notes)
    named = tpl.measure_template(G, inlet_nodes=TINY_ROLES["inlet"], outlet_nodes=TINY_ROLES["outlet"], run_basis="endothelial")
    assert named.diameter_basis == "endothelial"


def test_a_template_written_with_a_vessel_box_still_loads():
    """vessel_box_um was dropped on 2026-10-24 (Harvey: the bed's own volume
    takes its place); a template written before carries it."""
    data = fixture_template().to_dict()
    assert "vessel_box_um" not in data
    old = {**data, "vessel_box_um": [[0.0, 0.0, 0.0], [120.0, 200.0, 200.0]]}
    assert NetworkTemplate.from_dict(old) == fixture_template()


def test_an_older_template_without_a_basis_still_loads(tmp_path):
    template = replace(fixture_template(), diameter_basis="endothelial")
    data = template.to_dict()
    assert data["diameter_basis"] == "endothelial"
    assert NetworkTemplate.from_dict(data) == template
    assert template.scaled(2.0).diameter_basis == "endothelial"
    data.pop("diameter_basis")  # a template written before the basis was read
    assert NetworkTemplate.from_dict(data).diameter_basis is None
    assert NetworkTemplate.from_dict({**data, "diameter_basis": "anatomical"}).diameter_basis == "endothelial"
    assert NetworkTemplate.from_dict({**data, "diameter_basis": "lumen"}).diameter_basis is None


# -- the data model on its own -----------------------------------------------


def test_json_round_trip_keeps_every_field(tmp_path):
    template = load_template(write_fixture_run(tmp_path), "graph")
    path = template.to_json(tmp_path / "template.json")
    back = NetworkTemplate.from_json(path)
    assert back == template
    assert json.loads(path.read_text(encoding="utf-8"))["classes"]["capillary"]["samples"]["diameters_um"] == [5.0, 4.0, 5.0, 5.0, 4.0]


def test_nan_is_written_as_null():
    template = replace(fixture_template(), mean_tortuosity=math.nan)
    assert template.to_dict()["mean_tortuosity"] is None


def test_scaled_counts_go_as_the_volume_and_endpoints_as_the_faces():
    template = fixture_template()
    scaled = template.scaled(2.0)
    assert scaled.extent_um == (240.0, 400.0, 400.0)
    assert scaled.image_shape == (241, 409, 409)
    assert scaled.classes["capillary"].count == 160 * 8
    assert scaled.classes["capillary"].total_length_um == pytest.approx(160 * 50.0 * 8)
    assert scaled.classes["capillary"].mean_length_um == 50.0
    assert scaled.classes["capillary"].mean_diameter_um == 5.0
    assert [o.edge_count for o in scaled.orders] == [16, 32, 48, 16, 32]
    assert len(scaled.inlets) == 8 and len(scaled.outlets) == 8
    assert {e.face for e in scaled.inlets} == {"x0"}
    assert scaled.loops.independent_loops == 51 * 8
    assert scaled.loops.independent_loops_per_mm3 == template.loops.independent_loops_per_mm3
    assert scaled.dead_ends.leaves_image == 8 * 4
    assert scaled.dead_ends.interior == 8 * 8
    assert scaled.dead_ends.total == 8 * 4 + 8 * 8
    assert scaled.arteriole_boundary_count == 48
    assert any("scaled by 2" in n for n in scaled.provenance.notes)
    assert template.scaled(1.0) is template


def test_scaling_down_keeps_a_face_with_one_inlet():
    scaled = fixture_template().scaled(0.5)
    assert len(scaled.inlets) == 1 and scaled.inlets[0].face == "x0"
    assert scaled.classes["capillary"].count == 20


@pytest.mark.parametrize("factor", [0.0, -1.0, math.nan])
def test_a_bad_scale_factor_raises(factor):
    with pytest.raises(ValueError):
        fixture_template().scaled(factor)


def _lognormal_mean_median(mu: float, sigma: float) -> tuple[float, float]:
    return math.exp(mu + sigma**2 / 2.0), math.exp(mu)


def test_lognormal_from_mean_and_median():
    mu, sigma = lognormal_from_mean_median(38.0, 34.0)
    assert sigma**2 == pytest.approx(2.0 * math.log(38.0 / 34.0))
    assert _lognormal_mean_median(mu, sigma) == pytest.approx((38.0, 34.0))
    assert lognormal_from_mean_median(30.0, 30.0) == (math.log(30.0), 0.1)
    assert lognormal_from_mean_median(30.0, 35.0)[1] == 0.1
    with pytest.raises(ValueError):
        lognormal_from_mean_median(0.0, 1.0)


def test_lognormal_from_mean_and_cv():
    mu, sigma = lognormal_from_mean_cv(5.0, 0.3)
    mean, _median = _lognormal_mean_median(mu, sigma)
    assert mean == pytest.approx(5.0)
    assert math.sqrt(math.exp(sigma**2) - 1.0) == pytest.approx(0.3)
    assert lognormal_from_mean_cv(5.0, 0.0) == (math.log(5.0), 0.0)


def test_lognormal_from_median_and_percentiles():
    mu, sigma = lognormal_from_median_p10_p90(30.0, 18.0, 50.0)
    assert mu == math.log(30.0)
    z = 1.2815515655446004
    assert math.exp(mu + z * sigma) / math.exp(mu - z * sigma) == pytest.approx(50.0 / 18.0)
    assert lognormal_from_median_p10_p90(30.0, 30.0, 30.0)[1] == 0.1


def test_samples_default_to_empty_tuples():
    assert Samples() == Samples((), (), ())
    assert Endpoint("x0", (0.5, 0.5, 0.0)).face == "x0"
