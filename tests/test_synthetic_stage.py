"""A synthetic run through the stages (``network_source=synthetic``): the
Input, Skeletonise and Graph stages replaced, the Boundaries stage reading
the generator's roles, the widths kept, every output named after the
template, and preflight saying what a template lacks."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.dead_ends import DEAD_END_KIND
from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE
from haemolynx.haemodynamics.poiseuille import DIAMETER_SOURCE_SYNTHETIC, EDGE_DIAMETER_BASIS, PoiseuilleModel
from haemolynx.haemodynamics.viscosity import run_viscosity_law
from haemolynx.io.vessel_masks import vessel_masks_are_global
from haemolynx.pipeline import default_schema, preflight, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.checks import check_synthetic_template
from haemolynx.pipeline.naming import run_stem
from haemolynx.pipeline.stages import (
    STATISTICS_REPORT,
    SegmentedInputs,
    SkeletonisedVolume,
    assign_boundaries,
    assign_diameters,
    build_haemodynamic_model,
    build_network,
    raw_image_is_set,
    segment,
    segmented_input_path,
    skeletonise,
    synthetic_box_shape,
)
from haemolynx.synthetic import COMPARISON_SECTION, GRAPH_TEMPLATE_RECORD, boundary_nodes, clear_template_cache
from haemolynx.synthetic.compare import CSV_COLUMNS
from synthetic_fixtures import fixture_template, write_fixture_run, write_generated_run

SCHEMA = default_schema()


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_template_cache()
    yield
    clear_template_cache()


@pytest.fixture(scope="module")
def template_run(tmp_path_factory) -> Path:
    return write_generated_run(tmp_path_factory.mktemp("template"))


@pytest.fixture(scope="module")
def endothelial_template_run(tmp_path_factory) -> Path:
    """A template run whose segmentation was of the vessel wall."""
    return write_generated_run(
        tmp_path_factory.mktemp("endothelial"), name="claudin.haemorun", settings={"segmentation_source": "endothelial"}
    )


def _settings(template: Path, out: Path, **overrides) -> dict:
    values = {setting.name: setting.default for setting in SCHEMA}
    values.update(
        {
            "network_source": "synthetic",
            "synthetic_template_path": template,
            "synthetic_seed": 3,
            "vtk_output_prefix": out / "out" / "run",
            "plot_dir": out / "plots",
            "statistics": True,
            "show_plots_in_ide": False,
            "interactive_plots": False,
            "visualize_results": False,
        }
    )
    values.update(overrides)
    return resolve_settings(values, schema=SCHEMA, config_path=None)


# -- the stages one at a time ---------------------------------------------------


def test_the_input_stage_reads_the_template_and_leaves_input_path_alone(template_run, tmp_path):
    settings = _settings(template_run, tmp_path, input_path=tmp_path / "stale.tif")
    inputs = segment(settings)
    assert isinstance(inputs, SegmentedInputs)
    assert inputs.synthetic and inputs.input_format == "haemorun"
    assert inputs.template_path == template_run and inputs.image_path == template_run
    assert inputs.output_dir == tmp_path / "out"
    assert settings["input_path"] == tmp_path / "stale.tif"  # never resolved, never replaced
    assert raw_image_is_set({**settings, "raw_image_path": "raw.tif"}) is False
    assert segmented_input_path(settings) is None
    assert run_stem(settings) == "e14_5_synthetic"


def test_the_input_stage_needs_a_template(tmp_path):
    settings = _settings(tmp_path / "none.haemorun", tmp_path)
    settings["synthetic_template_path"] = None
    with pytest.raises(ValueError, match="synthetic_template_path"):
        segment(settings)


def test_the_skeletonise_stage_makes_an_empty_volume_with_the_templates_voxels(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    volume = skeletonise(settings, segment(settings))
    assert isinstance(volume, SkeletonisedVolume)
    assert volume.synthetic
    assert volume.image is None and volume.skeleton is None
    assert volume.voxel_size_zyx == fixture_template().voxel_size_zyx
    assert volume.voxel_size_xyz == tuple(reversed(fixture_template().voxel_size_zyx))
    assert (tmp_path / "out").is_dir() and (tmp_path / "plots").is_dir()
    assert not list((tmp_path / "out").glob("*_skeleton.npy"))


def test_the_graph_stage_draws_the_network_and_pickles_it(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    volume = skeletonise(settings, segment(settings))
    network = build_network(settings, volume, SCHEMA)
    G = network.graph
    assert isinstance(G, nx.MultiGraph) and G.number_of_edges() > 100
    assert G.graph["network_source"] == "synthetic"
    assert G.graph[GRAPH_TEMPLATE_RECORD]["seed"] == 3
    assert G.graph["image_voxel_size_zyx"] == fixture_template().voxel_size_zyx
    assert synthetic_box_shape(G) == fixture_template().image_shape
    assert all(DEAD_END_KIND in G.nodes[n] for n in G if G.degree(n) == 1)
    pickled = tmp_path / "out" / "e14_5_synthetic_graph.pkl"
    assert pickled.is_file()
    with pickled.open("rb") as handle:
        assert pickle.load(handle).number_of_edges() == G.number_of_edges()
    again = build_network(settings, volume, SCHEMA).graph
    assert sorted(again.edges(keys=True)) == sorted(G.edges(keys=True))
    other = build_network({**settings, "synthetic_seed": 4}, volume, SCHEMA).graph
    assert sorted(other.edges(keys=True)) != sorted(G.edges(keys=True)) or any(
        not np.array_equal(other.nodes[n]["pos"], G.nodes[n]["pos"]) for n in G if n in other
    )


def test_the_graph_stage_draws_the_bed_model_the_settings_choose(template_run, tmp_path):
    settings = _settings(template_run, tmp_path, synthetic_bed_model="voronoi")
    G = build_network(settings, skeletonise(settings, segment(settings)), SCHEMA).graph
    assert G.graph[GRAPH_TEMPLATE_RECORD]["options"]["bed_model"] == "voronoi"
    assert G.graph[GRAPH_TEMPLATE_RECORD]["generated"]["fill_phases"]["voronoi_edges"] > 0


def test_the_graph_stage_hands_a_resumed_network_back(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    volume = skeletonise(settings, segment(settings))
    first = build_network(settings, volume, SCHEMA).graph
    resumed = build_network(settings, volume, SCHEMA, resumed_graph=first).graph
    assert resumed is first


def test_the_boundaries_stage_takes_the_generators_roles(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    network = build_network(settings, skeletonise(settings, segment(settings)), SCHEMA)
    roles = boundary_nodes(network.graph)
    boundaries = assign_boundaries(settings, network)
    assert boundaries.inlet_nodes == roles["inlet"] and boundaries.outlet_nodes == roles["outlet"]
    assert boundaries.arteriole_boundary_nodes == roles["arteriole_boundary"]
    assert boundaries.venule_boundary_nodes == roles["venule_boundary"]
    assert boundaries.resistance_node_pair == (roles["inlet"][0], roles["outlet"][0])
    assert boundaries.large_arteriole_mask is None and boundaries.small_arteriole_mask is None
    # The run's own lists, filled in place as the manual methods fill them.
    assert settings["inlet_nodes"] == roles["inlet"]
    assert settings["venule_boundary_nodes"] == roles["venule_boundary"]


def test_the_boundaries_stage_refuses_a_network_without_an_outlet(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    network = build_network(settings, skeletonise(settings, segment(settings)), SCHEMA)
    from haemolynx.synthetic.generate import SYNTHETIC_ROLES

    for node in network.graph:
        roles = network.graph.nodes[node].get(SYNTHETIC_ROLES)
        if roles and "outlet" in roles:
            network.graph.nodes[node][SYNTHETIC_ROLES] = tuple(r for r in roles if r != "outlet")
    with pytest.raises(ValueError, match="no inlet or no outlet"):
        assign_boundaries(settings, network)


def test_the_diameters_stage_keeps_the_drawn_widths(template_run, tmp_path):
    settings = _settings(template_run, tmp_path)
    network = build_network(settings, skeletonise(settings, segment(settings)), SCHEMA)
    drawn = {(u, v, k): d["diameter_um"] for u, v, k, d in network.graph.edges(keys=True, data=True)}
    boundaries = assign_boundaries(settings, network)
    model = assign_diameters(settings, network, boundaries, SCHEMA)
    G = model.graph
    for u, v, k, d in G.edges(keys=True, data=True):
        assert d["diameter_source"] == DIAMETER_SOURCE_SYNTHETIC
        assert d["diameter_um"] == drawn[(u, v, k)]
        assert d[EDGE_DIAMETER_BASIS] == "plasma_column"  # the template run's
        assert d["branch_order"] is not None
    labels = {d["branch_order"] for *_e, d in G.edges(data=True)}
    assert {"Art1", "Art3", "Ven1", "Ven2", "B01"} <= labels


def test_the_drawn_vessels_keep_the_templates_basis_through_the_haemodynamics(endothelial_template_run, tmp_path):
    settings = _settings(endothelial_template_run, tmp_path)
    assert settings["segmentation_source"] == "plasma_column"  # this run's own, not the template's
    network = build_network(settings, skeletonise(settings, segment(settings)), SCHEMA)
    boundaries = assign_boundaries(settings, network)
    model = build_haemodynamic_model(settings, assign_diameters(settings, network, boundaries, SCHEMA), SCHEMA)
    G = model.graph
    law = PoiseuilleModel(1.0, 1.0, viscosity_law=run_viscosity_law(settings), haematocrit=settings["haematocrit"])
    assert law.diameter_basis == "plasma_column"
    for *_e, d in G.edges(data=True):
        assert d[EDGE_DIAMETER_BASIS] == "endothelial"
        wall_to_wall = law.resistance_of_uniform_segment(d["length"], d["diameter_um"], diameter_basis="endothelial")
        assert d["resistance"] == pytest.approx(wall_to_wall, rel=1e-9)
    # Read on this run's own basis, every capillary would have had another resistance.
    capillary = next(d for *_e, d in G.edges(data=True) if str(d["branch_order"]).startswith("B"))
    assert law.resistance_of_uniform_segment(capillary["length"], capillary["diameter_um"]) != pytest.approx(
        capillary["resistance"], rel=1e-3
    )


def test_no_vessel_mask_is_global_for_a_synthetic_network(template_run, tmp_path):
    settings = _settings(
        template_run, tmp_path,
        automated_vessel_assignment=True, use_large_vessel_masks=True,
        large_arteriole_mask_path=tmp_path / "large.tif", large_venule_mask_path=None,
    )
    assert vessel_masks_are_global(settings) is False
    assert vessel_masks_are_global({**settings, "network_source": "segmented_image"}) is True


# -- the whole run ------------------------------------------------------------


@pytest.fixture(scope="module")
def whole_run(template_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    settings = _settings(template_run, out)
    report = preflight(settings, SCHEMA)
    G = run_pipeline_stages(settings, SCHEMA)
    return settings, report, G, out


def test_preflight_passes_and_names_the_template(whole_run):
    _settings_, report, _G, _out = whole_run
    assert report.ok, report.errors
    assert any(label == "synthetic template" and "178 vessels" in detail for label, detail in report.passed)


def test_every_output_is_named_after_the_template(whole_run):
    _settings_, _report, G, out = whole_run
    names = {path.name for path in (out / "out").iterdir()}
    for suffix in (
        "_graph.pkl", "_statistics.csv", "_branch_statistics.csv", "_template_comparison.csv",
        "_run_manifest.json", "_run_settings.yaml", "_citations.txt",
    ):
        assert f"e14_5_synthetic{suffix}" in names, (suffix, names)
    assert "run_vessels.vtp" in names
    assert not any(name.startswith("stale") for name in names)
    assert G.number_of_edges() > 100


def test_the_run_solves_and_reports_its_comparison(whole_run):
    _settings_, _report, G, out = whole_run
    assert all("flow_abs" in d for *_e, d in G.edges(data=True))
    report = G.graph[STATISTICS_REPORT]
    assert COMPARISON_SECTION in report["sections"]
    section = report["sections"][COMPARISON_SECTION]
    within, _of, fitted = section["fitted characteristics within tolerance"].split()
    assert int(fitted) >= 50 and int(within) >= int(fitted) - 3
    with (out / "out" / "e14_5_synthetic_template_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert any(row[1] == "inlets" and row[6] == "yes" for row in rows[1:])


def test_the_manifest_records_the_template_as_the_runs_input(whole_run):
    settings, _report, _G, out = whole_run
    manifest = json.loads((out / "out" / "e14_5_synthetic_run_manifest.json").read_text(encoding="utf-8"))
    inputs = {record["setting"]: record for record in manifest["inputs"]}
    assert "synthetic_template_path" in inputs
    assert "input_path" not in inputs
    assert manifest["settings"]["network_source"] == "synthetic"
    assert manifest["status"] == "finished"
    assert manifest["network_source"] == "synthetic"
    assert manifest["synthetic"]["seed"] == 3
    assert manifest["synthetic"]["bed_model"] == "delaunay"
    assert manifest["synthetic"]["source"] == "statistics"
    assert manifest["synthetic"]["template"].endswith("e14_5.haemorun")
    citations = (out / "out" / "e14_5_synthetic_citations.txt").read_text(encoding="utf-8")
    assert "Network generation methods" in citations and "- skan" not in citations


def test_a_synthetic_run_can_be_rerun_with_thick_vessel_skeletonisation_left_on(whole_run, template_run, tmp_path):
    settings = _settings(template_run, tmp_path, use_thick_vessel_skeletonisation=True, statistics=False)
    warnings = check_synthetic_template(settings).warnings
    assert any(w.startswith("use_thick_vessel_skeletonisation is on but not read") for w in warnings)
    G = run_pipeline_stages(settings, SCHEMA)
    # Nothing is skeletonised, so nothing is bridged: the network, its widths
    # and its flows are those of the same seed's run without the switch.
    _settings_, _report, base, _out = whole_run
    assert not any(d.get(IS_ZERO_RESISTANCE) for *_e, d in G.edges(data=True))
    assert sorted(G.edges(keys=True)) == sorted(base.edges(keys=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        other = base.edges[u, v, k]
        assert d["diameter_um"] == other["diameter_um"]
        assert d["resistance"] == other["resistance"]
        assert d["flow_abs"] == pytest.approx(other["flow_abs"], rel=1e-9, abs=1e-30)


# -- preflight on what a template lacks ----------------------------------------


def test_preflight_refuses_a_file_that_is_not_a_haemorun(tmp_path):
    bad = tmp_path / "stack.tif"
    bad.write_bytes(b"\x00")
    report = check_synthetic_template(_settings(bad, tmp_path))
    assert not report.ok and "not a .haemorun file" in report.errors[0]
    assert "Fix: save the template run from the panel (Save run)" in report.errors[0]


def test_preflight_refuses_a_haemorun_that_is_not_a_run_and_says_what_to_choose(tmp_path):
    bad = tmp_path / "renamed.haemorun"
    bad.write_bytes(b"not a run")
    report = check_synthetic_template(_settings(bad, tmp_path))
    assert not report.ok and len(report.errors) == 1
    assert report.errors[0].startswith(f"synthetic_template_path: Could not read {bad}")
    assert report.errors[0].endswith("Fix: choose a run the panel saved (Save run), of a microscopy-image run.")


def test_preflight_names_the_statistics_a_template_lacks(tmp_path):
    path = write_fixture_run(tmp_path, with_report=False)
    report = check_synthetic_template(_settings(path, tmp_path))
    assert not report.ok
    assert "Statistics off" in report.errors[0] and "statistics" in report.errors[0]
    path = write_fixture_run(tmp_path, drop_sections=("loop_hierarchy",), name="partial.haemorun")
    report = check_synthetic_template(_settings(path, tmp_path))
    assert "statistics_loop_hierarchy" in report.errors[0] and "Additional measurements" in report.errors[0]


def test_preflight_refuses_a_synthetic_template(tmp_path):
    path = write_fixture_run(tmp_path, settings={"network_source": "synthetic"})
    report = check_synthetic_template(_settings(path, tmp_path))
    assert not report.ok and "itself a synthetic network" in report.errors[0]


def test_preflight_warns_when_this_runs_basis_is_not_the_templates(endothelial_template_run, tmp_path):
    report = check_synthetic_template(_settings(endothelial_template_run, tmp_path))
    assert report.ok, report.errors
    basis = [w for w in report.warnings if w.startswith("segmentation_source")]
    assert len(basis) == 1
    assert "segmentation_source is plasma_column but the template's diameters are endothelial" in basis[0]
    assert "Fix: set segmentation_source to endothelial" in basis[0]
    same = check_synthetic_template(_settings(endothelial_template_run, tmp_path, segmentation_source="endothelial"))
    assert not any(w.startswith("segmentation_source") for w in same.warnings)


def test_preflight_refuses_what_needs_an_image_and_warns_about_what_is_not_read(template_run, tmp_path):
    settings = _settings(
        template_run, tmp_path,
        measurement_3d_to_cell_mask=True, cell_mask_path=tmp_path / "cells.tif",
        measure_tissue_volume=True,
        use_ilastik_segmentation=True, raw_image_path=tmp_path / "raw.tif",
        automated_vessel_assignment=True, do_skeletonize=False,
    )
    report = check_synthetic_template(settings)
    assert len(report.errors) == 2
    assert any("measurement_3d_to_cell_mask" in e for e in report.errors)
    assert any("measure_tissue_volume" in e for e in report.errors)
    joined = " ".join(report.warnings)
    assert "use_ilastik_segmentation" in joined and "automated_vessel_assignment" in joined and "do_skeletonize" in joined


def test_preflight_does_not_demand_paths_the_input_tab_hides(template_run, tmp_path):
    settings = _settings(
        template_run, tmp_path,
        use_ilastik_segmentation=True, raw_image_path=tmp_path / "raw.tif",
        ilastik_classifier_path=tmp_path / "missing.ilp",
        automated_vessel_assignment=True, use_large_vessel_masks=True,
        large_arteriole_mask_path=tmp_path / "missing_a.tif", large_venule_mask_path=tmp_path / "missing_v.tif",
    )
    report = preflight(settings, SCHEMA)
    assert report.ok, report.errors


def test_preflight_still_demands_the_template_itself(tmp_path):
    report = preflight(_settings(tmp_path / "missing.haemorun", tmp_path), SCHEMA)
    assert not report.ok
    assert any("synthetic_template_path" in e for e in report.errors)
