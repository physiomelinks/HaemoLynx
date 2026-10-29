"""The output data dictionary (export.6a) and the units in every export (export.6b):
each file's documented columns are the ones its writer writes, every column
has a defined unit, code-style column names carry theirs, and the VTK files
name each array's unit in their field data."""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from haemolynx.pipeline import export_dictionary as dictionary
from haemolynx.pipeline.export_dictionary import FILES, UNIT_SUFFIXES, UNITS, output_file, undocumented

REPO = Path(__file__).resolve().parents[1]
CODE_STYLE = re.compile(r"[a-z0-9_]+")


def _names(file_name: str) -> list[str]:
    return [c.name for c in output_file(file_name).columns]


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def test_the_page_is_written_from_the_module():
    page = (REPO / dictionary.EXPORT_DICTIONARY_PAGE).read_text(encoding="utf-8")
    assert page == dictionary.export_dictionary_markdown(), (
        "docs/export_dictionary.md is stale: python -m haemolynx.pipeline.export_dictionary"
    )


def test_every_column_has_a_defined_unit_and_a_meaning():
    assert set(UNIT_SUFFIXES) <= set(UNITS)
    for item in FILES:
        assert item.what and item.where, item.name
        for column in item.columns:
            assert column.unit in UNITS, (item.name, column.name, column.unit)
            assert column.meaning, (item.name, column.name)
        names = [c.name for c in item.columns]
        assert len(names) == len(set(names)), item.name


def test_code_style_csv_columns_carry_their_unit_in_their_name():
    """Unit in the name is how a column CSV says its unit; the VTK arrays keep
    their names and say it in field data instead."""
    missing = []
    for item in FILES:
        if ".csv" not in item.name:
            continue
        for column in item.columns:
            suffix = UNIT_SUFFIXES.get(column.unit)
            plain = re.sub(r"\{[^}]*\}|<[^>]*>", "x", column.name)
            if suffix is None or not CODE_STYLE.fullmatch(plain) or column.name in ("<metric>", "value"):
                continue
            if not (plain.endswith(suffix) or f"{suffix}_" in plain or plain.startswith(f"{suffix[1:]}_")):
                missing.append((item.name, column.name, column.unit))
    assert missing == []


def test_every_vtk_array_is_documented_and_its_unit_is_in_the_file(tmp_path):
    pv = pytest.importorskip("pyvista")
    from haemolynx.graph import FLOW_SOLVED, FLOW_STATE, tag_vessels_by_size
    from haemolynx.visualization import graph_to_vtk
    from haemolynx.visualization.vtk_io import ARRAY_UNITS, surface_mesh_to_vtk

    G = nx.MultiGraph(viscosity_law="pries", haematocrit=0.45)
    pos = {0: (0.0, 0.0, 0.0), 1: (200.0, 0.0, 0.0), 2: (200.0, 50.0, 0.0)}
    for n, p in pos.items():
        G.add_node(n, pos=np.asarray(p), pressure=100.0 - n)
    for u, v in ((0, 1), (1, 2)):
        G.add_edge(
            u, v, voxels=[pos[u], pos[v]], length=float(np.linalg.norm(np.subtract(pos[u], pos[v]))),
            branch_order="B01", diameter_um=5.0, diameter_source="fwhm", resistance=1e15,
            conductance=1e-15, pressure_u=100.0, pressure_v=99.0, pressure_drop=1.0,
            flow_signed=1e-15, flow_abs=1e-15, discharge_haematocrit=0.4,
            **{FLOW_SOLVED: True, FLOW_STATE: "solved"},
        )
    # A size convention adds its class to every vessel, and its thresholds
    # to the field data; a surround says which network each vessel is.
    tag_vessels_by_size(G, 6.0, 15.0)
    G.edges[0, 1, 0].update(network_part="image", image_branch_order="Art2")
    G.edges[1, 2, 0].update(network_part="synthetic")
    out = graph_to_vtk(G, tmp_path / "run")
    surface = surface_mesh_to_vtk(np.eye(3), np.array([[0, 1, 2]]), tmp_path / "run_tissue_surface.vtp")

    written = set()
    for file_name, path, arrays in (
        ("{stem}_vessels.vtp", out["vessels_path"], "cell_data"),
        ("{stem}_nodes.vtp", out["nodes_path"], "point_data"),
        ("{stem}_pericytes.vtp", out["pericytes_path"], "point_data"),
        ("{stem}_tissue_surface.vtp", surface, "point_data"),
    ):
        mesh = pv.read(str(path))
        names = list(getattr(mesh, arrays).keys())
        written |= set(names)
        assert undocumented(file_name, names) == [], file_name
        assert str(mesh.field_data["points_unit"][0]) == "um"
        assert str(mesh.field_data["points_axis_order"][0]) == "z, y, x"
        for name in names:
            assert str(mesh.field_data[f"{name}_unit"][0]) == ARRAY_UNITS[name], (file_name, name)
            documented = {c.name: c.unit for c in output_file(file_name).columns}
            assert documented[name] == ARRAY_UNITS[name]
    # This graph exercises every array the VTK writer has.
    assert written == set(ARRAY_UNITS)
    vessels = pv.read(out["vessels_path"])
    assert str(vessels.field_data["resistance_unit"][0]) == "Pa.s/m^3"
    assert list(vessels.cell_data["vessel_size_class"]) == ["narrow", "narrow"]
    assert float(vessels.field_data["vessel_size_mid_min_um"][0]) == 6.0
    assert float(vessels.field_data["vessel_size_wide_min_um"][0]) == 15.0
    assert list(vessels.cell_data["network_part"]) == ["image", "synthetic"]
    assert list(vessels.cell_data["image_branch_order"]) == ["Art2", ""]


@pytest.mark.parametrize(
    ("file_name", "columns", "whole"),
    [
        ("{name}_edges.csv", "haemolynx.pipeline.stages:PERTURBATION_EDGE_COLUMNS", True),
        ("{name}_summary.csv", "haemolynx.pipeline.stages:PERTURBATION_SUMMARY_COLUMNS", True),
        ("{name}_blocked_vessels.csv", "haemolynx.pipeline.stages:CAPILLARY_BLOCK_VESSEL_COLUMNS", True),
        ("{name}_flow_by_branch_order.csv", "haemolynx.pipeline.stages:CAPILLARY_BLOCK_ORDER_COLUMNS", True),
        ("{name}_mask_blocks.csv", "haemolynx.pipeline.stages:CAPILLARY_BLOCK_MASK_COLUMNS", True),
        ("{name}_mask_distribution.csv", "haemolynx.pipeline.stages:MASK_DISTRIBUTION_COLUMNS", False),
        ("{name}_mask_density_by_branch_order.csv",
         "haemolynx.pipeline.stages:MASK_DENSITY_ORDER_COLUMNS", True),
        ("{name}_mask_objects.csv", "haemolynx.pipeline.stages:MASK_OBJECT_COLUMNS", True),
        ("{name}_block_replicates.csv", "haemolynx.pipeline.stages:CAPILLARY_BLOCK_REPLICATE_COLUMNS", False),
        ("{stem}_connectivity.csv", "haemolynx.graph.connectivity:CONNECTIVITY_COLUMNS", True),
        ("{stem}_loop_review.csv", "haemolynx.gui.post_processing:LOOP_REVIEW_COLUMNS", True),
        ("{stem}_{kind}_review.csv", "haemolynx.gui.post_processing:REVIEW_CSV_COLUMNS", False),
        ("batch_summary.csv", "haemolynx.gui.batch_run:SUMMARY_COLUMNS", True),
        ("{stem}_template_comparison.csv", "haemolynx.synthetic.compare:CSV_COLUMNS", True),
        ("{stem}_image_flows.csv", "haemolynx.synthetic.surround:IMAGE_FLOW_COLUMNS", True),
        ("{stem}_pericytes.csv", "haemolynx.haemodynamics.pericyte_measurements:PERICYTE_COLUMNS", True),
        ("{stem}_pericyte_branches.csv", "haemolynx.haemodynamics.pericyte_measurements:BRANCH_COLUMNS", True),
        (
            "{stem}_pericyte_profiles/pericyte_<NNNN>.csv",
            "haemolynx.haemodynamics.pericyte_measurements:PROFILE_COLUMNS",
            True,
        ),
    ],
)
def test_the_documented_columns_are_the_writers_own(file_name, columns, whole):
    import importlib

    module, constant = columns.split(":")
    written = list(getattr(importlib.import_module(module), constant))
    documented = _names(file_name)
    if whole:
        assert documented == written
    else:
        assert documented[: len(written)] == written


def test_the_sweep_csvs_write_the_documented_headers(tmp_path):
    from haemolynx.haemodynamics.pericyte_geometry_sweep import (
        GEOMETRY_SWEEP_COLUMNS,
        write_geometry_sweep_csv,
    )
    from haemolynx.haemodynamics.pericyte_sweep import SWEEP_COLUMNS, write_sweep_csv

    point = {name: 1.0 for name in SWEEP_COLUMNS}
    path = write_sweep_csv([point], tmp_path / "pericyte_dilation_sweep.csv")
    sweep = next(f.name for f in FILES if "inlet_pressure_sweep.csv" in f.name)
    assert _header(path) == _names(sweep)
    with path.open(encoding="utf-8") as handle:
        (row,) = csv.DictReader(handle)
    assert row["total_inlet_flow_m3_s"] == "1.0"

    point = {name: 1.0 for name in GEOMETRY_SWEEP_COLUMNS}
    path = write_geometry_sweep_csv([point], tmp_path / "pericyte_spacing_sweep.csv")
    geometry = next(f.name for f in FILES if "pericyte_spacing_sweep.csv" in f.name)
    assert _header(path) == _names(geometry)


def test_the_cell_distance_csvs_write_the_documented_headers(tmp_path):
    from haemolynx.statistics.three_dim_distances import (
        export_object_distance_details_csv,
        export_object_distance_summary_csv,
    )

    details = export_object_distance_details_csv([], tmp_path / "d.csv")
    summary = export_object_distance_summary_csv([], tmp_path / "s.csv")
    assert _header(details) == _names("{stem}_cell_to_vessel_3d_dual_distances.csv")
    assert _header(summary) == _names("{stem}_cell_to_vessel_3d_dual_distances_summary.csv")
    assert not any("microns" in name for name in _header(details) + _header(summary))


def _review_measure_names() -> set[str]:
    """Every measure name a review list gives its items, read from the
    finders' source: the ``measures`` dicts and the keys set on one."""
    import haemolynx.graph.network_review as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()

    def keys(node: ast.AST) -> None:
        if isinstance(node, ast.Dict):
            names.update(k.value for k in node.keys if isinstance(k, ast.Constant))

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "measures":
            keys(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "measures":
                    keys(node.value)
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "measures" and isinstance(target.slice, ast.Constant)):
                    names.add(target.slice.value)
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if "id='measures'" in body:
                names.update(e.value for e in node.iter.elts if isinstance(e, ast.Constant))
    return names


def test_every_review_measure_is_documented():
    names = _review_measure_names()
    assert {"connector_um", "tip_radius_um", "gap_um", "volume_um3"} <= names
    assert undocumented("{stem}_{kind}_review.csv", sorted(names)) == []


def test_a_templated_name_matches_only_what_it_lists():
    assert undocumented(
        "{name}_block_comparison.csv",
        ["mean_transit_time_s_baseline", "capillary_red_cell_flux_cv_blocked",
         "total_inflow_percent_change_draws_high", "mean_transit_time_s_after"],
    ) == ["mean_transit_time_s_after"]
