"""The template-versus-synthetic comparison (``synthetic/compare.py``)."""
from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from haemolynx.synthetic import compare
from haemolynx.synthetic.compare import (
    CSV_COLUMNS,
    TOLERANCES,
    ComparisonRow,
    compare_run,
    compare_templates,
    comparison_section,
    comparison_summary,
    row,
    write_template_comparison_csv,
)
from haemolynx.synthetic.generate import GenerationOptions, boundary_nodes, generate_network
from haemolynx.synthetic.template import measure_template
from synthetic_fixtures import fixture_template, tiny_hierarchical_graph, tiny_sections, TINY_ROLES


def test_a_row_carries_ratio_tolerance_and_verdict():
    r = row("loops", "independent loops", 50, 51, ("abs", 2.0))
    assert (r.ratio, r.tolerance, r.within) == (pytest.approx(1.02), "abs 2", True)
    assert row("x", "y", 100.0, 104.0, ("rel", 0.05)).within is True
    assert row("x", "y", 100.0, 106.0, ("rel", 0.05)).within is False
    assert row("x", "y", 3, 3, ("exact", 0.0)).within is True
    assert row("x", "y", 3, 4, ("exact", 0.0)).within is False
    assert row("x", "y", 3, 3).tolerance == "" and row("x", "y", 3, 3).within is None


def test_a_missing_or_not_available_value_has_no_ratio_or_verdict():
    r = row("x", "y", None, 4.0, ("rel", 0.1))
    assert r.ratio is None and r.within is None and "N/A" in r.template
    r = row("x", "y", "N/A (no junction)", 4.0, ("rel", 0.1))
    assert r.ratio is None and r.within is None
    assert row("x", "y", 0.0, 0.0, ("rel", 0.1)).within is True
    assert row("x", "y", 0.0, 1.0, ("rel", 0.1)).within is False


def test_comparing_a_template_with_itself_is_within_everywhere():
    T = fixture_template()
    rows = compare_templates(T, T)
    assert all(r.within in (True, None) for r in rows)
    assert all(r.ratio in (None, pytest.approx(1.0)) for r in rows)
    sections = {r.section for r in rows}
    assert {"box", "endpoints", "network", "class Capillaries", "order Art1", "loops", "dead ends"} <= sections
    summary = comparison_summary(rows)
    assert summary["within"] == summary["fitted"] > 30 and summary["outside"] == []


def test_the_exact_rows_are_the_constructed_ones():
    rows = compare_templates(fixture_template(), fixture_template())
    exact = {(r.section, r.characteristic) for r in rows if r.tolerance == "exact"}
    assert ("endpoints", "inlets") in exact
    assert ("order Art1", "vessels") in exact
    assert ("class Small Arterioles", "vessels") in exact
    assert ("class Capillaries", "vessels") not in exact
    banded = {r.characteristic for r in rows if r.tolerance.startswith(("rel", "abs"))}
    assert {"independent loops", "daughter angle (degrees)", "mean tortuosity", "total length (um)"} <= banded
    unbanded = {r.characteristic for r in rows if r.tolerance == ""}
    assert "Murray exponent, median fitted" in unbanded
    assert "mean emergence angle (degrees)" in unbanded


def test_a_changed_template_shows_as_outside():
    T = fixture_template()
    S = replace(T, daughter_angle_deg=95.0, loops=replace(T.loops, independent_loops=60))
    rows = compare_templates(T, S)
    outside = {r.characteristic for r in rows if r.within is False}
    assert outside == {"daughter angle (degrees)", "independent loops", "independent loops per mm^3"} or outside == {"daughter angle (degrees)", "independent loops"}
    assert comparison_summary(rows)["outside"]


def test_the_csv_has_the_columns_and_one_row_each(tmp_path):
    rows = compare_templates(fixture_template(), fixture_template())
    path = write_template_comparison_csv(rows, tmp_path / "out" / "stack_synthetic_template_comparison.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        read = list(csv.reader(handle))
    assert tuple(read[0]) == CSV_COLUMNS
    assert len(read) == len(rows) + 1
    by_name = {(r[0], r[1]): r for r in read[1:]}
    inlets = by_name[("endpoints", "inlets")]
    assert inlets[2] == "2" and inlets[3] == "2" and inlets[4] == "1" and inlets[5] == "exact" and inlets[6] == "yes"
    murray = by_name[("network", "Murray exponent, median fitted")]
    assert murray[5] == "" and murray[6] == ""


def test_the_report_section_reads_as_text():
    rows = compare_templates(fixture_template(), fixture_template())
    section = comparison_section(rows)
    assert section["endpoints: inlets"].startswith("2 -> 2")
    assert "within exact" in section["endpoints: inlets"]
    assert section["fitted characteristics within tolerance"].endswith(f"of {comparison_summary(rows)['fitted']}")


def test_compare_run_reads_the_runs_own_report_or_measures_the_graph():
    G = tiny_hierarchical_graph()
    from haemolynx.graph.branch_order import assign_vessel_branch_orders  # noqa: F401  (the tiny graph is labelled already)

    T = measure_template(
        G,
        inlet_nodes=TINY_ROLES["inlet"],
        outlet_nodes=TINY_ROLES["outlet"],
        arteriole_boundary_nodes=TINY_ROLES["arteriole_boundary"],
        venule_boundary_nodes=TINY_ROLES["venule_boundary"],
        large_arteriole_boundary_nodes=TINY_ROLES["large_arteriole_boundary"],
        large_venule_boundary_nodes=TINY_ROLES["large_venule_boundary"],
        source="statistics",
    )
    roles = {role: list(nodes) for role, nodes in TINY_ROLES.items()}
    from_report = compare_run(T, G, roles=roles, sections=tiny_sections(G))
    measured = compare_run(T, G, roles=roles)
    assert [r.characteristic for r in from_report] == [r.characteristic for r in measured]
    assert all(r.within in (True, None) for r in from_report)
    assert all(r.within in (True, None) for r in measured)
    graph_source = replace(T, provenance=replace(T.provenance, source="graph"))
    assert all(r.within in (True, None) for r in compare_run(graph_source, G, roles=roles, sections=tiny_sections(G)))


def test_compare_run_on_a_generated_network_scores_its_rows():
    T = fixture_template()
    G = generate_network(T, seed=5)
    roles = boundary_nodes(G)
    from haemolynx.graph.branch_order import assign_vessel_branch_orders

    assign_vessel_branch_orders(G, roles["inlet"], outlet_nodes=roles["outlet"], arteriole_boundary_nodes=roles["arteriole_boundary"], venule_boundary_nodes=roles["venule_boundary"])
    rows = compare_run(T, G, roles=roles)
    assert all(isinstance(r, ComparisonRow) for r in rows)
    exact = [r for r in rows if r.tolerance == "exact"]
    assert exact and all(r.within is not False for r in exact)


def test_the_diameter_basis_row_compares_the_names():
    T = replace(fixture_template(), diameter_basis="endothelial")

    def basis_row(template, synthetic):
        return next(r for r in compare_templates(template, synthetic) if r.characteristic == "diameter basis")

    r = basis_row(T, T)
    assert (r.section, r.template, r.synthetic, r.ratio, r.tolerance, r.within) == (
        "network", "endothelial", "endothelial", None, "exact", True,
    )
    assert basis_row(T, replace(T, diameter_basis="plasma_column")).within is False
    assert basis_row(fixture_template(), fixture_template()).within is None  # a template naming none


class _Recording(dict):
    def __init__(self, data):
        super().__init__(data)
        self.read: set = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def test_every_band_in_the_table_is_one_a_row_reads(monkeypatch):
    table = _Recording(TOLERANCES)
    monkeypatch.setattr(compare, "TOLERANCES", table)
    T = fixture_template()
    # The template's open ends: a closed network reports the leaving-the-image row, unbanded.
    G = generate_network(T, seed=5, options=GenerationOptions(open_ends="template"))
    roles = boundary_nodes(G)
    from haemolynx.graph.branch_order import assign_vessel_branch_orders

    assign_vessel_branch_orders(G, roles["inlet"], outlet_nodes=roles["outlet"], arteriole_boundary_nodes=roles["arteriole_boundary"], venule_boundary_nodes=roles["venule_boundary"])
    rows = compare_run(T, G, roles=roles)
    assert set(TOLERANCES) <= table.read, set(TOLERANCES) - table.read
    # The capillary orders the bed makes are reported, never banded.
    assert all(r.tolerance == "" for r in rows if r.section.startswith("order B"))
    assert any(r.section.startswith("order B") for r in rows)


def test_a_closed_network_reports_the_templates_open_ends_as_not_drawn():
    T = fixture_template()
    closed = {"open_ends": "closed"}
    rows = {r.characteristic: r for r in compare_templates(T, T, options=closed) if r.section == "dead ends"}
    assert rows["leaving the image"].within is None
    assert rows["leaving the image"].tolerance == "not drawn (a closed network)"
    assert rows["dead ends"].within is None  # its total holds the kind not drawn
    assert rows["inside the tissue"].within is True
    rows = {r.characteristic: r for r in compare_templates(T, T, options={"open_ends": "template"}) if r.section == "dead ends"}
    assert rows["leaving the image"].within is True and rows["dead ends"].within is True
    section = comparison_section(compare_templates(T, T, options=closed))
    assert section["dead ends: leaving the image"].endswith("not drawn (a closed network)")


def test_the_dead_end_total_is_unbanded_when_the_template_has_kinds_a_network_cannot_draw():
    """E14.5's template counts 17 artefact ends and 94 ends of pieces without
    a junction; one connected network draws neither, so its total was marked
    outside while the kinds drawn (leaving the image, inside) were exact."""
    T = fixture_template()
    with_islands = replace(T, dead_ends=replace(T.dead_ends, total=T.dead_ends.total + 5, island_ends=3, artefacts=2))

    def by_name(template, synthetic):
        return {r.characteristic: r for r in compare_templates(template, synthetic) if r.section == "dead ends"}

    rows = by_name(with_islands, T)
    assert rows["dead ends"].tolerance == "" and rows["dead ends"].within is None
    assert rows["leaving the image"].within is True and rows["inside the tissue"].within is True
    rows = by_name(T, T)
    assert rows["dead ends"].tolerance == "abs 2" and rows["dead ends"].within is True
