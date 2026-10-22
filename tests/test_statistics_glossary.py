"""The statistics glossary (``statistics/glossary.py``): every row of a run's
statistics files has a definition and its unit, each unit spelled one way,
the units a run decides read from the run's own rows, and older CSVs read
back to the same metrics and units."""
from __future__ import annotations

import csv
import dataclasses
import math
import re
from pathlib import Path

import numpy as np
import pytest

from haemolynx.pipeline import default_schema, resolve_settings, run_pipeline_stages
from haemolynx.pipeline.naming import run_stem
from haemolynx.pipeline.stages import HAEMODYNAMICS_SECTION, STATISTICS_REPORT
from haemolynx.statistics import (
    STATISTIC_MEASURES,
    ZERO_RESISTANCE_BRIDGES,
    capillary_bed_joins,
    export_statistics_to_csv,
    glossary,
)
from haemolynx.statistics.batch_comparison import read_branch_statistics_csv, read_statistics_csv
from haemolynx.statistics.bifurcation import export_branch_order_statistics_to_csv
from haemolynx.statistics.tissue_volume import TissueVolume
from haemolynx.synthetic import clear_template_cache
from synthetic_fixtures import write_generated_run

SCHEMA = default_schema()
REPO = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_sections(report: dict):
    """The report's sections the statistics CSV writes: every measure's, the
    zero-resistance bridges' and the resistances (the branch orders, the
    weighted measures and the synthetic comparison go to files of their own)."""
    written = (*STATISTIC_MEASURES, ZERO_RESISTANCE_BRIDGES, HAEMODYNAMICS_SECTION)
    sections = [(key, rows) for key, rows in report["sections"].items() if key in written]
    assert len(sections) > 25, list(report["sections"])
    return sections


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


# -- a run with every statistic on ---------------------------------------------


@pytest.fixture(scope="module", params=["fast", "full"])
def every_statistic(request, tmp_path_factory):
    """A synthetic network solved and measured with every statistics_*
    toggle on, in one statistics mode: its statistics CSVs and report."""
    clear_template_cache()
    out = tmp_path_factory.mktemp(f"glossary_{request.param}")
    template = write_generated_run(out / "template")
    values = {setting.name: setting.default for setting in SCHEMA}
    for setting in SCHEMA:
        if setting.name.startswith("statistics_") and isinstance(setting.default, bool):
            values[setting.name] = True
    values.update(
        {
            "network_source": "synthetic",
            "synthetic_template_path": template,
            "synthetic_seed": 3,
            "vtk_output_prefix": out / "out" / "run",
            "plot_dir": out / "plots",
            "statistics": True,
            "statistics_mode": request.param,
            "show_plots_in_ide": False,
            "interactive_plots": False,
            "visualize_results": False,
        }
    )
    settings = resolve_settings(values, schema=SCHEMA, config_path=None)
    G = run_pipeline_stages(settings, SCHEMA)
    clear_template_cache()
    stem = run_stem(settings)
    return out / "out", stem, G.graph[STATISTICS_REPORT]


def test_every_row_has_a_definition_and_every_number_a_unit(every_statistic):
    folder, stem, _report = every_statistic
    for name in (f"{stem}_statistics.csv", f"{stem}_statistics_connected.csv"):
        rows = [row for row in _rows(folder / name) if row["Section"] != "Metadata"]
        assert len(rows) > 150, name
        for row in rows:
            what = (name, row["Section"], row["Metric"], row["Value"][:40])
            assert row["Notes"].replace("Not computable with current inputs.", "").strip(), what
            if _is_number(row["Value"]):
                assert row["Unit"] in glossary.UNITS, what
            else:
                assert row["Unit"] in ("", *glossary.UNITS), what


def test_every_row_name_has_a_term(every_statistic):
    _folder, _stem, report = every_statistic

    def names(rows):
        """The rows' names; a nested dict is a section, not a row."""
        for name, value in rows.items():
            if isinstance(value, dict):
                yield from names(value)
            else:
                yield str(name)

    checked = 0
    for key, section in _csv_sections(report):
        for name in names(section):
            assert glossary.term_for(name) is not None, (key, name)
            checked += 1
    assert checked > 150


def test_a_unit_the_name_spells_is_the_terms_unit(every_statistic):
    """The name and the glossary never disagree: a name's own "(microns)" is
    the term's um, and so on."""
    _folder, _stem, report = every_statistic
    checked = 0
    for _key, section in _csv_sections(report):
        for name in section:
            _shown, in_name = glossary.split_name(str(name))
            term = glossary.term_for(str(name))
            if in_name is None or isinstance(term.unit, (glossary.ByWeighting, glossary.FromRow)):
                continue
            assert glossary.unit_for(str(name), section) == in_name, name
            checked += 1
    assert checked > 30


def test_the_run_decides_the_route_units(every_statistic):
    folder, stem, _report = every_statistic
    rows = {row["Metric"]: row for row in _rows(folder / f"{stem}_statistics.csv")}
    assert rows["Shunt Route Weighting"]["Value"] == "length"
    assert rows["Median Inlet-Outlet Route"]["Unit"] == "um"
    assert rows["Equivalent Inlet-Outlet Resistance"]["Unit"] == "um"
    assert rows["Algebraic Connectivity lambda_2"]["Unit"] == "1/um"
    assert rows["Bottleneck Min-Cut Capacity"]["Unit"] == "vessels"
    assert rows["Loop Length Median"]["Unit"] == "um"


def test_the_two_fractal_dimensions_keep_their_names(every_statistic):
    """Both used to be written "Fractal Dimension", with the units "Node
    Positions" and "Centreline"."""
    folder, stem, _report = every_statistic
    rows = {row["Metric"]: row for row in _rows(folder / f"{stem}_statistics.csv")}
    for which in ("Node Positions", "Centreline"):
        assert rows[f"Fractal Dimension ({which})"]["Unit"] == "dimensionless"
        assert rows[f"Fractal Dimension ({which}) Box Sizes"]["Unit"] == "um"
    assert "Fractal Dimension" not in rows


def test_the_bed_table_columns_have_terms(every_statistic):
    folder, stem, _report = every_statistic
    path = folder / f"{stem}_capillary_bed_joins.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == capillary_bed_joins.BED_TABLE_COLUMNS
    for column in header:
        assert glossary.term_for(column).topic.startswith("Capillary bed table"), column


# -- the units ------------------------------------------------------------------


def test_every_term_unit_is_one_of_the_units():
    for term in glossary.GLOSSARY:
        units = (
            [term.unit.resistance, term.unit.length, term.unit.topology]
            if isinstance(term.unit, glossary.ByWeighting)
            else [] if isinstance(term.unit, glossary.FromRow) or term.unit in ("", glossary.UNIT_IN_NAME)
            else [term.unit]
        )
        for unit in units:
            assert unit in glossary.UNITS, term.shown


def test_every_spelling_is_of_one_unit():
    for spelling, unit in glossary.UNIT_SPELLINGS.items():
        assert unit in glossary.UNITS, spelling
    assert all(re.fullmatch(r"[ -~]+", unit) for unit in glossary.UNITS), "units are ASCII"
    assert glossary.canonical_unit("microns") == glossary.canonical_unit("micron") == "um"
    assert glossary.canonical_unit("micron³") == "um^3"
    assert glossary.canonical_unit("Node Positions") is None


def test_no_term_is_hidden_by_an_earlier_one():
    """A literal term must be the one its own name finds; a pattern placed
    after a broader one would never be reached."""
    for term in glossary.GLOSSARY:
        if term.pattern == re.escape(term.shown):
            assert glossary.term_for(term.shown) is term, term.shown


def test_parentheses_that_are_not_a_unit_stay_in_the_name():
    assert glossary.split_name("Fractal Dimension (Node Positions)") == ("Fractal Dimension (Node Positions)", None)
    assert glossary.split_name("Blind Ends (marked by hand)") == ("Blind Ends (marked by hand)", None)
    assert glossary.split_name("Total Edge Length (microns)") == ("Total Edge Length", "um")
    assert glossary.split_name("Loop Length P90 (vessels)") == ("Loop Length P90", "vessels")
    assert glossary.unit_for("Blind Ends (marked by hand)") == "count"


@pytest.mark.parametrize(
    "weighting, unit",
    [("resistance", "Pa.s/m^3"), ("length", "um"), ("topology", "vessels"), ("topology (length unavailable)", "vessels")],
)
def test_a_route_length_takes_the_unit_of_its_weighting(weighting, unit):
    rows = {"Shunt Route Weighting": weighting}
    assert glossary.unit_for("Median Inlet-Outlet Route", rows) == unit
    assert glossary.unit_for("Shortest Inlet-Outlet Route", rows) == unit


def test_a_unit_row_gives_its_rows_unit():
    assert glossary.unit_for(
        "Bottleneck Min-Cut Capacity", {"Bottleneck Min-Cut Capacity Unit": "conductance (m^3/(Pa.s))"}
    ) == "m^3/(Pa.s)"
    resistance_rows = {"Equivalent Inlet-Outlet Resistance Unit": "vessels (unit resistance per vessel)"}
    assert glossary.unit_for("Kirchhoff Index", resistance_rows) == "vessels"
    assert glossary.unit_for("Max Hypoperfused Length", {"Occlusion Hypoperfused Length Unit": "um"}) == "um"
    # Without the row it says nothing rather than guess.
    assert glossary.unit_for("Median Inlet-Outlet Route") == ""


def test_the_tissue_rows_have_terms_and_units():
    values = {field.name: 1.0 for field in dataclasses.fields(TissueVolume)}
    values.update(
        threshold_method="otsu",
        working_voxel_size_zyx=(4.0, 4.0, 4.0),
        component_count=1,
        vertices=np.zeros((0, 3)),
        faces=np.zeros((0, 3), dtype=int),
    )
    rows = TissueVolume(**values).rows()
    for name, value in rows.items():
        term = glossary.term_for(name)
        assert term is not None, name
        if isinstance(value, (int, float)):
            assert glossary.unit_for(name, rows) in glossary.UNITS, name


def test_the_branch_statistics_columns_have_terms_and_one_spelling(tmp_path):
    path = export_branch_order_statistics_to_csv(
        {"BO1": {"Edge Count": 2, "Mean Length (microns)": 3.0, "Mean Diameter (microns)": 4.0}},
        tmp_path / "branch.csv",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    for column in header[:-1]:
        assert glossary.term_for(column) is not None, column
        unit = glossary.split_name(column)[1]
        if "(" in column:
            assert unit in glossary.UNITS, column
    assert "Mean Length (um)" in header and "Mean Diameter (um)" in header
    values = read_branch_statistics_csv(path)
    assert values[("Branch Order BO1", "Mean Length", "um")] == 3.0


# -- older CSVs ------------------------------------------------------------------


def test_an_older_csv_reads_back_as_the_same_metrics_and_units(tmp_path):
    """A batch may mix runs from before every row had its unit with runs
    after: both must give the same (section, metric, unit) keys."""
    stats = {
        "Total Edge Length (microns)": 340.5,
        "Fractal Dimension (Node Positions)": 1.5,
        "Fractal Dimension (Centreline)": 1.7,
        "Blind Ends (marked by hand)": 3,
        "Path Efficiency Pair Coverage": 0.25,
        "Shunt Route Weighting": "topology",
        "Median Inlet-Outlet Route": 12.0,
        "Vessel Density in Tissue (microns/micron³)": 0.002,
    }
    new = read_statistics_csv(export_statistics_to_csv(stats, tmp_path / "new.csv"))
    old = tmp_path / "old.csv"
    old.write_text(
        "Section,Metric,Value,Unit,Notes\n"
        "Core Statistics,Total Edge Length,340.5,microns,\n"
        "Core Statistics,Fractal Dimension,1.5,Node Positions,\n"
        "Core Statistics,Fractal Dimension,1.7,Centreline,\n"
        "Core Statistics,Blind Ends,3,marked by hand,\n"
        "Core Statistics,Path Efficiency Pair Coverage,25.00%,,\n"
        "Core Statistics,Shunt Route Weighting,topology,,\n"
        "Core Statistics,Median Inlet-Outlet Route,12,,\n"
        "Core Statistics,Vessel Density in Tissue,0.002,microns/micron³,\n",
        encoding="utf-8",
    )
    assert read_statistics_csv(old) == new
    assert new == {
        ("Core Statistics", "Total Edge Length", "um"): 340.5,
        ("Core Statistics", "Fractal Dimension (Node Positions)", "dimensionless"): 1.5,
        ("Core Statistics", "Fractal Dimension (Centreline)", "dimensionless"): 1.7,
        ("Core Statistics", "Blind Ends (marked by hand)", "count"): 3.0,
        ("Core Statistics", "Path Efficiency Pair Coverage", "fraction"): pytest.approx(0.25),
        ("Core Statistics", "Median Inlet-Outlet Route", "vessels"): 12.0,
        ("Core Statistics", "Vessel Density in Tissue", "um/um^3"): 0.002,
    }


# -- the page --------------------------------------------------------------------


def test_the_glossary_page_is_up_to_date():
    page = REPO / glossary.GLOSSARY_PAGE
    assert page.read_text(encoding="utf-8") == glossary.glossary_markdown(), (
        "Regenerate it: python -m haemolynx.statistics.glossary"
    )


def test_the_page_lists_every_term_and_unit():
    page = glossary.glossary_markdown()
    for unit in glossary.UNITS:
        assert f"| `{unit}` |" in page
    for term in glossary.GLOSSARY:
        assert f"| {term.shown} |" in page
