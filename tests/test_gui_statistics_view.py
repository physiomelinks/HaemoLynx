"""What the left panel's Statistics tab shows: a button per section of the
last run's statistics report, and the rows behind each (no Qt)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui.statistics_view import (
    MAX_LIST_ITEMS,
    MAX_ROWS,
    OTHER_SECTIONS,
    StatisticRow,
    section_title,
    section_tooltip,
    statistics_report,
    statistics_sections,
    statistics_summary,
    value_text,
)
from haemolynx.pipeline.stages import (
    BRANCH_ORDER_SECTION,
    HAEMODYNAMICS_SECTION,
    STATISTICS_REPORT,
    TISSUE_VOLUME_SECTION,
    WEIGHTED_MEASURES_SECTION,
)
from haemolynx.statistics import ZERO_RESISTANCE_BRIDGES

TAB = "9. Additional measurements"


def _report(sections, mode="fast", csv="out/run_statistics.csv"):
    return {"mode": mode, "csv": csv, "sections": sections}


def test_a_section_per_measure_in_the_order_the_measures_tab_lists_them_then_the_others():
    # Out of order on purpose: the report is in the order things ran.
    report = _report(
        {
            TISSUE_VOLUME_SECTION: {"Tissue Volume (micron³)": 1.0},
            "murray_law": {"Mean Murray Ratio": 1.0},
            ZERO_RESISTANCE_BRIDGES: {"Zero-resistance Bridges (not counted as vessels)": 2},
            HAEMODYNAMICS_SECTION: {"Two-Point Equivalent Resistance (Pa.s/m^3)": 3.0},
            "basic": {"Total Nodes": 7},
            BRANCH_ORDER_SECTION: {},
            WEIGHTED_MEASURES_SECTION: {},
            "k_core": {"Max Core Number": 2},
        }
    )

    sections = statistics_sections(report)

    assert [s.key for s in sections] == [
        "basic",
        "murray_law",
        "k_core",
        HAEMODYNAMICS_SECTION,
        BRANCH_ORDER_SECTION,
        WEIGHTED_MEASURES_SECTION,
        ZERO_RESISTANCE_BRIDGES,
        TISSUE_VOLUME_SECTION,
    ]
    assert [s.title for s in sections] == [
        "Basic",
        "Murray law",
        "K core",
        "Haemodynamics",
        "Branch orders",
        "Weighted betweenness and communities",
        "Zero-resistance bridges",
        "Tissue volume",
    ]


def test_a_measures_title_is_its_checkbox_name_without_statistics():
    """The checkbox on the Additional measurements tab reads "Statistics tree
    asymmetry"; its button reads "Tree asymmetry"."""
    from haemolynx.gui.form import label_for
    from haemolynx.statistics import STATISTIC_MEASURES

    for measure in STATISTIC_MEASURES:
        assert label_for(f"statistics_{measure}") == f"Statistics {section_title(measure).lower()}"


def test_every_section_has_its_own_tooltip_explaining_its_statistic():
    """Every measure and every other section has an entry of its own, none
    shared and none for a section that does not exist, in the tooltips'
    voice (a phrase of a few words, no closing full stop)."""
    from haemolynx.gui.chrome_tooltips import (
        STATISTICS_SECTION_TOOLTIP,
        STATISTICS_SECTION_TOOLTIPS,
    )
    from haemolynx.statistics import STATISTIC_MEASURES

    keys = (*STATISTIC_MEASURES, *OTHER_SECTIONS)
    assert set(STATISTICS_SECTION_TOOLTIPS) == set(keys)
    tooltips = [section_tooltip(key) for key in keys]
    assert len(set(tooltips)) == len(keys)
    assert STATISTICS_SECTION_TOOLTIP not in tooltips
    for key, text in zip(keys, tooltips):
        assert len(text.split()) >= 8, key
        assert not text.rstrip().endswith("."), key


def test_a_section_without_a_tooltip_says_what_its_button_does():
    from haemolynx.gui.chrome_tooltips import STATISTICS_SECTION_TOOLTIP

    assert section_tooltip("a_measure_added_later") == STATISTICS_SECTION_TOOLTIP


def test_no_report_no_buttons_and_a_note_saying_where_statistics_are_turned_on():
    for report in (None, {}, _report({})):
        assert statistics_sections(report) == ()
        note = statistics_summary(report, TAB)
        assert note.startswith("No statistics yet")
        assert TAB in note


def test_the_summary_counts_the_results_and_says_the_mode_and_the_csv():
    report = _report({"basic": {"Total Nodes": 7}, "k_core": {"Max Core Number": 2}}, mode="full")
    note = statistics_summary(report, TAB)
    assert note.startswith("2 results from the last run in full mode.")
    assert note.endswith("Saved to out/run_statistics.csv")

    tissue_only = {"mode": None, "csv": None, "sections": {TISSUE_VOLUME_SECTION: {"a": 1.0}}}
    note = statistics_summary(tissue_only, TAB)
    assert note == "1 result from the last run. Press one to show it."


def test_rows_are_the_sections_values_and_nested_ones_sit_under_a_heading():
    (section,) = statistics_sections(
        _report(
            {
                "dead_ends": {
                    "Dead Ends": {"Dead Ends": 3, "Fraction": 0.123456789},
                    "Dead End Kinds": {},
                    "Statistics Note": "N/A (no mask)",
                }
            }
        )
    )

    assert section.rows == (
        StatisticRow("Dead Ends", heading=True),
        StatisticRow("Dead Ends", "3", depth=1),
        StatisticRow("Fraction", "0.123457", depth=1),
        StatisticRow("Dead End Kinds", heading=True),
        StatisticRow("None", depth=1),
        StatisticRow("Statistics Note", "N/A (no mask)"),
    )


def test_a_section_with_nothing_in_it_says_so():
    (section,) = statistics_sections(_report({"flow_hierarchy": {}}))
    assert section.rows == (StatisticRow("Nothing to report"),)


def test_each_branch_order_is_a_heading_over_its_record_without_repeating_its_name():
    (section,) = statistics_sections(
        _report(
            {
                BRANCH_ORDER_SECTION: {
                    "Art1": {"Branch Order": "Art1", "Edge Count": 4},
                    "BO2": {"Branch Order": "BO2", "Edge Count": 9},
                }
            }
        )
    )
    assert section.rows == (
        StatisticRow("Art1", heading=True),
        StatisticRow("Edge Count", "4", depth=1),
        StatisticRow("BO2", heading=True),
        StatisticRow("Edge Count", "9", depth=1),
    )


def test_the_weighted_measures_headings_say_what_they_are_weighted_by():
    (section,) = statistics_sections(
        _report(
            {
                WEIGHTED_MEASURES_SECTION: {
                    "edge_resistance": {"Betweenness": {"Betweenness Mean": 0.5}},
                    "edge_length": {"Betweenness": {"Betweenness Mean": 0.25}},
                    "edge_flow_abs": {"Betweenness": {"Betweenness Mean": "N/A (no flow)"}},
                }
            }
        )
    )
    headings = [row.name for row in section.rows if row.heading and row.depth == 0]
    assert headings == ["Weighted by resistance", "Weighted by length", "Weighted by |flow|"]


def test_full_mode_betweenness_is_a_row_per_node_cut_short_past_the_limit():
    nodes = MAX_ROWS + 25
    (section,) = statistics_sections(
        _report({"betweenness": {node: node / nodes for node in range(nodes)}}, mode="full")
    )
    assert section.rows[0] == StatisticRow("Node 0", "0")
    assert section.rows[1].name == "Node 1"
    assert len(section.rows) == MAX_ROWS + 1
    assert section.rows[-1] == StatisticRow("… and 25 more in the exported files")


@pytest.mark.parametrize(
    "value, text",
    [
        (7, "7"),
        (np.int64(7), "7"),
        (2.5e15, "2.5e+15"),
        (np.float32(0.25), "0.25"),
        (1 / 3, "0.333333"),
        (True, "True"),
        (np.bool_(False), "False"),
        (None, "None"),
        ("N/A (no flow solved)", "N/A (no flow solved)"),
        ([1, 2.5, "a"], "[1, 2.5, a]"),
        ((3, 1), "[3, 1]"),
        ({3, 1, 2}, "{1, 2, 3}"),
        (np.array([1.0, 2.0]), "[1, 2]"),
        ({"a": 1}, "{a: 1}"),
    ],
)
def test_a_value_reads_as_the_statistics_csv_writes_numbers(value, text):
    assert value_text(value) == text


def test_a_long_list_is_cut_short_and_counted():
    many = list(range(MAX_LIST_ITEMS + 8))
    shown = ", ".join(str(item) for item in range(MAX_LIST_ITEMS))
    assert value_text(many) == f"[{shown}, …] ({len(many)} items)"

    # Full mode's communities: each one cut short, and the list too.
    three = [set(range(i * 50, i * 50 + 50)) for i in range(3)]
    assert value_text(three).count("(50 items)") == 3
    assert value_text(three).endswith("(50 items)]")
    thirty = [set(range(i * 50, i * 50 + 50)) for i in range(30)]
    assert value_text(thirty).startswith("[{0, 1, 2")
    assert value_text(thirty).endswith(", …] (30 items)")


def test_the_report_is_read_from_the_graph_the_export_left_it_on():
    G = nx.MultiGraph()
    assert statistics_report(G) is None
    assert statistics_report(None) is None
    report = _report({"basic": {"Total Nodes": 1}})
    G.graph[STATISTICS_REPORT] = report
    assert statistics_report(G) is report
