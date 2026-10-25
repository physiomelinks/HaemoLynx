"""The template's layout -- junction shape, coverage, columns -- and its
capillaries' tortuosity tail: read, carried, compared, averaged and drawn to
(``statistics/layout.py``; Harvey, 2026-10-25, after grid D1d)."""
from __future__ import annotations

import copy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from haemolynx.statistics.layout import ACUTE_JUNCTION_DEG, junction_angle_triples
from haemolynx.synthetic import bed, clear_template_cache
from haemolynx.synthetic.average import average_templates
from haemolynx.synthetic.compare import LAYOUT_CHARACTERISTICS, compare_templates
from haemolynx.synthetic.generate import SYNTHETIC_CLASS, GenerationOptions, generate_network
from haemolynx.synthetic.template import (
    LAYOUT_ROWS,
    LAYOUT_SECTIONS,
    LayoutTarget,
    NetworkTemplate,
    read_layout,
    template_from_report,
)
from synthetic_fixtures import TINY_ROLES, fixture_template, tiny_hierarchical_graph, tiny_sections

#: E14.5's layout (its MCA run, 2026-10-25).
E14P5_LAYOUT = LayoutTarget(
    junction_smallest_angle_median_deg=97.1,
    junction_smallest_angle_p25_deg=86.8,
    junction_largest_angle_median_deg=138.8,
    acute_junction_share=0.012,
    junction_angle_sum_median_deg=356.3,
    box_within_20_share=0.506,
    box_beyond_50_share=0.005,
    distance_median_um=19.8,
    distance_p90_um=34.7,
    column_length_share=0.215,
    columns_per_mm2=781.6,
    column_spacing_median_um=13.6,
    column_clark_evans=0.90,
    column_tortuosity_median=1.08,
)


@pytest.fixture(autouse=True)
def _fresh():
    clear_template_cache()
    yield
    clear_template_cache()


def test_the_layout_is_the_reports_rows():
    sections, G = tiny_sections(), tiny_hierarchical_graph()
    layout = read_layout(sections, G)
    for field, (section, row) in LAYOUT_ROWS.items():
        value = sections[section][row]
        assert getattr(layout, field) == (value if isinstance(value, float) else getattr(layout, field))
    assert layout.junction_angle_sum_median_deg == sections["junction_shape"]["Junction Angle Sum Median (degrees)"]


def test_a_report_saved_before_the_layout_has_it_measured_off_the_graph():
    sections, G = tiny_sections(), tiny_hierarchical_graph()
    older = {name: rows for name, rows in sections.items() if name not in LAYOUT_SECTIONS}
    assert read_layout(older, G) == read_layout(sections, G)
    assert read_layout(older, None) is None


def _without_p90(sections):
    out = copy.deepcopy(sections)
    for item in out["vessel_classes"]["Vessel Classes"].values():
        if isinstance(item, dict):
            item.pop("90th Percentile Tortuosity Index", None)
    return out


def test_the_class_tortuosity_tail_is_the_reports_row_else_the_graphs():
    sections, G = tiny_sections(), tiny_hierarchical_graph()
    T = template_from_report(sections, G, boundaries=TINY_ROLES)
    row = sections["vessel_classes"]["Vessel Classes"]["Capillaries"]["90th Percentile Tortuosity Index"]
    assert T.classes["capillary"].p90_tortuosity == pytest.approx(row)
    older = template_from_report(_without_p90(sections), G, boundaries=TINY_ROLES)
    assert older.classes["capillary"].p90_tortuosity == pytest.approx(row)


def test_the_layout_and_the_tail_go_through_json_and_scaling():
    T = fixture_template()
    T = replace(T, layout=E14P5_LAYOUT, classes={**T.classes, "capillary": replace(T.classes["capillary"], p90_tortuosity=1.45)})
    again = NetworkTemplate.from_dict(T.to_dict())
    assert again.layout == E14P5_LAYOUT
    assert again.classes["capillary"].p90_tortuosity == 1.45
    assert T.scaled(1.5).layout == E14P5_LAYOUT
    assert NetworkTemplate.from_dict({**T.to_dict(), "layout": None}).layout is None


def test_a_batch_template_averages_its_runs_layouts():
    a = replace(fixture_template(), layout=E14P5_LAYOUT)
    b = replace(fixture_template(), layout=replace(E14P5_LAYOUT, acute_junction_share=0.052, distance_p90_um=44.7))
    average = average_templates([a, b])
    assert average.layout.acute_junction_share == pytest.approx(0.032)
    assert average.layout.distance_p90_um == pytest.approx(39.7)
    assert average.layout.column_clark_evans == pytest.approx(0.90)
    assert average_templates([fixture_template(), fixture_template()]).layout is None


def test_the_comparison_bands_every_layout_row_and_the_tail():
    T = replace(fixture_template(), layout=E14P5_LAYOUT)
    S = replace(fixture_template(), layout=replace(E14P5_LAYOUT, acute_junction_share=0.40, junction_smallest_angle_median_deg=53.0))
    rows = {(r.section, r.characteristic): r for r in compare_templates(T, S)}
    for characteristic, _field in LAYOUT_CHARACTERISTICS:
        assert rows[("layout", characteristic)].tolerance, characteristic
    assert rows[("layout", "junctions with two vessels within 45 degrees (share)")].within is False
    assert rows[("layout", "junction smallest angle median (degrees)")].within is False
    assert rows[("layout", "column spacing Clark-Evans ratio")].within is True
    assert ("class Capillaries", "tortuosity P90") in rows
    # A template without a layout has no layout rows.
    plain = {r.section for r in compare_templates(fixture_template(), fixture_template())}
    assert "layout" not in plain


def _acute_share(G) -> float:
    triples = np.array(junction_angle_triples(G))
    return float(np.mean(triples[:, 0] < ACUTE_JUNCTION_DEG))


def test_the_acute_weight_follows_the_templates_own_junctions():
    assert bed.acute_weight_for(0.0) == bed.ACUTE_WEIGHT
    assert bed.acute_weight_for(None) == bed.ACUTE_WEIGHT
    assert bed.acute_weight_for(bed.UNAIDED_ACUTE_SHARE / 2) == pytest.approx(bed.ACUTE_WEIGHT / 2)
    assert bed.acute_weight_for(0.6) == 0.0


def test_a_template_with_a_layout_draws_wider_junctions_and_the_same_counts():
    plain = fixture_template()
    shaped = replace(plain, layout=E14P5_LAYOUT)
    before, after = [], []
    for seed in (1, 2, 3):
        G0 = generate_network(plain, seed=seed)
        G1 = generate_network(shaped, seed=seed)
        before.append(_acute_share(G0))
        after.append(_acute_share(G1))
        # The trades keep every degree and the bed in one piece: the loops
        # are the same, the capillaries within the degree-2 merges' wobble.
        loops = [G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G) for G in (G0, G1)]
        assert loops[0] == loops[1]
        caps = [sum(1 for *_e, d in G.edges(data=True) if d.get(SYNTHETIC_CLASS) == "capillary") for G in (G0, G1)]
        assert abs(caps[0] - caps[1]) <= 2
    assert np.mean(after) < np.mean(before) - 0.08, (before, after)


def test_relaxing_trades_ends_to_widen_a_junction_and_keeps_every_degree():
    """v leaves to a and b almost together; c-d lies below. Trading ends
    (v-c, a-d) gives v a wide pair, at no vertex's degree."""
    points = np.array([[0, 0, 0], [0, 0, 10], [0, 1, 10], [0, 0, -10], [0, 1, -10], [0, -20, 0]], dtype=float)
    v, a, b, c, d, e = range(6)
    edges = ((v, a), (v, b), (c, d), (a, e), (e, c))
    candidates = {v: [(c, 10.0), (a, 10.0), (b, 10.0)], a: [(d, 20.0), (v, 10.0)], c: [(v, 10.0), (d, 1.0)], d: [(a, 20.0), (c, 1.0)], b: [(v, 10.0)]}

    def fill_of(**kw):
        fill = bed._Fill(points, np.full(6, 3), np.zeros(6, bool), np.random.default_rng(0), 80.0, 10.0, target_chord_um=15.0, **kw)
        for i, j in edges:
            fill.connect(i, j)
        return fill

    fill = fill_of(junction_angle_deg=120.0, acute_share=0.0)
    degrees = fill.degree.copy()
    assert fill.junction_cost(v)[1]
    assert fill.relax_junctions(candidates) == 1
    pairs = sorted((min(i, j), max(i, j)) for i, j in ((v, b), (v, c), (a, d), (a, e), (e, c)))
    assert sorted(fill.edges) == pairs
    assert list(fill.degree) == list(degrees)
    assert not fill.junction_cost(v)[1]
    # Without the template's junction angle, or for a template whose own
    # junctions are as acute as the fill draws unaided, nothing is relaxed.
    assert fill_of().relax_junctions(candidates) == 0
    assert fill_of(junction_angle_deg=120.0, acute_share=bed.UNAIDED_ACUTE_SHARE).relax_junctions(candidates) == 0


def test_a_trade_that_would_cut_the_bed_in_two_is_not_made():
    points = np.array([[0, 0, 0], [0, 0, 10], [0, 1, 10], [0, 0, -10], [0, 1, -10]], dtype=float)
    v, a, b, c, d = range(5)
    fill = bed._Fill(points, np.full(5, 3), np.zeros(5, bool), np.random.default_rng(0), 80.0, 10.0,
                     target_chord_um=15.0, junction_angle_deg=120.0, acute_share=0.0)
    for i, j in ((v, a), (v, b), (c, d), (b, c)):
        fill.connect(i, j)
    # v-a and c-d traded for v-c and a-d would leave a and d on their own.
    candidates = {v: [(c, 10.0)], a: [(d, 20.0)], c: [(v, 10.0)], d: [(a, 20.0)]}
    assert fill.relax_junctions(candidates) == 0
    assert (v, a) in fill.edges


@pytest.fixture(scope="module")
def run_template(tmp_path_factory):
    """A template read from a saved run (it has a layout of cells, which the
    lengths by tortuosity need)."""
    from haemolynx.synthetic import load_template
    from synthetic_fixtures import write_generated_run

    return load_template(write_generated_run(tmp_path_factory.mktemp("run")), "statistics")


def test_the_capillaries_tortuosity_tail_follows_the_templates_median_and_p90(run_template):
    """The tail's shape -- its 90th percentile's excess over straight against
    the median's -- is the template's; the class's total length sets the
    scale, so a longer tail with the same length bends the rest less."""

    def tail(template, seed):
        G = generate_network(template, seed=seed, options=GenerationOptions())
        values = []
        for _u, _v, d in G.edges(data=True):
            if d.get(SYNTHETIC_CLASS) != "capillary":
                continue
            p = np.asarray(d["voxels"], float)
            chord = float(np.linalg.norm(p[-1] - p[0]))
            if chord > 0:
                values.append(d["length"] / chord)
        median, p90 = np.percentile(values, [50, 90])
        return (p90 - 1.0) / (median - 1.0)

    capillary = run_template.classes["capillary"]
    assert capillary.median_tortuosity is not None and capillary.p90_tortuosity is not None
    narrow = replace(run_template, classes={**run_template.classes, "capillary": replace(capillary, p90_tortuosity=None)})
    winding = replace(run_template, classes={
        **run_template.classes, "capillary": replace(capillary, p90_tortuosity=capillary.median_tortuosity + 0.5),
    })
    assert np.mean([tail(winding, s) for s in (1, 2)]) > 1.2 * np.mean([tail(narrow, s) for s in (1, 2)])
