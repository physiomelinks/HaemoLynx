"""The cut-out harness (``scripts/surround_check.py``, ``PLAN_synthetic_surround.md``
phase 1): its scores on hand-set flows, where it takes the image's box from,
and one whole case, whose oracle -- the truth's own pressures at the ends --
must give the truth back."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import surround_check as sc
from haemolynx.synthetic.surround import ARTERIAL, VENOUS, ImageOpenEnd


def test_flow_scores_count_the_vessels_above_the_floor():
    truth = [10.0, -10.0, 5.0, 0.1]
    model = [10.0, 10.0, 2.5, -7.0]  # right; reversed; half; under the floor
    scores = sc.flow_scores(truth, model, floor=1.0)
    assert scores["vessels"] == 3
    assert scores["direction_agreement"] == pytest.approx(2 / 3)
    assert scores["flow_error_median"] == pytest.approx(0.5)  # errors 0, 2, 0.5
    assert scores["flow_error_p90"] == pytest.approx(np.percentile([0.0, 2.0, 0.5], 90))
    assert scores["flow_rank_correlation"] == pytest.approx(1.0)  # sizes ranked alike: 10, 10, 2.5 against 10, 10, 5
    assert sc.flow_scores([0.1], [1.0], floor=1.0)["direction_agreement"] is None


def test_the_share_error_forgives_a_network_fed_at_the_wrong_pressure():
    truth = np.array([4.0, 2.0, 1.0])
    assert sc.share_error_median(truth, 3.0 * truth, 0.0, inflow_true=8.0, inflow_model=24.0) == pytest.approx(0.0)
    assert sc.flow_scores(truth, 3.0 * truth, 0.0)["flow_error_median"] == pytest.approx(2.0)
    assert sc.share_error_median(truth, truth, 0.0, inflow_true=0.0, inflow_model=1.0) is None


def test_an_open_end_given_no_flow_is_wrong_and_shares_count_only_inflow():
    assert sc.direction_agreement_at_ends([3.0, -2.0, 0.01], [1.0, 0.0, 5.0], floor=0.1) == pytest.approx(0.5)
    assert sc.direction_agreement_at_ends([0.01], [1.0], floor=0.1) is None
    # Blood leaving (negative) enters no share.
    assert sc.inflow_share([3.0, -5.0], [1.0, -4.0]) == pytest.approx(0.75)
    assert sc.inflow_share([-1.0], [-1.0]) is None


def test_the_widest_ends_on_a_side_are_the_inlets():
    def end(node, side_order, width):
        return ImageOpenEnd(node, (0.0, 0.0, 0.0), "x0", None, width, side_order, {"A": "arteriole", "V": "venule"}[side_order], None)

    ends = [end(1, "A", 8.0), end(2, "A", 12.0), end(3, "V", 20.0), end(4, "A", 12.0)]
    assert sc.widest_ends(ends, ARTERIAL, 2) == [2, 4]
    assert sc.widest_ends(ends, VENOUS, 2) == [3]


def test_boxes_keep_off_the_outer_faces_along_the_axes_grown():
    corners = sc.box_positions((120.0, 300.0, 300.0), (120.0, 200.0, 200.0))
    assert np.all(corners[:, 0] == 0.0)  # the depth is the template's: one place
    ys = np.unique(corners[:, 1])
    assert ys == pytest.approx([20.0, 45.0, 70.0])  # 10% of 200 off each face, steps of 200 / 8
    assert len(corners) == 9


def test_a_box_is_judged_by_the_tree_vessels_its_faces_cut():
    G = nx.MultiGraph(image_extent_um=(10.0, 100.0, 100.0))
    for i, (start, end, order) in enumerate([
        ((5, 50, 0), (5, 50, 40), "Art1"),  # crosses x = 30
        ((5, 50, 60), (5, 50, 100), "Ven1"),  # crosses x = 70
        ((5, 20, 40), (5, 20, 60), "B3"),  # a capillary: never counted
    ]):
        G.add_node(2 * i, pos=np.asarray(start, float))
        G.add_node(2 * i + 1, pos=np.asarray(end, float))
        voxels = [list(np.asarray(start) + (np.asarray(end) - np.asarray(start)) * t) for t in np.linspace(0, 1, 41)]
        G.add_edge(2 * i, 2 * i + 1, voxels=voxels, branch_order=order)
    corners = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 41.0]])
    cut = sc.tree_vessels_cut(G, corners, (10.0, 100.0, 40.0))
    assert cut.tolist() == [[1, 1], [0, 0], [0, 1]]  # x 30-70; 0-40 holds the arteriole whole; 41-81


@pytest.fixture(scope="module")
def fixture_case():
    from synthetic_fixtures import fixture_template

    return sc.run_case(fixture_template(), seed=1, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0)


def test_the_oracle_gives_the_truth_back(fixture_case):
    oracle = next(r for r in fixture_case if r["model"] == sc.ORACLE)
    assert oracle["vessels"] > 50 and oracle["open_ends"] > 10
    assert oracle["direction_agreement"] == 1.0 and oracle["open_end_direction_agreement"] == 1.0
    assert oracle["flow_error_p90"] < 1e-6 and oracle["share_error_median"] < 1e-6
    assert oracle["flow_rank_correlation"] > 0.9999  # near-equal flows may swap ranks
    # Each end is held by a conductance 1e8 times the stiffest vessel's, not
    # fixed outright: the inflow through two ends comes back within 1e-5.
    assert oracle["designated_inflow_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert oracle["open_end_inflow_share_model"] == pytest.approx(oracle["open_end_inflow_share_truth"], rel=1e-5)


def test_every_model_is_scored_on_the_same_image(fixture_case):
    models = [r["model"] for r in fixture_case]
    assert models[0] == sc.ORACLE and "estimated" in models and "closed" in models and len(models) == 8
    assert models[-1] == sc.SURROUND
    for column in ("vessels_in_crop", "open_ends", "vessels", "open_end_inflow_share_truth", "box_lo_um"):
        assert len({r[column] for r in fixture_case}) == 1, column
    first = fixture_case[0]
    assert first["inlets"] == 2 and first["outlets"] == 2
    assert 0.0 < first["open_end_inflow_share_truth"] < 1.0
    closed = next(r for r in fixture_case if r["model"] == "closed")
    assert closed["open_end_inflow_share_model"] == pytest.approx(0.0, abs=1e-9)
    for row in fixture_case:
        for name in ("direction_agreement", "open_end_direction_agreement"):
            assert 0.0 <= row[name] <= 1.0


def test_the_trunk_rule_is_counted_once_a_case_and_flagged_at_half(fixture_case):
    rows = [
        {"model": sc.ORACLE, "arterial_ends_upstream": 3, "arterial_ends_judged": 4, "venous_ends_upstream": 1, "venous_ends_judged": 2},
        {"model": "estimated", "arterial_ends_upstream": 3, "arterial_ends_judged": 4},  # the same truth again
        {"model": sc.ORACLE, "arterial_ends_upstream": 0, "arterial_ends_judged": 2, "venous_ends_upstream": None},
    ]
    assert sc.trunk_rule(rows) == (4, 8)
    assert sc.trunk_rule_line(4, 8).endswith("revisit D5.**")
    assert "right at 5 of 8 tree ends (62%)." in sc.trunk_rule_line(5, 8) and "revisit" not in sc.trunk_rule_line(5, 8)
    assert "no tree end" in sc.trunk_rule_line(0, 0)
    # On a real case the counts are the truth's, alike in every model's row.
    oracle = fixture_case[0]
    for row in fixture_case:
        assert (row["arterial_ends_upstream"], row["venous_ends_judged"]) == (
            oracle["arterial_ends_upstream"], oracle["venous_ends_judged"],
        )
    if oracle["arterial_ends_judged"]:
        assert oracle["arterial_ends_upstream_in_truth"] == pytest.approx(
            oracle["arterial_ends_upstream"] / oracle["arterial_ends_judged"]
        )


def test_the_surround_is_scored_on_the_image_it_was_drawn_round(fixture_case):
    row = fixture_case[-1]
    assert row["model"] == sc.SURROUND
    assert row["surround_vessels"] > row["vessels_in_crop"] and row["surround_trunks"] > 0
    # The truth is closed and the box its depth: every end is on a face
    # the surround grows past, and every one is joined.
    assert row["surround_left_open"] == 0
    assert row["surround_trunks"] + row["surround_bed_joins"] == row["open_ends"] + row["inlets"] + row["outlets"]
    for name in ("direction_agreement", "open_end_direction_agreement", "flow_rank_correlation"):
        assert 0.0 <= row[name] <= 1.0
    assert row["open_end_inflow_share_model"] > 0.0  # the surround feeds the image through its open ends


def test_the_surrounds_pressures_are_read_back_onto_the_image():
    """Blood is conserved at every junction of the image, solved as part of
    the surround: the pressures land on the right nodes, with the right
    conductances."""
    from haemolynx.graph import crop_graph_to_box
    from synthetic_fixtures import fixture_template

    template = fixture_template()
    truth, _roles = sc.truth_network(template, seed=1, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0)
    lo = sc.candidate_boxes(truth, template.extent_um, 2, 2)[0]
    image, inlets, outlets, _ends = sc.stand_in_image(crop_graph_to_box(truth, lo, lo + np.asarray(template.extent_um)), 2, 2)
    C, S = sc.solve_surrounded(
        image, template, inlets, outlets, seed=7, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0,
    )
    assert set(C.nodes) == set(image.nodes) and all(C.nodes[n]["pressure"] == S.nodes[n]["pressure"] for n in C)
    largest = max(abs(float(d["conductance"]) * (C.nodes[a]["pressure"] - C.nodes[b]["pressure"])) for a, b, d in C.edges(data=True))
    for node in C:
        if C.degree(node) < 2:
            continue
        net = sum(float(d["conductance"]) * (C.nodes[node]["pressure"] - C.nodes[other]["pressure"]) for _n, other, d in C.edges(node, data=True))
        assert abs(net) < 1e-6 * largest, node
    assert 1000.0 < min(C.nodes[n]["pressure"] for n in C) and max(C.nodes[n]["pressure"] for n in C) < 4500.0


@pytest.fixture(scope="module")
def stand_in():
    from haemolynx.graph import crop_graph_to_box
    from synthetic_fixtures import fixture_template

    template = fixture_template()
    truth, roles = sc.truth_network(template, seed=1, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0)
    lo = sc.candidate_boxes(truth, template.extent_um, 2, 2)[0]
    image, inlets, outlets, ends = sc.stand_in_image(crop_graph_to_box(truth, lo, lo + np.asarray(template.extent_um)), 2, 2)
    return template, roles, image, inlets, outlets, ends


def test_a_runs_own_template_is_measured_from_the_stand_in_image(stand_in):
    from haemolynx.statistics.bifurcation import branch_order_category

    template, roles, image, inlets, outlets, ends = stand_in
    measured = sc.image_template(image, inlets, outlets, roles)
    assert measured.extent_um == pytest.approx(template.extent_um)
    assert measured.inlet_count == len(inlets) and measured.outlet_count == len(outlets)
    assert {e.face for e in measured.inlets} == {"x0"} and {e.face for e in measured.outlets} == {"x1"}
    # Every vessel its faces cut is a dead end leaving the image, as in a run.
    assert measured.dead_ends.leaves_image == len(ends) and measured.dead_ends.interior == 0
    by_class = {}
    for *_e, d in image.edges(data=True):
        category = branch_order_category(str(d["branch_order"]))
        by_class[category] = by_class.get(category, 0) + 1
    assert {k: c.count for k, c in measured.classes.items() if c.count} == by_class
    # The boundary nodes are the truth's within the image: what perfect masks give.
    assert measured.arteriole_boundary_count == sum(1 for n in roles["arteriole_boundary"] if n in image)
    assert measured.venule_boundary_count == sum(1 for n in roles["venule_boundary"] if n in image)
    assert measured.provenance.source == "statistics"


def test_a_surround_is_drawn_to_each_template_asked_for(monkeypatch):
    from synthetic_fixtures import fixture_template

    drawn_to = []
    real = sc.solve_surrounded

    def spy(image, template, *args, **kwargs):
        drawn_to.append(template)
        return real(image, template, *args, **kwargs)

    monkeypatch.setattr(sc, "solve_surrounded", spy)
    rows = sc.run_case(
        fixture_template(), seed=1, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0,
        surround_scale=2.0, surround_templates=(sc.TRUTH_TEMPLATE, sc.IMAGE_TEMPLATE),
    )
    truth_row, image_row = rows[-2:]
    assert (truth_row["model"], image_row["model"]) == (sc.SURROUND, sc.SURROUND_TEMPLATES[sc.IMAGE_TEMPLATE])
    assert drawn_to[0] is not drawn_to[1] and drawn_to[0] == fixture_template()
    assert image_row["surround_template_vessels"] == image_row["vessels_in_crop"] == drawn_to[1].vessel_count
    assert 0.0 <= image_row["direction_agreement"] <= 1.0 and image_row["surround_vessels"] > image_row["vessels_in_crop"]
    assert sc.kill_criterion(rows, image_row["model"]).startswith(f"Kill criterion ({image_row['model']} beats estimated")


def _row(model, bed, seed, error, direction):
    return {"model": model, "bed": bed, "seed": seed, "flow_error_median": error, "direction_agreement": direction}


def test_the_kill_criterion_needs_the_surround_to_beat_estimated_on_both():
    met = [
        _row("estimated", "d", 1, 0.8, 0.70), _row(sc.SURROUND, "d", 1, 0.5, 0.80),
        _row("estimated", "d", 2, 0.6, 0.75), _row(sc.SURROUND, "d", 2, 0.7, 0.85),
        _row("estimated", "v", 1, 0.9, 0.72), _row(sc.SURROUND, "v", 1, 0.4, 0.70),
    ]
    line = sc.kill_criterion(met)
    assert line.endswith("**met**.")
    assert "median flow error 0.5 against 0.8 (better in 2 of 3 runs)" in line
    assert "direction agreement 0.8 against 0.72 (better in 2 of 3 runs)" in line
    # Better on flows, worse on directions: not met.
    worse = [_row("estimated", "d", 1, 0.8, 0.9), _row(sc.SURROUND, "d", 1, 0.5, 0.8)]
    assert "**not met**" in sc.kill_criterion(worse) and "(D9)" in sc.kill_criterion(worse)
    # A score neither measured fails it; a missing row cannot judge it.
    assert "not met" in sc.kill_criterion([_row("estimated", "d", 1, None, 0.5), _row(sc.SURROUND, "d", 1, 0.1, 0.9)])
    assert "no surround or estimated row" in sc.kill_criterion([_row("estimated", "d", 1, 0.8, 0.7)])
    # A surround that could not be drawn loses its run, however well the rest do.
    failed = [*met, _row("estimated", "v", 2, 0.9, 0.7), {"model": sc.SURROUND, "bed": "v", "seed": 2, "surround_failed": "no bed"}]
    line = sc.kill_criterion(failed)
    assert "not drawn in 1 of 4 runs" in line and "**not met**" in line
    assert "(better in 2 of 4 runs)" in line


def test_a_surround_too_thin_to_draw_loses_its_run_and_keeps_the_models(monkeypatch):
    from synthetic_fixtures import fixture_template

    def cannot_draw(*_args, **_kwargs):
        raise ValueError("The template leaves no capillary bed to draw")

    monkeypatch.setattr(sc, "solve_surrounded", cannot_draw)
    rows = sc.run_case(fixture_template(), seed=1, scale=1.5, axes="xy", bed_model="delaunay", inlet_pa=4500.0, outlet_pa=1000.0)
    assert len(rows) == 8 and rows[-1]["model"] == sc.SURROUND
    assert rows[-1]["surround_failed"].startswith("The template leaves no capillary bed")
    assert "direction_agreement" not in rows[-1] and rows[-2]["direction_agreement"] is not None


def test_the_report_summarises_each_model(tmp_path):
    rows = [
        {"model": "a", "x": 1.0, "box": "0 1 2"}, {"model": "a", "x": 3.0, "box": "0 1 2"}, {"model": "b", "x": None},
    ]
    summary = sc.summarise(rows, ["x", "box"])
    assert summary == [{"model": "a", "runs": 2, "x": 2.0, "box": None}, {"model": "b", "runs": 1, "x": None, "box": None}]
    assert sc.markdown_table(summary, ["model", "x"]).splitlines()[2] == "| a | 2 |"
