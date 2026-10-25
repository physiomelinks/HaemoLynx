"""A batch condition's runs as one synthetic template (``synthetic/average.py``,
``synthetic/batch_template.py``; ``PLAN_batch_templates.md``, Harvey's B1a,
B2a, B3a, B4b)."""
from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from haemolynx.synthetic import (
    average_templates,
    batch_template_filename,
    clear_template_cache,
    is_batch_template,
    load_template,
    read_batch_template,
    write_batch_template,
)
from haemolynx.synthetic.average import common_extent, common_volume_factors, representative_positions
from haemolynx.synthetic.compare import ComparisonRow, comparison_section, with_members_range
from haemolynx.synthetic.organisation import measure_organisation
from haemolynx.synthetic.template import Endpoint, diameter_basis_of_run
from synthetic_fixtures import fixture_template, write_generated_run


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_template_cache()
    yield
    clear_template_cache()


def test_two_runs_of_different_size_average_their_densities_in_the_mean_box():
    """B1a: each axis the mean extent; every count the members' mean density
    times that box's volume."""
    small = fixture_template()
    large = small.scaled(2.0 ** (1.0 / 3.0))  # twice the volume, twice the vessels
    average = average_templates([small, large])
    assert average.extent_um == pytest.approx(common_extent([small, large]))
    volume = float(np.prod(average.extent_um))
    factor = volume / float(np.prod(small.extent_um))
    assert common_volume_factors([small, large]) == pytest.approx([factor, volume / float(np.prod(large.extent_um))])
    for category, target in small.classes.items():
        expected = (target.count * factor + large.classes[category].count * volume / np.prod(large.extent_um)) / 2
        assert average.classes[category].count == int(round(expected))
    assert average.total_length_um == pytest.approx(
        (small.total_length_um * factor + large.total_length_um * volume / np.prod(large.extent_um)) / 2
    )


def test_one_run_averaged_with_itself_is_itself():
    T = fixture_template()
    twice = average_templates([T, T])
    assert twice.extent_um == T.extent_um and twice.vessel_count == T.vessel_count
    assert sorted((o.label, o.edge_count) for o in twice.orders) == sorted((o.label, o.edge_count) for o in T.orders)
    assert {k: v.count for k, v in twice.classes.items()} == {k: v.count for k, v in T.classes.items()}
    assert twice.dead_ends == T.dead_ends and twice.loops.independent_loops == T.loops.independent_loops
    for mine, theirs in ((twice.inlets, T.inlets), (twice.outlets, T.outlets)):
        assert sorted(e.face for e in mine) == sorted(e.face for e in theirs)
        assert sorted(e.position_rel for e in mine) == pytest.approx(sorted(e.position_rel for e in theirs))


def test_an_order_a_run_lacks_counts_none_for_it():
    T = fixture_template()
    deepest = max(T.orders, key=lambda o: o.index if o.category == "arteriole" else -1)
    shallower = replace(T, orders=tuple(o for o in T.orders if o.label != deepest.label))
    average = average_templates([T, shallower])
    drawn = {o.label: o.edge_count for o in average.orders}
    assert drawn[deepest.label] == int(round(deepest.edge_count / 2))
    # Its widths and lengths are the run that has it.
    kept = next(o for o in average.orders if o.label == deepest.label)
    assert kept.mean_diameter_um == pytest.approx(deepest.mean_diameter_um)


def test_inlets_and_outlets_take_the_mean_count_on_the_faces_most_use():
    """B3a."""
    T = fixture_template()
    a = replace(T, inlets=(Endpoint("x0", (0.4, 0.3, 0.0)), Endpoint("x0", (0.6, 0.7, 0.0))))
    b = replace(T, inlets=(Endpoint("x0", (0.5, 0.5, 0.0)), Endpoint("x0", (0.4, 0.4, 0.0)), Endpoint("y1", (0.5, 1.0, 0.5)), Endpoint("x0", (0.5, 0.6, 0.0))))
    average = average_templates([a, b])
    assert len(average.inlets) == 3  # (2 + 4) / 2
    faces = [e.face for e in average.inlets]
    assert faces.count("x0") == 3 or (faces.count("x0") == 2 and faces.count("y1") == 1)
    assert all(e.face in ("x0", "y1") for e in average.inlets)


def test_representative_positions_are_the_centres_of_groups():
    points = np.array([[0.1, 0.1, 0.0], [0.12, 0.1, 0.0], [0.9, 0.9, 0.0], [0.88, 0.92, 0.0]])
    centres = sorted(representative_positions(points, 2))
    assert centres[0] == pytest.approx((0.11, 0.1, 0.0)) and centres[1] == pytest.approx((0.89, 0.91, 0.0))
    assert len(representative_positions(points[:1], 3)) == 3
    assert representative_positions(points, 0) == []


def _graph(edges, extent):
    G = nx.MultiGraph()
    index: dict = {}
    for a, b in edges:
        for p in (a, b):
            if p not in index:
                index[p] = len(index)
                G.add_node(index[p], pos=np.asarray(p, float))
        G.add_edge(index[a], index[b], voxels=[list(a), list(b)], branch_order="B01")
    return G


def test_layouts_are_pooled_on_the_common_grid():
    """B2a: per-cell fields averaged as densities, capillary vertices pooled."""
    extent = (100.0, 200.0, 200.0)
    lower = _graph([((5.0, y, x), (40.0, y, x)) for y in range(10, 200, 20) for x in range(10, 200, 20)], extent)
    upper = _graph([((60.0, y, x), (95.0, y, x)) for y in range(10, 200, 20) for x in range(10, 200, 20)], extent)
    a, b = measure_organisation(lower, extent), measure_organisation(upper, extent)
    T = fixture_template()
    average = average_templates([replace(T, extent_um=extent, organisation=a), replace(T, extent_um=extent, organisation=b)])
    org = average.organisation
    assert org.shape == a.shape
    nodes = org.field("capillary_nodes")
    assert nodes[0].sum() == pytest.approx(a.field("capillary_nodes").sum() / 2)
    assert nodes[1].sum() == pytest.approx(b.field("capillary_nodes").sum() / 2)
    assert len(org.capillary_node_positions_rel) == len(a.capillary_node_positions_rel) + len(b.capillary_node_positions_rel)
    assert org.axis_share == pytest.approx(a.axis_share)


def test_layouts_on_grids_of_other_shapes_are_read_at_the_common_cells():
    small, large = (100.0, 100.0, 100.0), (300.0, 300.0, 300.0)
    g_small = _graph([((5.0, 50.0, 50.0), (40.0, 50.0, 50.0))], small)
    g_large = _graph([((15.0, 150.0, 150.0), (120.0, 150.0, 150.0))], large)
    a, b = measure_organisation(g_small, small), measure_organisation(g_large, large)
    assert a.shape != b.shape
    T = fixture_template()
    members = [replace(T, extent_um=small, organisation=a), replace(T, extent_um=large, organisation=b)]
    average = average_templates(members)
    assert average.organisation.shape == (4, 4, 4)  # 200 um a side, 50 um cells
    # Each member's vertices kept whole on the common grid, as a density (B1a).
    f_small, f_large = common_volume_factors(members)
    assert (f_small, f_large) == pytest.approx((8.0, 8.0 / 27.0))
    assert sum(average.organisation.capillary_nodes) == pytest.approx(
        (f_small * sum(a.capillary_nodes) + f_large * sum(b.capillary_nodes)) / 2
    )
    # The small run's vessel lies in its lower half: so does its share.
    lower = average.organisation.field("capillary_nodes")[:2].sum()
    assert lower == pytest.approx(f_small * sum(a.capillary_nodes) / 2 + f_large * b.field("capillary_nodes")[:3].sum() / 2)


@pytest.fixture
def two_runs(tmp_path):
    a = write_generated_run(tmp_path / "a", seed=11, name="WT_1.haemorun")
    b = write_generated_run(tmp_path / "b", seed=12, name="WT_2.haemorun", settings={"segmentation_source": "endothelial"})
    c = write_generated_run(tmp_path / "c", seed=13, name="WT_3.haemorun", settings={"segmentation_source": "endothelial"})
    return tmp_path, (a, b, c)


def test_a_haemobatch_is_written_read_and_averaged(two_runs):
    tmp_path, (a, b, c) = two_runs
    junk = tmp_path / "junk.haemorun"
    junk.write_bytes(b"not a run")
    path = write_batch_template(tmp_path / "wt.haemobatch", [("WT_1", a), ("WT_2", b), ("WT_3", c), ("junk", junk)], condition="wt")
    assert path == tmp_path / "wt.haemobatch" and is_batch_template(path)
    file = read_batch_template(path)
    assert file.condition == "wt" and [m.stem for m in file.members] == ["WT_1", "WT_2", "WT_3"]
    assert [stem for stem, _why in file.left_out] == ["junk"]
    for source in ("statistics", "graph"):
        T = load_template(path, source)
        members = [load_template(run, source) for run in (a, b, c)]
        # Each class's count is its own mean, rounded.
        assert abs(T.vessel_count - np.mean([m.vessel_count for m in members])) <= len(T.classes) / 2
        for category, target in T.classes.items():
            assert target.count == int(round(float(np.mean([m.classes[category].count for m in members]))))
        assert T.provenance.template_path == str(path) and T.provenance.source == source
        assert T.provenance.notes[0] == "the average of 3 runs of condition 'wt': WT_1, WT_2, WT_3"
        assert any("left out of the batch template: junk" in note for note in T.provenance.notes)
    # The basis most of the runs were read on.
    assert diameter_basis_of_run(path) == "endothelial"


def test_no_run_that_can_be_a_template_writes_no_file(tmp_path):
    junk = tmp_path / "junk.haemorun"
    junk.write_bytes(b"not a run")
    assert write_batch_template(tmp_path / "x.haemobatch", [("junk", junk)]) is None
    assert not (tmp_path / "x.haemobatch").exists()


def test_a_file_that_is_not_a_batch_template_is_refused(tmp_path):
    from haemolynx.synthetic import TemplateError

    bad = tmp_path / "bad.haemobatch"
    bad.write_bytes(b"\x1f\x8bnot gzip")
    with pytest.raises(TemplateError):
        load_template(bad, "statistics")


def test_the_file_is_named_after_its_condition():
    assert batch_template_filename("wt") == "wt.haemobatch"
    assert batch_template_filename("KO mice/2") == "KO_mice_2.haemobatch"
    assert batch_template_filename(None) == "batch.haemobatch"


def test_each_row_reports_the_members_range():
    """B4b: whether the synthetic value lies within the members' own."""
    def row(characteristic, template, synthetic):
        return ComparisonRow("network", characteristic, template, synthetic, None, "rel 10%", None)

    rows = [row("vessels", 100, 104), row("average degree", 2.3, 2.9), row("capillary length by cell (correlation)", 1.0, 0.7)]
    members = [[row("vessels", 96, 104), row("average degree", 2.2, 2.9), row("capillary length by cell (correlation)", 1.0, 0.7)],
               [row("vessels", 108, 104), row("average degree", 2.4, 2.9), row("capillary length by cell (correlation)", 1.0, 0.7)]]
    ranged = {r.characteristic: r for r in with_members_range(rows, members)}
    assert (ranged["vessels"].members_low, ranged["vessels"].members_high, ranged["vessels"].within_members) == (96, 108, True)
    assert ranged["average degree"].within_members is False
    assert ranged["capillary length by cell (correlation)"].within_members is None
    section = comparison_section(list(ranged.values()))
    assert section["characteristics within the members' range"] == "1 of 2"
    assert section["network: vessels"].endswith("members 96 to 108: within")
    assert math.isclose(ranged["vessels"].members_high, 108)
