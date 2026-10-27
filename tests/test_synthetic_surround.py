"""The image's open ends read for a synthetic network drawn round it
(``synthetic/surround.py``, ``PLAN_synthetic_surround.md`` phase 1)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.graph.dead_ends import DEAD_END_KIND, LEAVES_IMAGE
from haemolynx.haemodynamics.open_ends import DRAINS, FEEDS
from haemolynx.synthetic import surround
from haemolynx.synthetic.surround import (
    ARTERIAL,
    BED,
    CAPILLARY,
    CONTINUATIONS,
    TRUNK,
    UNKNOWN,
    VENOUS,
    ImageOpenEnd,
    read_open_ends,
)

CENTRE = (50.0, 50.0, 50.0)


def _star(arms: dict) -> nx.MultiGraph:
    """A junction at the box's centre with one vessel to each of *arms*
    (``{node: (tip position, edge attributes, kind)}``)."""
    G = nx.MultiGraph(image_extent_um=(100.0, 100.0, 100.0))
    G.add_node(0, pos=np.asarray(CENTRE))
    for node, (tip, attributes, kind) in arms.items():
        G.add_node(node, pos=np.asarray(tip, float), **({DEAD_END_KIND: kind} if kind else {}))
        voxels = [list(np.asarray(CENTRE) + (np.asarray(tip) - CENTRE) * t) for t in np.linspace(0, 1, 11)]
        G.add_edge(0, node, voxels=voxels, length=float(np.linalg.norm(np.asarray(tip) - CENTRE)), **attributes)
    return G


def _fixture() -> nx.MultiGraph:
    return _star({
        1: ((50.0, 50.0, 100.0), {"branch_order": "Art2", "diameter_um": 12.0}, LEAVES_IMAGE),
        2: ((0.0, 50.0, 50.0), {"branch_order": "Ven1", "diameter_um": 20.0}, LEAVES_IMAGE),
        3: ((50.0, 100.0, 50.0), {"branch_order": "B3", "diameter_um": 5.0}, LEAVES_IMAGE),
        4: ((50.0, 50.0, 30.0), {"branch_order": "B2", "diameter_um": 5.0}, "blind"),
    })


def test_each_open_end_is_read_with_its_face_heading_width_and_class():
    ends = {e.node: e for e in read_open_ends(_fixture())}
    assert set(ends) == {1, 2, 3}  # the blind end inside the tissue is not one
    art, ven, cap = ends[1], ends[2], ends[3]
    assert (art.face, ven.face, cap.face) == ("x1", "z0", "y1")
    assert art.heading == pytest.approx((0.0, 0.0, 1.0)) and ven.heading == pytest.approx((-1.0, 0.0, 0.0))
    assert (art.diameter_um, art.order, art.category, art.side) == (12.0, "Art2", "arteriole", ARTERIAL)
    assert (ven.order, ven.category, ven.side) == ("Ven1", "venule", VENOUS)
    assert (cap.order, cap.category, cap.side) == ("B3", "capillary", CAPILLARY)
    assert art.position_um == (50.0, 50.0, 100.0)
    # Unsolved: no direction, and none is needed.
    assert all(e.direction is None and e.designated is None for e in ends.values())
    assert (art.continuation, ven.continuation, cap.continuation) == (TRUNK, TRUNK, BED)


@pytest.mark.parametrize("direction", [FEEDS, DRAINS, None])
@pytest.mark.parametrize(
    "order, expected",
    [("Art2", TRUNK), ("Large_Art1", TRUNK), ("Ven1", TRUNK), ("Large_Ven2", TRUNK), ("B3", BED)],
)
def test_every_arteriole_and_venule_runs_on_as_a_trunk_whichever_way_it_flowed(monkeypatch, order, expected, direction):
    """D5 (Harvey, 2026-10-27): the solved direction was right at no more
    tree ends than a coin on the cut-out harness, so it decides nothing."""
    G = _star({1: ((50.0, 50.0, 100.0), {"branch_order": order, "diameter_um": 10.0}, LEAVES_IMAGE)})
    monkeypatch.setattr(surround, "open_end_directions", lambda _G: {} if direction is None else {1: direction})
    (end,) = read_open_ends(G)
    assert end.direction == direction and end.continuation == expected
    assert CONTINUATIONS == (BED, TRUNK)


def test_designated_inlets_and_outlets_run_on_as_trunks():
    G = _fixture()
    ends = {e.node: e for e in read_open_ends(G, inlets=[1], outlets=[2, 4])}
    assert ends[1].designated == "inlet" and ends[1].continuation == TRUNK
    assert ends[2].designated == "outlet" and ends[2].continuation == TRUNK
    # A designated dead end inside the tissue is read too.
    assert ends[4].designated == "outlet" and ends[4].face == "x0"


def test_a_synthetic_class_stands_in_for_a_branch_order():
    G = _star({
        1: ((50.0, 50.0, 100.0), {"synthetic_class": "venule", "diameter_um": 9.0}, LEAVES_IMAGE),
        2: ((50.0, 0.0, 50.0), {"diameter_um": 9.0}, LEAVES_IMAGE),
    })
    ends = {e.node: e for e in read_open_ends(G)}
    assert (ends[1].order, ends[1].category, ends[1].side) == (None, "venule", VENOUS)
    assert (ends[2].category, ends[2].side, ends[2].continuation) == (None, UNKNOWN, BED)


def test_the_extent_comes_from_the_graph_or_the_caller():
    G = _fixture()
    del G.graph["image_extent_um"]
    with pytest.raises(ValueError, match="image_extent_um"):
        read_open_ends(G)
    assert {e.node for e in read_open_ends(G, extent_um=(100.0, 100.0, 100.0))} == {1, 2, 3}


def test_an_open_end_is_a_plain_record():
    end = ImageOpenEnd(1, (0.0, 0.0, 0.0), "z0", None, None, None, None, None)
    assert end.side == UNKNOWN and end.continuation == BED
