"""No second, parallel vessel inside one segmented branch.

Every site that adds a path to the network -- the graph's gap, optimise and
orphan reconnects, the mask recovery -- asks
``graph._helpers.duplicates_existing_vessel`` first: does the path run beside
an existing edge in the same lumen? ``graph.diagnose_parallel_duplicates_in_lumen``
asks the same of a finished graph. Two real vessels with background between
them are not duplicates, nor is a branch meeting the vessel it leaves.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from haemolynx.graph import (
    diagnose_parallel_duplicates_in_lumen,
    duplicates_existing_vessel,
    format_parallel_duplicates_report,
    reconnect_orphan_and_dangling_nodes,
)
from haemolynx.graph._helpers import EdgeSampleIndex
from haemolynx.preprocessing import MaskSupport


def _tube_along_x(shape, centre_zy, radius):
    zz, yy, _xx = np.indices(shape)
    return (zz - centre_zy[0]) ** 2 + (yy - centre_zy[1]) ** 2 <= radius**2


def _line(z, y, x0, x1):
    return np.array([[z, y, x] for x in np.arange(x0, x1 + 0.5, 1.0)], dtype=float)


def _graph(edges) -> nx.MultiGraph:
    """Nodes named by position; each edge a straight polyline."""
    G = nx.MultiGraph()
    G.graph["voxel_size"] = (1.0, 1.0, 1.0)
    for start, end in edges:
        a, b = tuple(map(float, start)), tuple(map(float, end))
        for node in (a, b):
            G.add_node(node, pos=np.asarray(node))
        n = int(np.ceil(np.linalg.norm(np.subtract(b, a))))
        voxels = [list(np.add(a, np.subtract(b, a) * t)) for t in np.linspace(0, 1, n + 1)]
        G.add_edge(a, b, voxels=voxels, length=float(np.linalg.norm(np.subtract(b, a))))
    return G


def _wide_vessel():
    """A 12 um vessel along x, centred on (7, 7)."""
    return _tube_along_x((15, 15, 60), (7, 7), 6)


def _two_vessels_with_background_between():
    """Two vessels 7 um wide along x, with 3 um of background between their walls."""
    mask = np.zeros((9, 22, 60), dtype=bool)
    mask[1:8, 1:8] = True
    mask[1:8, 11:18] = True
    return mask


def _duplicates(path, G, mask):
    support = MaskSupport(mask, (1.0, 1.0, 1.0))
    return duplicates_existing_vessel(path, G, support.inside, support.radius)


def test_a_second_strand_in_a_wide_vessel_duplicates_its_centreline():
    mask = _wide_vessel()
    G = _graph([((7, 4, 2), (7, 4, 57))])

    assert _duplicates(_line(7, 10, 5, 55), G, mask)


def test_a_vessel_beside_another_behind_background_is_not_a_duplicate():
    mask = _two_vessels_with_background_between()
    G = _graph([((4, 4, 2), (4, 4, 57))])

    assert not _duplicates(_line(4, 14, 5, 55), G, mask)


def test_a_branch_meeting_a_vessel_is_not_a_duplicate_of_it():
    mask = _tube_along_x((9, 40, 60), (4, 5), 3)
    mask[1:8, 5:35, 28:33] = True
    G = _graph([((4, 5, 0), (4, 5, 59))])
    branch = np.array([[4, y, 30] for y in range(5, 34)], dtype=float)

    assert not _duplicates(branch, G, mask)


def test_paths_added_after_the_index_was_built_are_judged_too():
    """A pass builds one index and adds its own paths to it as it goes."""
    mask = _wide_vessel()
    G = _graph([((7, 7, 2), (7, 7, 3))])
    support = MaskSupport(mask, (1.0, 1.0, 1.0))
    index = EdgeSampleIndex(G)

    first = _line(7, 4, 5, 55)
    assert not duplicates_existing_vessel(first, G, support.inside, support.radius, index=index)
    index.add(first)
    assert duplicates_existing_vessel(
        _line(7, 10, 5, 55), G, support.inside, support.radius, index=index
    )


def test_the_diagnostic_reports_two_edges_in_one_lumen_and_nothing_else():
    wide = _wide_vessel()
    doubled = _graph([((7, 4, 2), (7, 4, 57)), ((7, 10, 2), (7, 10, 57))])
    single = _graph([((7, 7, 2), (7, 7, 57))])
    separate = _graph([((4, 4, 2), (4, 4, 57)), ((4, 14, 2), (4, 14, 57))])

    report = diagnose_parallel_duplicates_in_lumen(doubled, wide)

    assert report["duplicate_pair_count"] == 1
    assert report["edge_count"] == 2
    assert "1 pair(s)" in format_parallel_duplicates_report(report)
    assert diagnose_parallel_duplicates_in_lumen(single, wide)["duplicate_pair_count"] == 0
    assert (
        diagnose_parallel_duplicates_in_lumen(
            separate, _two_vessels_with_background_between()
        )["duplicate_pair_count"]
        == 0
    )


def _a_broken_second_strand(with_first_strand: bool) -> nx.MultiGraph:
    """In the wide vessel: the strand at y=10 broken between x=20 and x=38,
    its two ends facing; and, optionally, the vessel's own centreline at y=4."""
    edges = [((7, 10, 5), (7, 10, 20)), ((7, 10, 38), (7, 10, 52))]
    if with_first_strand:
        edges.append(((7, 4, 0), (7, 4, 59)))
    return _graph(edges)


def test_an_orphan_reconnect_does_not_lay_a_second_strand_in_one_lumen():
    """Regression: with a 20 um reach the two ends of a second strand were
    joined along the vessel's existing centreline, 6 um away in one lumen."""
    support = MaskSupport(_wide_vessel(), (1.0, 1.0, 1.0))
    settings = dict(reconnect_threshold=20.0, validate_reconnections=False)
    ends = ((7.0, 10.0, 20.0), (7.0, 10.0, 38.0))

    unguarded = reconnect_orphan_and_dangling_nodes(_a_broken_second_strand(True), **settings)
    guarded = reconnect_orphan_and_dangling_nodes(
        _a_broken_second_strand(True), mask_support=support, **settings
    )
    alone = reconnect_orphan_and_dangling_nodes(
        _a_broken_second_strand(False), mask_support=support, **settings
    )

    def pairs(G):
        return diagnose_parallel_duplicates_in_lumen(G, _wide_vessel())["duplicate_pair_count"]

    before = pairs(_a_broken_second_strand(True))
    assert unguarded.has_edge(*ends) and pairs(unguarded) == before + 1
    assert not guarded.has_edge(*ends) and pairs(guarded) == before
    assert alone.has_edge(*ends)
    assert alone.edges[(*ends, 0)]["bridge_kind"] == "orphan"
