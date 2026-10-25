"""Grid P's interactive page (``scripts/thick_junction_review``): what each
case hands the page -- the segmentation and the thick-vessel region a plane
at a time across the sheets' slab, and every line in the sheet's own view --
and the page built from it. The grid itself is drawn from a run and never in
CI."""
from __future__ import annotations

import base64
import json

import numpy as np

from thick_bridge_review.build_grid import HALF_SLAB_UM, HALF_WINDOW_UM, PLANES
from thick_junction_review.grid_page import PLACEHOLDER, build_page
from thick_junction_review.phase1_rules_grid import case_view

VS = np.array([1.0, 0.5, 0.5])


def _unpack(text: str, shape) -> np.ndarray:
    flat = np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))
    return flat[: int(np.prod(shape))].reshape(shape).astype(bool)


def _scene():
    """A 60 x 120 x 120 voxel stack (60 x 60 x 60 um): a thick vessel along y
    at z = 30 um, x = 30 um, its region the vessel's own voxels, and a thin
    vessel's centreline along x at y = 30 um ending at the wall."""
    shape = (60, 120, 120)
    zz, yy, xx = np.indices(shape)
    z_um, x_um = zz * VS[0], xx * VS[2]
    thick = (z_um - 30) ** 2 + (x_um - 30) ** 2 <= 6.0 ** 2
    mask = thick.copy()
    mask[28:33, 58:63, 72:] = True
    thin = np.array([[30.0, 30.0, x] for x in np.arange(36.0, 59.5, 0.5)])
    far = np.array([[30.0 + d, 10.0, 25.0] for d in np.arange(-15.0, 15.5, 0.5)])
    tree = np.array([[30.0, y, 30.0] for y in np.arange(0.0, 60.0, 0.5)])
    return mask, thick, tree, [(thin, False), (far, True)], thin


def test_a_case_hands_the_page_its_slab_a_plane_at_a_time():
    mask, region, tree, paths, thin = _scene()
    case = {"id": "P01", "rule": "wall", "vox": (30, 60, 72), "view": "down z", "axis": 0,
            "centre": np.array([30.0, 30.0, 36.0]), "piece": thin[:10], "route": thin[[0, -1]]}

    view = case_view(case, mask, region, tree, paths, VS)

    label, h, w = PLANES[0]
    assert (view["depth"], view["down"], view["across"]) == ("z", "y", "x")
    n, rows, cols = view["shape"]
    # The sheets' slab: the case's plane and HALF_SLAB_UM either side.
    assert n == int(np.ceil(2 * HALF_SLAB_UM / VS[0])) + 1
    assert cols == int(np.ceil(2 * HALF_WINDOW_UM / VS[2])) + 1 or view["origin_um"][0] == 0.0
    lo = np.round(np.asarray([view["origin_um"][2], view["origin_um"][1], view["origin_um"][0]]) / VS).astype(int)
    expected = mask[lo[0]:lo[0] + n, lo[1]:lo[1] + rows, lo[2]:lo[2] + cols]
    assert np.array_equal(_unpack(view["seg"], view["shape"]), expected)
    assert np.array_equal(_unpack(view["region"], view["shape"]),
                          region[lo[0]:lo[0] + n, lo[1]:lo[1] + rows, lo[2]:lo[2] + cols])
    # Lines are [across, down, depth] microns: x, y, z for a view down z.
    assert view["piece"][0] == [36.0, 30.0, 30.0]
    assert view["route"] == [[36.0, 30.0, 30.0], [59.0, 30.0, 30.0]]


def test_only_what_the_slab_holds_is_handed_over_split_where_it_leaves():
    """The bridge runs through z 15-45 um: the slab is z 24-36 (and 2 um
    either side for lines), so only that stretch reaches the page, as one
    run; the thick vessel's tree is kept within the window."""
    mask, region, tree, paths, thin = _scene()
    case = {"id": "P02", "rule": "taper", "vox": (30, 20, 50), "view": "down z", "axis": 0,
            "centre": np.array([30.0, 10.0, 25.0]), "piece": thin[:4]}

    view = case_view(case, mask, region, tree, paths, VS)

    (bridge,) = [line for line in view["lines"] if line["kind"] == "bridge"]
    depths = [p[2] for p in bridge["points"]]
    assert min(depths) >= 30.0 - HALF_SLAB_UM - 2.0 and max(depths) <= 30.0 + HALF_SLAB_UM + 2.0
    assert all(abs(p[1] - 10.0) <= HALF_WINDOW_UM + 5 for p in view["tree"])
    assert view["route"] is None and view["tick"] is None


def test_the_page_holds_the_data_once_and_nothing_in_it_closes_its_script():
    data = {"dataset": "test", "voxel_size_zyx": [1, 1, 1], "slab_um": 12,
            "sheets": [{"rule": "wall", "title": "Walls", "question": "a </script> question",
                        "marks": [{"key": "J", "label": "Join"}], "cases": []}]}

    page = build_page(data)

    assert PLACEHOLDER not in page
    block = page.split('<script id="grid-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(block) == data
