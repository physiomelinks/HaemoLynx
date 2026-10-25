"""Grid S's tools (``scripts/stub_protrusion_review``, the stub prune's
protrusion test in ``PLAN_one_centreline_per_lumen.md``): the spy that measures
every dead end the stub prune is handed, the pool of those the protrusion test
decides, and how the 30 are spread. The grid itself is built on E14.5 and never
in CI."""
from __future__ import annotations

import random

import numpy as np
import pytest

from haemolynx.graph import assemble, prune
from haemolynx.graph.assemble import lumen_cleanup
from lumen_artefact_fixtures import VOXEL_SIZE, hair_along_a_wall
from stub_protrusion_review import build_grid_s, prune_spy


def _cleanup(mask):
    return lumen_cleanup(mask, VOXEL_SIZE, image_shape=mask.shape, min_stub_length=10.0,
                         min_stub_length_radius_multiple=3.0)


def _spied(monkeypatch, tmp_path, mask, G):
    """The clean-up's own stub prune, once, on *G* with the spy round it."""
    cleanup = _cleanup(mask)
    # Registered first, so the spy is taken off again after the test.
    monkeypatch.setattr(assemble, "prune_vascular_stubs", assemble.prune_vascular_stubs)
    prune_spy.install(assemble, prune, tmp_path, (np.asarray(mask.shape) - 1) * np.asarray(VOXEL_SIZE))
    H = cleanup.prune(G.copy())
    return H, build_grid_s.read_records(tmp_path)


def _record_at(records, tip):
    rows = [r for r in records if np.allclose(r["tip"], tip, atol=0.6)]
    assert rows, f"no record of the dead end at {tip}"
    return rows[0]


def _edges(G):
    return sorted(str((u, v)) for u, v in G.edges())


def test_the_spy_leaves_the_prune_as_it_was(monkeypatch, tmp_path):
    mask, G = hair_along_a_wall()
    plain = _cleanup(mask).prune(G.copy())

    spied, records = _spied(monkeypatch, tmp_path, mask, G)

    assert _edges(spied) == _edges(plain)
    assert len(records) == 3 and (tmp_path / "call_001.pkl").exists()


def test_the_protrusion_test_cannot_see_a_long_hair_and_the_uncapped_reading_takes_a_collinear_end(
    monkeypatch, tmp_path,
):
    """The 2026-10-11 finding, pinned on the hair fixture. Today's test reads
    the 16 nearest centreline samples with the stub's own masked out: along
    the 25 um hair those are all its own, so it is never judged and the
    prune keeps it (the doubled-stretch step takes it in the clean-up).
    Asking for 16 samples of other centrelines judges the hair (it never
    leaves the trunk's lumen) -- and the trunk's own straight end past the
    junction, beside the hair, at the image face, as well."""
    mask, G = hair_along_a_wall()

    H, records = _spied(monkeypatch, tmp_path, mask, G)

    hair = _record_at(records, (7.0, 11.0, 54.0))
    assert hair["ratio_capped"] == float("inf") and not hair["removed"] and H.has_node((7.0, 11.0, 54.0))
    assert hair["ratio_others"] < 0.0
    assert hair["deviation_deg"] == pytest.approx(45.0, abs=1.0)
    # 25 um against three radii of the 6 um trunk: the length rule keeps it.
    assert hair["length_um"] > hair["threshold_um"]
    end = _record_at(records, (7.0, 7.0, 57.0))
    assert end["deviation_deg"] == pytest.approx(0.0, abs=1.0) and end["face_um"] == pytest.approx(2.0)
    assert end["ratio_capped"] == float("inf") and end["ratio_others"] < 1.0
    far = _record_at(records, (7.0, 7.0, 2.0))
    assert far["ratio_others"] > 1.0


def _row(**overrides):
    row = dict(call=1, tip=[10.0, 10.0, 10.0], junction=[10.0, 10.0, 0.0], length_um=12.0, threshold_um=9.0,
               face_um=20.0, ratio_capped=0.5, ratio_others=0.5, deviation_deg=60.0, half_off_mask=False,
               tip_near_junction=False, removed=False)
    row.update(overrides)
    return row


def test_the_pool_holds_only_what_the_protrusion_test_decides():
    kept = build_grid_s.pool([
        _row(),
        _row(tip=[0.0, 30.0, 30.0], half_off_mask=True),                         # off the mask first
        _row(tip=[0.0, 40.0, 40.0], tip_near_junction=True),                     # in its parent's lumen
        _row(tip=[0.0, 50.0, 50.0], length_um=5.0),                              # the length rule
        _row(tip=[0.0, 60.0, 60.0], ratio_capped=2.5, ratio_others=2.0),         # sticks out
        _row(tip=[0.0, 70.0, 70.0], length_um=5.0, face_um=4.0),                 # short, at a face
        _row(tip=[0.0, 80.0, 80.0], ratio_capped=float("inf"), ratio_others=0.4),  # never judged today
    ])
    tips = sorted(tuple(s["tip"]) for s in kept)
    assert tips == [(0.0, 70.0, 70.0), (0.0, 80.0, 80.0), (10.0, 10.0, 10.0)]
    by_tip = {tuple(s["tip"]): s for s in kept}
    assert by_tip[(0.0, 70.0, 70.0)]["at_face"]
    assert not by_tip[(0.0, 80.0, 80.0)]["capped_removes"] and by_tip[(0.0, 80.0, 80.0)]["others_removes"]


def test_a_stub_seen_by_several_calls_is_one_case_kept_from_the_call_that_removed_it():
    stubs = build_grid_s.pool([
        _row(call=1, removed=False),
        _row(call=2, tip=[10.3, 10.0, 10.0], removed=True, ratio_capped=0.2),
        _row(call=3, removed=False),
    ])
    assert len(stubs) == 1
    assert stubs[0]["call"] == 2 and stubs[0]["removed_by_call"] and stubs[0]["ratio_capped"] == 0.2


def test_the_cases_are_spread_round_robin_over_the_cells():
    stubs = []
    for i in range(40):
        ratio = (0.2, 0.7, 1.2, 1.7)[i % 4]
        stubs.append(_row(tip=[5.0 * i, 0.0, 0.0], ratio_others=ratio, deviation_deg=10.0 if i < 8 else 90.0))
    stubs = build_grid_s.pool(stubs)

    picked = build_grid_s.spread(stubs, 12, random.Random(1))

    cells = [build_grid_s.cell(s) for s in picked]
    assert len(picked) == 12
    # Eight cells hold stubs (four readings, collinear or not): each gives one
    # before any gives a third.
    assert len(set(cells)) == 8 and max(cells.count(c) for c in set(cells)) == 2
    assert [s["tip"] for s in build_grid_s.spread(stubs, 12, random.Random(1))] == [s["tip"] for s in picked]
