"""The G4 dead-end grid's candidate measures (``scripts/dead_end_review/build_g4.py``):
how bright a stain reads along a chain against its own background, on volumes
whose answer is known."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dead_end_review import build_g4
from lumen_artefact_fixtures import capsule

VS = (1.0, 1.0, 1.0)
SHAPE = (24, 40, 60)


def _volume_with_a_vessel(brightness=100.0, *, noise=0.0, seed=0):
    """A segmented vessel of radius 2.5 along x at (z, y) = (12, 12), filled
    at *brightness*; the background 10, plus *noise*."""
    rng = np.random.default_rng(seed)
    mask = capsule(np.zeros(SHAPE, dtype=bool), (12, 12, -5), (12, 12, 65), 2.5)
    volume = np.full(SHAPE, 10.0) + rng.normal(0.0, noise, SHAPE) if noise else np.full(SHAPE, 10.0)
    volume[mask] = brightness
    return volume, mask


def _path(y, x0=10.0, x1=40.0):
    return np.column_stack([np.full(61, 12.0), np.full(61, float(y)), np.linspace(x0, x1, 61)])


def test_a_chain_along_a_bright_vessel_reads_its_brightness_over_its_background():
    volume, mask = _volume_with_a_vessel()
    level = build_g4.channel_level(volume, mask, _path(12), 2.5, VS)
    assert level["inside"] == pytest.approx(100.0)
    assert level["background"] == pytest.approx(10.0)
    assert build_g4.channel_contrast(level, reference=90.0) == pytest.approx(1.0)


def test_a_chain_through_background_reads_no_contrast():
    """The case G4 asks about: a chain where no stain says there is a vessel."""
    volume, mask = _volume_with_a_vessel(noise=2.0)
    level = build_g4.channel_level(volume, mask, _path(30), 2.0, VS)
    assert abs(build_g4.channel_contrast(level, reference=90.0)) < 0.1


def test_a_segmented_neighbour_is_left_out_of_the_background():
    """The shell 4-8 um round a chain beside another vessel would read that
    vessel; the mask (grown by 1.5 um) keeps it out of the background."""
    volume, mask = _volume_with_a_vessel()
    level = build_g4.channel_level(volume, mask, _path(19), 1.5, VS)  # 7 um from the vessel's axis
    assert level["background"] == pytest.approx(10.0)


def test_an_empty_path_or_reference_reads_nothing():
    volume, mask = _volume_with_a_vessel()
    assert np.isnan(build_g4.channel_level(volume, mask, np.empty((0, 3)), 2.0, VS)["inside"])
    assert np.isnan(build_g4.channel_contrast({"inside": 5.0, "background": 1.0}, reference=0.0))


def test_the_grid_is_spread_across_both_stains():
    """Ten from each third of the endothelial contrast, each ten spread
    across the dextran contrast."""
    rng = np.random.default_rng(1)
    ends = [SimpleNamespace(tip=i) for i in range(300)]
    measures = {i: {"dextran_tip_contrast": float(rng.uniform(0, 1)), "endothelium_tip_contrast": float(rng.uniform(0, 1))}
                for i in range(300)}
    chosen = build_g4.choose(ends, measures, count=30)
    assert len({e.tip for e in chosen}) == 30
    endo = np.asarray([measures[e.tip]["endothelium_tip_contrast"] for e in chosen])
    cuts = np.percentile([m["endothelium_tip_contrast"] for m in measures.values()], [100 / 3, 200 / 3])
    assert np.histogram(endo, bins=[-np.inf, *cuts, np.inf])[0].tolist() == [10, 10, 10]
    dextran = sorted(measures[e.tip]["dextran_tip_contrast"] for e in chosen)
    assert dextran[0] < 0.1 and dextran[-1] > 0.9


# --- the fresh grids G1'-G3' (build_grids.py --fresh, fit_marks.FRESH_RULES) -----------


def _end(tip, **values):
    base = dict(tip=tip, kind="unknown", tip_um=(50.0, float(tip), 0.0), chain_length_um=10.0,
                junction_radius_um=2.0, beside_share=0.0, face_distance_um=None, ahead_um=None,
                ahead_turn_deg=None)
    base.update(values)
    return SimpleNamespace(**base)


def test_the_tips_earlier_grids_showed_are_read_from_their_keys_in_microns(tmp_path):
    from dead_end_review import build_grids

    key = tmp_path / "G1_key.csv"
    key.write_text('id,tip_zyx_vox\nG1-01,"(10, 20, 30)"\nG1-02,\n', encoding="utf-8")
    (tmp_path / "other.csv").write_text("id,kind\nX-1,unknown\n", encoding="utf-8")

    shown = build_grids.shown_tips_um([key, tmp_path / "other.csv"], (1.0, 0.5, 0.5))

    assert shown.tolist() == [[10.0, 10.0, 15.0]]


def test_a_dead_end_an_earlier_grid_showed_is_left_out_of_the_fresh_grids():
    from dead_end_review import build_grids

    ends = [_end(0, tip_um=(0.0, 0.0, 0.0)), _end(1, tip_um=(0.0, 0.0, 2.9)), _end(2, tip_um=(0.0, 0.0, 3.1))]
    kept = build_grids._not_shown(ends, np.asarray([[0.0, 0.0, 0.0]]))
    assert [e.tip for e in kept] == [2]


def test_each_fresh_grid_is_spread_half_each_side_of_the_rule_it_checks():
    from dead_end_review import build_grids
    from fit_marks import FRESH_RULES

    rng = np.random.default_rng(3)
    ends = {}
    for tip in range(400):
        depth = 10.0 if tip % 2 else 120.0
        ends[tip] = _end(
            tip, tip_um=(depth, float(tip), 0.0), chain_length_um=float(rng.uniform(3, 60)),
            beside_share=float(rng.choice([0.0, rng.uniform(0.02, 0.9)])),
            kind=str(rng.choice(["unknown", "leaves_image", "stops_short"])),
            face_distance_um=float(rng.uniform(0.0, 40.0)),
            ahead_um=float(rng.uniform(1, 10)) if tip % 3 else None,
            ahead_turn_deg=float(rng.uniform(0, 60)),
        )
    pages = build_grids.fresh_pages(ends, [], np.empty((0, 3)))

    g1 = [e for e, _x in pages["G1b"]]
    assert len(g1) == 30 and all(e.kind not in ("leaves_image", "island_end") for e in g1)
    beside = FRESH_RULES["G1b"][0]
    for surface in (True, False):
        layer = [e for e in g1 if (e.tip_um[0] < build_grids.SURFACE_DEPTH_UM) == surface]
        assert len(layer) == 15
        assert sum(beside.says_positive({"beside_share": e.beside_share}) for e in layer) == 7
    g2 = [e for e, _x in pages["G2b"]]
    assert len(g2) == 30 and all(2.0 <= e.face_distance_um <= 25.0 for e in g2)
    assert sum(e.kind == "leaves_image" for e in g2) == 15
    g3 = [e for e, _x in pages["G3b"]]
    assert len(g3) == 30 and all(e.ahead_um is not None for e in g3)
    assert sum(e.ahead_turn_deg <= 30.5 for e in g3) == 15
    assert len({e.tip for page in pages.values() for e, _x in page}) == 90  # no dead end twice


def test_a_side_of_the_rule_with_too_few_is_made_up_from_the_other():
    from dead_end_review import build_grids

    pool = [_end(t, beside_share=0.5 if t < 3 else 0.0) for t in range(40)]
    chosen = build_grids._halves(pool, 15, lambda e: e.beside_share > 0.0104, lambda e: e.tip)
    assert len(chosen) == 15
    assert sum(e.beside_share > 0.0104 for e in chosen) == 3


def test_a_fixed_rule_is_scored_on_the_marks_unfitted_against_all_one_way():
    from fit_marks import check_rules

    rows = [
        {"id": "G3b-01", "ahead_turn_deg": "10"},
        {"id": "G3b-02", "ahead_turn_deg": "40"},
        {"id": "G3b-03", "ahead_turn_deg": ""},
        {"id": "G3b-04", "ahead_turn_deg": "20"},
        {"id": "G3b-05", "ahead_turn_deg": "50"},
    ]
    marks = {"G3b-01": {"verdict": "J"}, "G3b-02": {"verdict": "N"}, "G3b-03": {"verdict": "N"},
             "G3b-04": {"verdict": "N"}, "G3b-05": {"verdict": "?"}}

    (result,) = check_rules("G3b", rows, marks)

    assert result["judged"] == 4 and result["baseline"] == 3
    assert result["right"] == 3 and result["wrong"] == ["G3b-04"]


def test_the_face_rule_of_g2_prime_reads_the_classifiers_kind():
    from fit_marks import check_rules

    rows = [{"id": "a", "kind": "leaves_image", "face_distance_um": "20", "depth_um": "100"},
            {"id": "b", "kind": "unknown", "face_distance_um": "5", "depth_um": "5"}]
    marks = {"a": {"verdict": "F"}, "b": {"verdict": "I"}}

    by_rule = {c["rule"]: c["right"] for c in check_rules("G2b", rows, marks)}

    assert by_rule == {
        "the classifier's leaves_image is F": 2,
        "face_distance_um <= 11.3 is F": 0,
        "depth_um <= 10.4 is F": 0,
    }


def test_g3_prime_is_chosen_from_the_longer_look_ahead_when_given():
    from dead_end_review import build_grids

    ends = {t: _end(t) for t in range(10)}
    wider = {t: _end(t, ahead_um=12.0, ahead_turn_deg=float(t * 6)) for t in range(10)}

    pages = build_grids.fresh_pages(ends, [], np.empty((0, 3)), ahead_ends=wider)

    assert {e.tip for e, _x in pages["G3b"]} == set(range(10))
    assert all(e.ahead_um == 12.0 for e, _x in pages["G3b"])
    assert pages["G1b"] == []  # every interior end went to G3'
