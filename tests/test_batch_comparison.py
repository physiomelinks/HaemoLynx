"""A batch's statistics side by side: conditions read from file names, each
image's statistics CSVs read back, the summaries, the two-condition tests and
the three CSVs (`statistics/batch_comparison.py`)."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from haemolynx.statistics import export_branch_order_statistics_to_csv, export_statistics_to_csv
from haemolynx.statistics.batch_comparison import (
    ALL_IMAGES,
    BOTH_CONDITIONS,
    BY_IMAGE_FILENAME,
    COMPARISON_FILENAME,
    PAIRS_FILENAME,
    SUMMARY_FILENAME,
    BatchConditions,
    Pair,
    benjamini_hochberg,
    collect_batch_statistics,
    compare_paired,
    compare_two,
    parse_condition,
    read_branch_statistics_csv,
    read_statistics_csv,
    smallest_mann_whitney_p,
    smallest_wilcoxon_p,
    summarise,
    write_batch_statistics,
)


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _dict_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_image(root: Path, stem: str, length: float, degree: float, constant: float = 3.0) -> None:
    """One image's statistics files, as export_results writes them."""
    folder = root / stem
    export_statistics_to_csv(
        {
            "Total Edge Length (microns)": length,
            "Average Degree": degree,
            "Murray Law Exponent": constant,
            "Path Efficiency": "N/A (disconnected graph)",
        },
        folder / f"{stem}_statistics.csv",
    )
    export_branch_order_statistics_to_csv(
        {"BO1": {"Edge Count": 4, "Mean Length (microns)": length / 10.0}},
        folder / f"{stem}_branch_statistics.csv",
    )


# -- conditions ---------------------------------------------------------------


def test_a_condition_is_its_phrases_ignoring_case():
    condition = parse_condition(" WT ,  Wildtype,, ")
    assert condition.label == "WT, Wildtype"
    assert condition.phrases == ("wt", "wildtype")
    assert condition.holds("E14_wt_01_Simple Segmentation")
    assert condition.holds("E14_WILDTYPE_2")
    assert not condition.holds("E14_KO_1")
    assert parse_condition("") is None
    assert parse_condition(" , ") is None
    assert parse_condition(None) is None


def test_each_image_is_in_one_condition_both_or_neither():
    conditions = BatchConditions.from_text("WT", "KO")
    assert conditions.compares
    assert conditions.condition_of("E14_WT_1") == "WT"
    assert conditions.condition_of("E14_ko_1") == "KO"
    assert conditions.condition_of("E14_WT_KO_1") == BOTH_CONDITIONS
    assert conditions.condition_of("E14_HET_1") == ""
    names = ["WT_1", "WT_2", "KO_1", "HET_1", "WT_KO"]
    assert conditions.counts(names) == {"WT": 2, "KO": 1, BOTH_CONDITIONS: 1, "": 1}
    assert conditions.describe(names) == "2 in WT, 1 in KO, 1 in both, 1 in neither"
    # No condition: every image in none, nothing described.
    assert BatchConditions().condition_of("WT_1") == ""
    assert BatchConditions().describe(names) == ""
    # One condition: not a comparison.
    assert not BatchConditions.from_text("WT", "").compares


def test_conditions_that_cannot_be_told_apart_are_refused():
    assert "Both conditions" in BatchConditions.from_text("WT", "wt").problem()
    assert ALL_IMAGES in BatchConditions.from_text("All images", "KO").problem()
    assert BatchConditions.from_text("WT", "KO").problem() == ""


# -- reading each image's statistics ------------------------------------------


def test_only_numbers_are_read_from_a_statistics_csv(tmp_path):
    path = export_statistics_to_csv(
        {
            "Total Nodes": 12,
            "Total Edge Length (microns)": 340.5,
            "Path Efficiency": "N/A (disconnected graph)",
            "Top Nodes": [1, 2, 3],
            "Is Tree": True,
            "Pair Coverage": 0.25,
            "Strahler Network": {"Orders": {"1": {"Element Count": 7}}},
            # Full mode's per-node rows are named by node ID: not comparable.
            "17": 0.5,
            "Not finite": float("nan"),
        },
        tmp_path / "a_statistics.csv",
    )
    values = read_statistics_csv(path)
    assert values == {
        ("Core Statistics", "Total Nodes", ""): 12.0,
        ("Core Statistics", "Total Edge Length", "microns"): 340.5,
        ("Core Statistics", "Pair Coverage", ""): pytest.approx(0.25),
        ("Strahler Network > Orders > 1", "Element Count", ""): 7.0,
    }


def test_a_metric_met_twice_in_one_file_is_numbered(tmp_path):
    path = tmp_path / "a_statistics.csv"
    path.write_text(
        "Section,Metric,Value,Unit,Notes\n"
        "Core Statistics,Fractal Dimension,1.8,,\n"
        "Core Statistics,Fractal Dimension,1.6,,\n",
        encoding="utf-8",
    )
    assert read_statistics_csv(path) == {
        ("Core Statistics", "Fractal Dimension", ""): 1.8,
        ("Core Statistics", "Fractal Dimension (2)", ""): 1.6,
    }


def test_each_branch_order_is_a_section(tmp_path):
    path = export_branch_order_statistics_to_csv(
        {
            "BO1": {
                "Edge Count": 9,
                "Mean Length (microns)": 11.5,
                "Mean Tortuosity Index": 1.2,
                "Mean Emergence Angle (degrees)": "N/A (no unique parent junction)",
                "Mean Diameter (microns)": 5.0,
            },
        },
        tmp_path / "a_branch_statistics.csv",
    )
    values = read_branch_statistics_csv(path)
    assert values[("Branch Order BO1", "Edge Count", "")] == 9.0
    assert values[("Branch Order BO1", "Mean Length", "microns")] == 11.5
    assert values[("Branch Order BO1", "Mean Diameter", "microns")] == 5.0
    assert ("Branch Order BO1", "Mean Emergence Angle", "degrees") not in values
    assert not any(metric == "Notes" for _section, metric, _unit in values)
    assert not any(section.startswith("Branch Order #") for section, _m, _u in values)


def test_an_image_with_no_statistics_is_reported_missing(tmp_path):
    _write_image(tmp_path, "WT_1", 100.0, 2.0)
    images, missing = collect_batch_statistics(
        tmp_path, ["WT_1", "KO_1"], BatchConditions.from_text("WT", "KO")
    )
    assert [image.stem for image in images] == ["WT_1"]
    assert images[0].condition == "WT"
    assert missing == ["KO_1"]


# -- summaries and tests ------------------------------------------------------


def test_summarise_against_hand_computed_values():
    summary = summarise([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert summary["N"] == 8
    assert summary["Mean"] == pytest.approx(5.0)
    assert summary["SD"] == pytest.approx(math.sqrt(32 / 7))
    assert summary["SEM"] == pytest.approx(math.sqrt(32 / 7) / math.sqrt(8))
    assert summary["Median"] == pytest.approx(4.5)
    assert summary["Q1"] == pytest.approx(4.0)
    assert summary["Q3"] == pytest.approx(5.5)
    assert summary["IQR"] == pytest.approx(1.5)
    assert (summary["Min"], summary["Max"], summary["Range"]) == (2.0, 9.0, 7.0)
    assert summary["CV"] == pytest.approx(math.sqrt(32 / 7) / 5.0)
    one = summarise([3.0])
    assert one["N"] == 1 and one["Median"] == 3.0 and math.isnan(one["SD"]) and math.isnan(one["CV"])
    assert summarise([])["N"] == 0


def test_the_smallest_p_a_mann_whitney_test_can_reach():
    assert smallest_mann_whitney_p(3, 3) == pytest.approx(0.1)
    assert smallest_mann_whitney_p(5, 5) == pytest.approx(2 / 252)
    assert smallest_mann_whitney_p(2, 2) == pytest.approx(1 / 3)
    assert math.isnan(smallest_mann_whitney_p(0, 3))


def test_two_conditions_apart_against_hand_computed_tests():
    test = compare_two([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    # Every b above every a: U for a is 0, the exact two-sided p 2 / C(6, 3).
    assert test.u == 0.0
    assert test.p == pytest.approx(0.1)
    assert test.smallest_p == pytest.approx(0.1)
    assert test.rank_biserial == pytest.approx(1.0)
    # Welch: means 2 and 5, variances 1 and 1 -> t = 3 / sqrt(2 / 3).
    assert test.welch_t == pytest.approx(3.0 / math.sqrt(2.0 / 3.0))
    # Hedges' g: (5 - 2) / 1, times 1 - 3 / (4 * 6 - 9).
    assert test.hedges_g == pytest.approx(3.0 * (1.0 - 3.0 / 15.0))
    assert "no difference can reach p < 0.05" in test.note
    reversed_test = compare_two([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    assert reversed_test.rank_biserial == pytest.approx(-1.0)
    assert reversed_test.hedges_g < 0
    larger = compare_two([1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.5])
    assert larger.p == pytest.approx(2 / 252)
    assert larger.note == ""


def test_overlapping_conditions_and_ties_give_a_p_between_zero_and_one():
    test = compare_two([1.0, 1.0, 2.0, 5.0], [2.0, 3.0, 3.0, 1.5])
    assert 0.0 < test.p < 1.0
    assert -1.0 < test.rank_biserial < 1.0


def test_what_is_not_tested_says_why():
    too_few = compare_two([1.0], [2.0, 3.0])
    assert not too_few.tested and "fewer than 2 images" in too_few.note
    constant = compare_two([3.0, 3.0], [3.0, 3.0, 3.0])
    assert not constant.tested and "same value" in constant.note
    # Each condition constant but different: the rank test runs, Welch cannot.
    split = compare_two([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert split.tested and math.isnan(split.welch_p) and math.isnan(split.hedges_g)


def test_benjamini_hochberg_against_hand_computed_q_values():
    q = benjamini_hochberg([0.01, 0.04, float("nan"), 0.03, 0.5])
    # Ranked .01 .03 .04 .5 of m = 4: .04, .06, .0533, .5, then made monotone.
    assert q[0] == pytest.approx(0.04)
    assert q[1] == pytest.approx(0.04 * 4 / 3)
    assert math.isnan(q[2])
    assert q[3] == pytest.approx(0.04 * 4 / 3)
    assert q[4] == pytest.approx(0.5)
    assert benjamini_hochberg([]) == []


# -- the CSVs -------------------------------------------------------------------


def _batch(root: Path) -> list[str]:
    for stem, length, degree in (
        ("E14_WT_1", 100.0, 2.0),
        ("E14_KO_1", 210.0, 2.4),
        ("E14_WT_2", 110.0, 2.1),
        ("E14_KO_2", 220.0, 2.2),
        ("E14_WT_3", 120.0, 2.3),
        ("E14_KO_3", 230.0, 2.0),
        ("E14_HET_1", 150.0, 2.1),
    ):
        _write_image(root, stem, length, degree)
    return ["E14_WT_1", "E14_KO_1", "E14_WT_2", "E14_KO_2", "E14_WT_3", "E14_KO_3", "E14_HET_1", "E14_gone"]


def test_two_conditions_write_by_image_summary_and_comparison(tmp_path):
    stems = _batch(tmp_path)
    files = write_batch_statistics(tmp_path, stems, BatchConditions.from_text("WT", "KO"))
    assert files.written == (
        tmp_path / BY_IMAGE_FILENAME,
        tmp_path / SUMMARY_FILENAME,
        tmp_path / COMPARISON_FILENAME,
    )
    assert files.images == 7 and files.missing == ("E14_gone",)

    by_image = _rows(tmp_path / BY_IMAGE_FILENAME)
    # Each condition's images together, then the rest, in batch order.
    assert by_image[0] == [
        "Section", "Metric", "Unit",
        "E14_WT_1", "E14_WT_2", "E14_WT_3", "E14_KO_1", "E14_KO_2", "E14_KO_3", "E14_HET_1",
    ]
    assert by_image[1] == ["", "Condition", "", "WT", "WT", "WT", "KO", "KO", "KO", ""]
    length_row = next(row for row in by_image if row[1] == "Total Edge Length")
    assert length_row[2] == "microns"
    assert length_row[3:] == ["100", "110", "120", "210", "220", "230", "150"]
    assert not any(row[1] == "Path Efficiency" for row in by_image)
    assert any(row[0] == "Branch Order BO1" and row[1] == "Mean Length" for row in by_image)

    summary = _dict_rows(tmp_path / SUMMARY_FILENAME)
    length = {row["Group"]: row for row in summary if row["Metric"] == "Total Edge Length"}
    assert list(length) == ["WT", "KO", ALL_IMAGES]
    assert (length["WT"]["N"], length["WT"]["Median"], length["WT"]["Min"], length["WT"]["Max"]) == (
        "3", "110", "100", "120",
    )
    assert length["WT"]["Range"] == "20"
    assert float(length["KO"]["SD"]) == pytest.approx(10.0)
    assert length[ALL_IMAGES]["N"] == "7"
    assert float(length[ALL_IMAGES]["Mean"]) == pytest.approx(np.mean([100, 110, 120, 210, 220, 230, 150]))

    comparison = _dict_rows(tmp_path / COMPARISON_FILENAME)
    row = next(r for r in comparison if r["Metric"] == "Total Edge Length")
    assert (row["N WT"], row["N KO"], row["Median WT"], row["Median KO"]) == ("3", "3", "110", "220")
    assert (row["Min KO"], row["Max KO"]) == ("210", "230")
    assert float(row["Difference of medians (KO - WT)"]) == pytest.approx(110.0)
    assert float(row["Ratio of medians (KO / WT)"]) == pytest.approx(2.0)
    assert float(row["P (Mann-Whitney U, two-sided)"]) == pytest.approx(0.1)
    assert float(row["Rank-biserial r (+: KO larger)"]) == pytest.approx(1.0)
    assert float(row["Smallest possible P"]) == pytest.approx(0.1)
    assert "no difference can reach p < 0.05" in row["Note"]
    # A setting the same in every image is listed, not tested.
    murray = next(r for r in comparison if r["Metric"] == "Murray Law Exponent")
    assert murray["P (Mann-Whitney U, two-sided)"] == "" and "same value" in murray["Note"]
    # q-values over the metrics tested.
    tested = [r for r in comparison if r["P (Mann-Whitney U, two-sided)"]]
    assert files.tested == len(tested)
    expected_q = benjamini_hochberg([float(r["P (Mann-Whitney U, two-sided)"]) for r in tested])
    assert [float(r["Q (Benjamini-Hochberg)"]) for r in tested] == pytest.approx(expected_q)
    assert "KO against WT" in files.message and "E14_gone" in files.message


def test_without_conditions_every_image_is_summarised_and_nothing_tested(tmp_path):
    stems = _batch(tmp_path)
    files = write_batch_statistics(tmp_path, stems)
    assert files.comparison is None and not (tmp_path / COMPARISON_FILENAME).exists()
    summary = _dict_rows(tmp_path / SUMMARY_FILENAME)
    assert {row["Group"] for row in summary} == {ALL_IMAGES}
    length = next(row for row in summary if row["Metric"] == "Total Edge Length")
    assert (length["N"], length["Min"], length["Max"], length["Range"]) == ("7", "100", "230", "130")


def test_one_condition_is_summarised_beside_all_images(tmp_path):
    stems = _batch(tmp_path)
    files = write_batch_statistics(tmp_path, stems, BatchConditions.from_text("WT", None))
    assert files.comparison is None
    groups = {row["Group"] for row in _dict_rows(tmp_path / SUMMARY_FILENAME)}
    assert groups == {"WT", ALL_IMAGES}
    assert "give a second" in files.message


def test_a_batch_with_no_statistics_writes_nothing(tmp_path):
    files = write_batch_statistics(tmp_path, ["A", "B"], BatchConditions.from_text("A", "B"))
    assert files.written == ()
    assert "Statistics on" in files.message
    assert not (tmp_path / SUMMARY_FILENAME).exists()


# -- paired by subject ------------------------------------------------------------

#: Harvey's AIA batch, 2026-10-19: two subjects with both sides imaged.
AIA = [
    "28_3_d_contra_patella_cropped_hdf5_Simple Segmentation_all_vessel",
    "28_3_d_ipsi_FITC_Simple Segmentation_all_vessel",
    "29_1_b_contra_FITC_Simple Segmentation_all_vessel",
    "29_1_f_contra_FITC_Simple Segmentation_all_vessel",
    "29_1_f_ipsi_FITC_Simple Segmentation_all_vessel",
    "29_2_c_ipsi_FITC_Simple Segmentation_all_vessel",
    "29_2_d_ipsi_FITC_Simple Segmentation_all_vessel",
    "29_2_f_contra_FITC_Simple Segmentation_all_vessel",
]


def test_a_subject_is_the_name_before_its_condition_phrase():
    conditions = BatchConditions.from_text("contra", "ipsi", paired=True)
    assert conditions.design == "paired"
    assert conditions.subject_of(AIA[0]) == "28_3_d"
    assert conditions.subject_of("29_1_F-IPSI.FITC") == "29_1_f"
    assert conditions.subject_of("contra_29_1_f") == ""  # the phrase begins the name
    assert conditions.subject_of("29_1_f_FITC") == ""  # in neither condition
    pattern = BatchConditions.from_text(
        "contra", "ipsi", paired=True, subject_pattern=r"(\d+_\d+_[a-z])_"
    )
    assert pattern.subject_of("side_contra_29_1_f_FITC") == "29_1_f"
    assert pattern.subject_of("side_contra_FITC") == ""
    whole = BatchConditions.from_text("WT", "KO", paired=True, subject_pattern=r"m\d+")
    assert whole.subject_of("E14_WT_M07") == "m07"


def test_the_aia_batch_pairs_two_subjects_and_lists_the_rest():
    conditions = BatchConditions.from_text("contra", "ipsi", paired=True)
    pairing = conditions.pairing(AIA)
    assert pairing.pairs == (
        Pair("28_3_d", AIA[0], AIA[1]),
        Pair("29_1_f", AIA[3], AIA[4]),
    )
    assert [(name[:6], subject, reason) for name, subject, reason in pairing.unpaired] == [
        ("29_1_b", "29_1_b", "no ipsi image of subject 29_1_b"),
        ("29_2_c", "29_2_c", "no contra image of subject 29_2_c"),
        ("29_2_d", "29_2_d", "no contra image of subject 29_2_d"),
        ("29_2_f", "29_2_f", "no ipsi image of subject 29_2_f"),
    ]
    assert pairing.partner_of(AIA[4]) == AIA[3]
    assert pairing.reason_left_out(AIA[2]) == "no ipsi image of subject 29_1_b"
    assert conditions.describe(AIA).endswith(
        "2 pair(s) by subject (28_3_d, 29_1_f), 4 image(s) with no partner"
    )


def test_every_image_that_cannot_pair_says_why():
    conditions = BatchConditions.from_text("WT", "KO", paired=True)
    pairing = conditions.pairing(
        ["m1_WT_a", "m1_WT_b", "m1_KO", "m2_WT", "WT_m3", "m4_HET", "m5_WT_KO", "m6_KO"]
    )
    assert pairing.pairs == ()
    reasons = {name: reason for name, _subject, reason in pairing.unpaired}
    assert reasons == {
        "m1_WT_a": "subject m1 has 2 WT images: which one pairs is unclear",
        "m1_WT_b": "subject m1 has 2 WT images: which one pairs is unclear",
        "m1_KO": "subject m1 has 2 WT images: which one pairs is unclear",
        "m2_WT": "no KO image of subject m2",
        "WT_m3": "no subject ID: the condition phrase begins the name",
        "m4_HET": "in neither condition",
        "m5_WT_KO": "in both conditions",
        "m6_KO": "no WT image of subject m6",
    }


def test_a_paired_design_needs_two_conditions_and_a_working_pattern():
    assert "needs both conditions" in BatchConditions.from_text("WT", None, paired=True).problem()
    assert "not a regular expression" in BatchConditions.from_text(
        "WT", "KO", paired=True, subject_pattern="(unclosed"
    ).problem()
    assert BatchConditions.from_text("WT", "KO", paired=True).problem() == ""


def test_the_smallest_p_a_signed_rank_test_can_reach():
    assert smallest_wilcoxon_p(2) == pytest.approx(0.5)
    assert smallest_wilcoxon_p(5) == pytest.approx(0.0625)
    assert smallest_wilcoxon_p(6) == pytest.approx(0.03125)
    assert math.isnan(smallest_wilcoxon_p(0))


def test_paired_tests_against_hand_computed_values():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [2.0, 3.0, 4.5, 6.0, 7.0, 9.0]  # differences 1 1 1.5 2 2 3, all positive
    test = compare_paired(a, b)
    assert test.n_pairs == 6
    assert test.w_plus == pytest.approx(21.0)  # 1 + 2 + ... + 6
    assert test.p == pytest.approx(2 / 2**6)  # exact: every sign the same
    assert test.smallest_p == pytest.approx(2 / 2**6)
    assert test.rank_biserial == pytest.approx(1.0)
    d = np.array(b) - np.array(a)
    assert test.cohen_dz == pytest.approx(d.mean() / d.std(ddof=1))
    assert test.t == pytest.approx(d.mean() / (d.std(ddof=1) / math.sqrt(6)))
    assert test.note == ""
    flipped = compare_paired(b, a)
    assert flipped.rank_biserial == pytest.approx(-1.0) and flipped.cohen_dz < 0
    two = compare_paired([1.0, 2.0], [2.0, 4.0])
    assert two.p == pytest.approx(0.5)
    assert "no difference can reach p < 0.05 (smallest possible p 0.5)" in two.note


def test_paired_differences_that_cannot_be_tested_say_why():
    assert "fewer than 2 pairs" in compare_paired([1.0], [2.0]).note
    same = compare_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert not same.tested and "same value in both images" in same.note
    one_tie = compare_paired([1.0, 2.0, 3.0], [1.0, 3.0, 4.0])
    assert one_tie.tested and "1 pair(s) with no difference left out" in one_tie.note


def test_a_paired_batch_writes_its_pairs_and_tests_over_them(tmp_path):
    lengths = {
        "s1_contra": 100.0, "s1_ipsi": 130.0,
        "s2_contra": 110.0, "s2_ipsi": 150.0,
        "s3_contra": 120.0, "s3_ipsi": 140.0,
        "s4_contra": 90.0,  # no ipsi image
    }
    for stem, length in lengths.items():
        _write_image(tmp_path, stem, length, 2.0)
    conditions = BatchConditions.from_text("contra", "ipsi", paired=True)
    files = write_batch_statistics(tmp_path, list(lengths), conditions)
    assert files.pairs == tmp_path / PAIRS_FILENAME
    pairs = _dict_rows(tmp_path / PAIRS_FILENAME)
    assert [
        (row["Subject"], row["contra image"], row["ipsi image"], row["Left out because"])
        for row in pairs
    ] == [
        ("s1", "s1_contra", "s1_ipsi", ""),
        ("s2", "s2_contra", "s2_ipsi", ""),
        ("s3", "s3_contra", "s3_ipsi", ""),
        ("s4", "s4_contra", "", "no ipsi image of subject s4"),
    ]
    by_image = _rows(tmp_path / BY_IMAGE_FILENAME)
    assert by_image[2][:4] == ["", "Subject", "", "s1"]
    comparison = _dict_rows(tmp_path / COMPARISON_FILENAME)
    row = next(r for r in comparison if r["Metric"] == "Total Edge Length")
    assert row["N pairs"] == "3"
    assert row["Median contra"] == "110"  # over the paired images: s4 is left out
    assert float(row["Median of paired differences (ipsi - contra)"]) == pytest.approx(30.0)
    assert float(row["Wilcoxon W+ (pairs with ipsi larger)"]) == pytest.approx(6.0)
    assert float(row["P (Wilcoxon signed-rank, two-sided)"]) == pytest.approx(0.25)
    assert float(row["Smallest possible P"]) == pytest.approx(0.25)
    assert "no difference can reach p < 0.05" in row["Note"]
    assert float(row["Q (Benjamini-Hochberg)"]) >= float(row["P (Wilcoxon signed-rank, two-sided)"])
    # The summary still covers every image of each condition, paired or not.
    summary = _dict_rows(tmp_path / SUMMARY_FILENAME)
    contra = next(
        r for r in summary if r["Metric"] == "Total Edge Length" and r["Group"] == "contra"
    )
    assert contra["N"] == "4"
    assert "paired by subject: 3 pair(s) (s1, s2, s3)" in files.message
    assert "left out with no partner: s4_contra" in files.message


def test_an_independent_batch_writes_no_pairs(tmp_path):
    stems = _batch(tmp_path)
    files = write_batch_statistics(tmp_path, stems, BatchConditions.from_text("WT", "KO"))
    assert files.pairs is None and not (tmp_path / PAIRS_FILENAME).exists()
    assert "Mann-Whitney U" in _rows(tmp_path / COMPARISON_FILENAME)[0]
