"""A batch's statistics side by side: every image, each condition, and the two
conditions tested against each other.

A batch run (``gui/batch_run.py``) writes each image's statistics into its own
folder: ``{output root}/{stem}/{stem}_statistics.csv`` (Section, Metric, Value,
Unit) and ``{stem}_branch_statistics.csv`` (one row per branch order). Here
each of those numbers is one value per image, an image being one sample, and
three CSVs are written beside ``batch_summary.csv``:

- ``batch_statistics_by_image.csv`` -- every metric, one column per image,
  with each image's condition;
- ``batch_statistics_summary.csv`` -- per metric and per group (each
  condition, then all images): N, mean, SD, SEM, median, quartiles, min, max,
  range and coefficient of variation;
- ``batch_condition_comparison.csv`` -- with two conditions, per metric: each
  condition's N, median, mean and range, the difference and ratio of medians,
  a two-sided Mann-Whitney U test (an image is one sample, few images, no
  normality assumed) with Benjamini-Hochberg q-values across every metric
  tested, the smallest p the test could reach at those N, the rank-biserial
  correlation, and Welch's t test with Hedges' g beside it.

A condition is one or more phrases typed by the user; an image is in it when
its file name holds one of them, ignoring case (:class:`BatchConditions`). An
image in both conditions or in neither is left out of the test, and says so.

Only numbers are read: a metric named only by digits (the per-node rows of a
full-mode run), the Metadata section, and every text, list or N/A value are
left out. Pure: CSVs in, CSVs out.
"""
from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

BY_IMAGE_FILENAME = "batch_statistics_by_image.csv"
SUMMARY_FILENAME = "batch_statistics_summary.csv"
COMPARISON_FILENAME = "batch_condition_comparison.csv"

#: Each image's statistics files, beside its ``.haemorun``.
STATISTICS_SUFFIX = "_statistics.csv"
BRANCH_STATISTICS_SUFFIX = "_branch_statistics.csv"

#: The group every image is in, whatever its condition.
ALL_IMAGES = "All images"
#: An image whose name holds a phrase of each condition.
BOTH_CONDITIONS = "both conditions"
#: An image whose name holds neither condition's phrases.
NO_CONDITION = ""

#: A metric: (section, metric, unit).
MetricKey = tuple[str, str, str]

_SKIPPED_SECTIONS = {"Metadata"}
_NODE_ID_METRIC = re.compile(r"^-?\d+$")
_UNIT = re.compile(r"\s*\(([^()]+)\)\s*$")


# -- conditions -------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """A condition: its label (the text as typed) and the phrases that put an
    image in it."""

    label: str
    phrases: tuple[str, ...]

    def holds(self, name: str) -> bool:
        lowered = name.lower()
        return any(phrase in lowered for phrase in self.phrases)


def parse_condition(text: str | None) -> Condition | None:
    """``"WT, wildtype"`` -> a condition matching either phrase; blank -> None."""
    if text is None:
        return None
    phrases = tuple(
        dict.fromkeys(part.strip().lower() for part in str(text).split(",") if part.strip())
    )
    if not phrases:
        return None
    return Condition(", ".join(part.strip() for part in str(text).split(",") if part.strip()), phrases)


@dataclass(frozen=True)
class BatchConditions:
    """Up to two conditions, each read from the images' file names."""

    a: Condition | None = None
    b: Condition | None = None

    @classmethod
    def from_text(cls, a: str | None = None, b: str | None = None) -> "BatchConditions":
        return cls(parse_condition(a), parse_condition(b))

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return tuple(condition for condition in (self.a, self.b) if condition is not None)

    @property
    def compares(self) -> bool:
        """Whether there are two conditions to test against each other."""
        return len(self.conditions) == 2

    def problem(self) -> str:
        """Why these conditions cannot be used, or ""."""
        labels = [condition.label.lower() for condition in self.conditions]
        if len(set(labels)) < len(labels):
            return f"Both conditions are '{self.a.label}': give each its own phrase."
        if ALL_IMAGES.lower() in labels:
            return f"'{ALL_IMAGES}' is the group every image is in: name the condition otherwise."
        return ""

    def condition_of(self, name: str) -> str:
        """The label of the one condition *name* holds a phrase of;
        :data:`BOTH_CONDITIONS` when it holds both, :data:`NO_CONDITION` when
        neither (or no condition is set)."""
        held = [condition.label for condition in self.conditions if condition.holds(name)]
        if len(held) > 1:
            return BOTH_CONDITIONS
        return held[0] if held else NO_CONDITION

    def counts(self, names: Iterable[str]) -> dict[str, int]:
        """How many of *names* fall in each condition, both, and neither."""
        found = [self.condition_of(name) for name in names]
        counts = {condition.label: found.count(condition.label) for condition in self.conditions}
        counts[BOTH_CONDITIONS] = found.count(BOTH_CONDITIONS)
        counts[NO_CONDITION] = found.count(NO_CONDITION)
        return counts

    def describe(self, names: Iterable[str]) -> str:
        """``"3 in WT, 3 in KO, 1 in neither"`` (empty with no condition set)."""
        if not self.conditions:
            return ""
        counts = self.counts(names)
        parts = [f"{counts[condition.label]} in {condition.label}" for condition in self.conditions]
        if counts[BOTH_CONDITIONS]:
            parts.append(f"{counts[BOTH_CONDITIONS]} in both")
        if counts[NO_CONDITION]:
            parts.append(f"{counts[NO_CONDITION]} in neither")
        return ", ".join(parts)


# -- reading each image's statistics -------------------------------------


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        value = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _keep(values: dict[MetricKey, float], key: MetricKey, value: float) -> None:
    """Store *value* under *key*; a key met again in one file is numbered."""
    section, metric, unit = key
    repeat = 2
    while key in values:
        key = (section, f"{metric} ({repeat})", unit)
        repeat += 1
    values[key] = value


def read_statistics_csv(path: Path | str) -> dict[MetricKey, float]:
    """The numbers in one ``{stem}_statistics.csv``, by (section, metric, unit)."""
    values: dict[MetricKey, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            section = (row.get("Section") or "").strip()
            metric = (row.get("Metric") or "").strip()
            if not metric or section in _SKIPPED_SECTIONS or _NODE_ID_METRIC.match(metric):
                continue
            value = _number(row.get("Value"))
            if value is not None:
                _keep(values, (section, metric, (row.get("Unit") or "").strip()), value)
    return values


def read_branch_statistics_csv(path: Path | str) -> dict[MetricKey, float]:
    """The numbers in one ``{stem}_branch_statistics.csv``: each branch
    order's columns, as section ``Branch Order {order}``."""
    values: dict[MetricKey, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return values
        columns = []
        for name in header[1:]:
            unit = _UNIT.search(name)
            columns.append(
                (_UNIT.sub("", name).strip(), unit.group(1).strip() if unit else "")
            )
        for row in reader:
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            section = f"Branch Order {row[0].strip()}"
            for (metric, unit), text in zip(columns, row[1:]):
                if metric == "Notes":
                    continue
                value = _number(text)
                if value is not None:
                    _keep(values, (section, metric, unit), value)
    return values


@dataclass
class ImageStatistics:
    """One image of a batch: its stem, condition and every number its
    statistics files hold."""

    stem: str
    condition: str = NO_CONDITION
    values: dict[MetricKey, float] = field(default_factory=dict)


def image_statistics_files(output_root: Path | str, stem: str) -> tuple[Path, Path]:
    """``{root}/{stem}/{stem}_statistics.csv`` and its branch-order file."""
    folder = Path(output_root) / stem
    return folder / f"{stem}{STATISTICS_SUFFIX}", folder / f"{stem}{BRANCH_STATISTICS_SUFFIX}"


def collect_batch_statistics(
    output_root: Path | str,
    stems: Sequence[str],
    conditions: BatchConditions | None = None,
) -> tuple[list[ImageStatistics], list[str]]:
    """Each of *stems*' statistics from its folder under *output_root*:
    the images with any, and the stems with none (no statistics CSV)."""
    conditions = conditions or BatchConditions()
    images: list[ImageStatistics] = []
    missing: list[str] = []
    for stem in stems:
        values: dict[MetricKey, float] = {}
        for path, reader in zip(
            image_statistics_files(output_root, stem),
            (read_statistics_csv, read_branch_statistics_csv),
        ):
            if not path.is_file():
                continue
            try:
                values.update(reader(path))
            except (OSError, csv.Error, UnicodeDecodeError) as error:
                logger.warning("Could not read %s: %s", path, error)
        if values:
            images.append(ImageStatistics(stem, conditions.condition_of(stem), values))
        else:
            missing.append(stem)
    return images, missing


def metric_keys(images: Sequence[ImageStatistics]) -> list[MetricKey]:
    """Every metric any image has, in the order first met."""
    keys: dict[MetricKey, None] = {}
    for image in images:
        keys.update(dict.fromkeys(image.values))
    return list(keys)


# -- summaries and tests ----------------------------------------------------


SUMMARY_FIELDS = (
    "N", "Mean", "SD", "SEM", "Median", "Q1", "Q3", "IQR", "Min", "Max", "Range", "CV",
)


def summarise(values: Sequence[float]) -> dict[str, float]:
    """N, mean, SD (n - 1), SEM, median, quartiles (linear), IQR, min, max,
    range and coefficient of variation (SD / |mean|); NaN where undefined."""
    data = np.asarray(values, dtype=float)
    n = int(data.size)
    nan = float("nan")
    if n == 0:
        return {name: (0 if name == "N" else nan) for name in SUMMARY_FIELDS}
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if n > 1 else nan
    q1, median, q3 = (float(q) for q in np.percentile(data, [25, 50, 75]))
    low, high = float(data.min()), float(data.max())
    return {
        "N": n,
        "Mean": mean,
        "SD": sd,
        "SEM": sd / math.sqrt(n) if n > 1 else nan,
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Min": low,
        "Max": high,
        "Range": high - low,
        "CV": sd / abs(mean) if n > 1 and mean != 0 else nan,
    }


def smallest_mann_whitney_p(n_a: int, n_b: int) -> float:
    """The smallest two-sided p an exact Mann-Whitney U test can give with
    *n_a* and *n_b* samples and no ties: 2 / C(n_a + n_b, n_a)."""
    if n_a < 1 or n_b < 1:
        return float("nan")
    return min(1.0, 2.0 / math.comb(n_a + n_b, n_a))


@dataclass(frozen=True)
class TwoGroupTest:
    """Condition B against condition A on one metric; differences are B - A."""

    u: float = float("nan")
    p: float = float("nan")
    smallest_p: float = float("nan")
    rank_biserial: float = float("nan")
    welch_t: float = float("nan")
    welch_p: float = float("nan")
    hedges_g: float = float("nan")
    note: str = ""

    @property
    def tested(self) -> bool:
        return not math.isnan(self.p)


#: Fewer images than this in a condition and nothing is tested.
MIN_IMAGES_PER_CONDITION = 2


def compare_two(a: Sequence[float], b: Sequence[float]) -> TwoGroupTest:
    """Mann-Whitney U (two-sided; exact for small samples without ties, as
    scipy chooses) and Welch's t test of *b* against *a*. The rank-biserial
    correlation and Hedges' g are positive when *b* tends to be larger."""
    from scipy import stats as sp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = int(a.size), int(b.size)
    if min(n_a, n_b) < MIN_IMAGES_PER_CONDITION:
        return TwoGroupTest(
            note=f"not tested: fewer than {MIN_IMAGES_PER_CONDITION} images in a condition"
        )
    if np.ptp(np.concatenate([a, b])) == 0:
        return TwoGroupTest(note="not tested: the same value in every image")
    smallest = smallest_mann_whitney_p(n_a, n_b)
    result = sp.mannwhitneyu(a, b, alternative="two-sided")
    u_a = float(result.statistic)
    notes = []
    if smallest > 0.05:
        notes.append(
            f"with {n_a} and {n_b} images no difference can reach p < 0.05 "
            f"(smallest possible p {smallest:.3g})"
        )
    welch_t = welch_p = hedges_g = float("nan")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    if var_a + var_b > 0:
        welch = sp.ttest_ind(b, a, equal_var=False)
        welch_t, welch_p = float(welch.statistic), float(welch.pvalue)
        pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled > 0:
            correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
            hedges_g = float((b.mean() - a.mean()) / pooled * correction)
    return TwoGroupTest(
        u=u_a,
        p=float(result.pvalue),
        smallest_p=smallest,
        rank_biserial=1.0 - 2.0 * u_a / (n_a * n_b),
        welch_t=welch_t,
        welch_p=welch_p,
        hedges_g=hedges_g,
        note="; ".join(notes),
    )


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg q-values for *pvalues*, NaN left as NaN and out of
    the count."""
    q = [float("nan")] * len(pvalues)
    tested = sorted(
        (p, index) for index, p in enumerate(pvalues) if p is not None and not math.isnan(p)
    )
    m = len(tested)
    running = 1.0
    for rank in range(m, 0, -1):
        p, index = tested[rank - 1]
        running = min(running, p * m / rank)
        q[index] = running
    return q


# -- writing ----------------------------------------------------------------


@dataclass
class BatchStatisticsFiles:
    """What :func:`write_batch_statistics` wrote."""

    by_image: Path | None = None
    summary: Path | None = None
    comparison: Path | None = None
    images: int = 0
    missing: tuple[str, ...] = ()
    metrics: int = 0
    tested: int = 0
    significant: int = 0
    message: str = ""

    @property
    def written(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.by_image, self.summary, self.comparison) if path)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(value) else f"{float(value):.6g}"
    return str(value)


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])
    partial.replace(path)
    return path


def _groups(
    images: Sequence[ImageStatistics], conditions: BatchConditions
) -> list[tuple[str, list[ImageStatistics]]]:
    groups = [
        (condition.label, [image for image in images if image.condition == condition.label])
        for condition in conditions.conditions
    ]
    groups.append((ALL_IMAGES, list(images)))
    return groups


def _values(group: Sequence[ImageStatistics], key: MetricKey) -> list[float]:
    return [image.values[key] for image in group if key in image.values]


def write_batch_statistics(
    output_root: Path | str,
    stems: Sequence[str],
    conditions: BatchConditions | None = None,
    *,
    alpha: float = 0.05,
) -> BatchStatisticsFiles:
    """Read *stems*' statistics under *output_root* and write the by-image,
    summary and (with two conditions) comparison CSVs there. Writes nothing,
    and says why in ``message``, when no image has statistics."""
    output_root = Path(output_root)
    conditions = conditions or BatchConditions()
    images, missing = collect_batch_statistics(output_root, stems, conditions)
    files = BatchStatisticsFiles(images=len(images), missing=tuple(missing))
    if not images:
        files.message = (
            "No image of the batch has a statistics CSV (is Statistics on?): "
            "no batch statistics were written."
        )
        return files
    order = {condition.label: rank for rank, condition in enumerate(conditions.conditions)}
    images = sorted(
        images, key=lambda image: order.get(image.condition, len(order))
    )  # stable: each condition's images together, in batch order
    keys = metric_keys(images)
    files.metrics = len(keys)

    files.by_image = _write(
        output_root / BY_IMAGE_FILENAME,
        ["Section", "Metric", "Unit", *(image.stem for image in images)],
        [
            ["", "Condition", "", *(image.condition for image in images)],
            *(
                [*key, *(image.values.get(key) for image in images)]
                for key in keys
            ),
        ],
    )

    groups = _groups(images, conditions)
    files.summary = _write(
        output_root / SUMMARY_FILENAME,
        ["Section", "Metric", "Unit", "Group", *SUMMARY_FIELDS],
        (
            [*key, label, *summarise(_values(group, key)).values()]
            for key in keys
            for label, group in groups
            if _values(group, key)
        ),
    )

    if conditions.compares:
        (label_a, group_a), (label_b, group_b) = groups[0], groups[1]
        tests = [compare_two(_values(group_a, key), _values(group_b, key)) for key in keys]
        qvalues = benjamini_hochberg([test.p for test in tests])
        header = [
            "Section", "Metric", "Unit",
            f"N {label_a}", f"N {label_b}",
            f"Median {label_a}", f"Median {label_b}",
            f"Mean {label_a}", f"Mean {label_b}",
            f"Min {label_a}", f"Max {label_a}", f"Min {label_b}", f"Max {label_b}",
            f"Difference of medians ({label_b} - {label_a})",
            f"Ratio of medians ({label_b} / {label_a})",
            "Mann-Whitney U", "P (Mann-Whitney U, two-sided)",
            "Q (Benjamini-Hochberg)", "Smallest possible P",
            f"Rank-biserial r (+: {label_b} larger)",
            "Welch t", "P (Welch t)", f"Hedges g (+: {label_b} larger)",
            "Note",
        ]
        rows = []
        for key, test, q in zip(keys, tests, qvalues):
            a, b = summarise(_values(group_a, key)), summarise(_values(group_b, key))
            both = a["N"] and b["N"]
            ratio = b["Median"] / a["Median"] if both and a["Median"] != 0 else float("nan")
            rows.append(
                [
                    *key,
                    a["N"], b["N"], a["Median"], b["Median"], a["Mean"], b["Mean"],
                    a["Min"], a["Max"], b["Min"], b["Max"],
                    b["Median"] - a["Median"] if both else float("nan"),
                    ratio,
                    test.u, test.p, q, test.smallest_p, test.rank_biserial,
                    test.welch_t, test.welch_p, test.hedges_g, test.note,
                ]
            )
        files.comparison = _write(output_root / COMPARISON_FILENAME, header, rows)
        files.tested = sum(test.tested for test in tests)
        files.significant = sum(not math.isnan(q) and q < alpha for q in qvalues)

    parts = [f"Batch statistics of {len(images)} image(s), {len(keys)} metrics: {files.summary}"]
    if files.comparison is not None:
        parts.append(
            f"{conditions.b.label} against {conditions.a.label}: {files.tested} metrics tested, "
            f"{files.significant} with q < {alpha:g}: {files.comparison}"
        )
    elif len(conditions.conditions) == 1:
        parts.append("one condition set: give a second to test the two against each other")
    if missing:
        parts.append(f"no statistics for {', '.join(missing)}")
    files.message = "; ".join(parts) + "."
    return files


__all__ = [
    "ALL_IMAGES",
    "BOTH_CONDITIONS",
    "BY_IMAGE_FILENAME",
    "BatchConditions",
    "BatchStatisticsFiles",
    "COMPARISON_FILENAME",
    "Condition",
    "ImageStatistics",
    "NO_CONDITION",
    "SUMMARY_FIELDS",
    "SUMMARY_FILENAME",
    "TwoGroupTest",
    "benjamini_hochberg",
    "collect_batch_statistics",
    "compare_two",
    "image_statistics_files",
    "metric_keys",
    "parse_condition",
    "read_branch_statistics_csv",
    "read_statistics_csv",
    "smallest_mann_whitney_p",
    "summarise",
    "write_batch_statistics",
]
