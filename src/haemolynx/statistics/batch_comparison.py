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

The comparison is ``independent`` (above) or ``paired``: images of one
subject in the two conditions (say one animal's contra and ipsi sides) are not
independent, so each subject's two images are paired and the differences
tested -- the Wilcoxon signed-rank test with Benjamini-Hochberg q-values, the
smallest p it could reach at that many pairs (0.5 at 2 pairs: no difference
can reach 0.05 below 6), the matched-pairs rank-biserial r, and the paired t
test with Cohen's dz beside it. A subject ID is read from each name
(:meth:`BatchConditions.subject_of`); ``batch_condition_pairs.csv`` lists every
pair and every image left out with its reason (no partner, more than one image
of its subject in its condition, no subject ID, in neither condition). The
summary still covers every image of each condition, paired or not.

Only numbers are read: a metric named only by digits (the per-node rows of a
full-mode run), the Metadata section, and every text, list or N/A value are
left out. Pure: CSVs in, CSVs out.
"""
from __future__ import annotations

import csv
import logging
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

BY_IMAGE_FILENAME = "batch_statistics_by_image.csv"
SUMMARY_FILENAME = "batch_statistics_summary.csv"
COMPARISON_FILENAME = "batch_condition_comparison.csv"
PAIRS_FILENAME = "batch_condition_pairs.csv"

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


#: The comparison's two designs: images independent of each other, or paired
#: across the conditions by a subject ID read from their names.
INDEPENDENT = "independent"
PAIRED = "paired"
DESIGNS = (INDEPENDENT, PAIRED)

#: What a paired comparison does with a subject that has more than one image
#: in a condition: leave them out (which one pairs would be a guess), or pair
#: the median or mean of that subject's images in each condition.
AGGREGATE_NONE = "none"
AGGREGATES = (AGGREGATE_NONE, "median", "mean")

#: The separators trimmed off a subject ID read from a name.
_SUBJECT_SEPARATORS = " _-."


@dataclass(frozen=True)
class Pair:
    """One subject's images in each condition: one each, or, aggregated,
    every image of the subject in that condition."""

    subject: str
    a: tuple[str, ...]
    b: tuple[str, ...]

    @property
    def images(self) -> tuple[str, ...]:
        return self.a + self.b


@dataclass(frozen=True)
class Pairing:
    """The images paired across the two conditions, and every image left out
    with its reason: (name, subject or "", reason)."""

    pairs: tuple[Pair, ...] = ()
    unpaired: tuple[tuple[str, str, str], ...] = ()

    def pair_of(self, name: str) -> Pair | None:
        """The pair *name* is one of the images of, or None."""
        for pair in self.pairs:
            if name in pair.images:
                return pair
        return None

    def subject_of(self, name: str) -> str:
        for pair in self.pairs:
            if name in pair.images:
                return pair.subject
        for unpaired, subject, _reason in self.unpaired:
            if unpaired == name:
                return subject
        return ""

    def reason_left_out(self, name: str) -> str:
        for unpaired, _subject, reason in self.unpaired:
            if unpaired == name:
                return reason
        return ""


@dataclass(frozen=True)
class BatchConditions:
    """Up to two conditions, each read from the images' file names, and
    whether the two are compared as independent images or paired by subject.

    Paired, each image is paired with the image of the other condition whose
    name is most like its own once the condition phrase is taken out of both.
    An image's subject ID is the part of its file name *before* the condition
    phrase it holds, separators (space, ``_``, ``-``, ``.``) trimmed and case
    ignored (``28_3_d_contra_FITC`` is subject ``28_3_d`` of ``contra``), and
    only images of one subject pair. Within a subject, one image a side pair;
    with more, the names are matched one to one, closest first: the most words
    in common from the start of the name (the phrase taken out), then the most
    words in common anywhere (``E14_WT_1`` with ``E14_KO_1``, ``E14_WT_2`` with
    ``E14_KO_2``); a tie is left out, not guessed. A *subject_pattern* (a
    regular expression, searched ignoring case; its first group if it has
    one, else the whole match) replaces the subject rule, for names whose
    subject is not what comes first
    (``^(\\d+_\\d+)`` reads the animal, ``28_3``, from ``28_3_d_contra_FITC``).

    Under *aggregate* ``"median"`` or ``"mean"`` a subject's images are not
    matched one to one: each of its conditions is the median or mean of all
    its images there, metric by metric, and those are paired."""

    a: Condition | None = None
    b: Condition | None = None
    paired: bool = False
    subject_pattern: str = ""
    aggregate: str = AGGREGATE_NONE

    @classmethod
    def from_text(
        cls,
        a: str | None = None,
        b: str | None = None,
        *,
        paired: bool = False,
        subject_pattern: str | None = None,
        aggregate: str | None = None,
    ) -> "BatchConditions":
        return cls(
            parse_condition(a),
            parse_condition(b),
            paired=bool(paired),
            subject_pattern=(subject_pattern or "").strip(),
            aggregate=(aggregate or AGGREGATE_NONE).strip().lower(),
        )

    @property
    def design(self) -> str:
        return PAIRED if self.paired else INDEPENDENT

    def _phrase_span(self, name: str) -> tuple[int, int] | None:
        """Where *name*'s condition phrase is (the earliest of its condition's
        phrases), as (start, end) in the lowered name, or None."""
        label = self.condition_of(name)
        condition = next((c for c in self.conditions if c.label == label), None)
        if condition is None:
            return None
        lowered = name.lower()
        start, phrase = min(
            (lowered.find(phrase), phrase) for phrase in condition.phrases if phrase in lowered
        )
        return start, start + len(phrase)

    def subject_of(self, name: str) -> str:
        """*name*'s subject ID (lower case): what comes before its condition
        phrase, or what *subject_pattern* reads; "" when there is none."""
        if self.subject_pattern:
            found = re.search(self.subject_pattern, name, re.IGNORECASE)
            if found is None:
                return ""
            subject = found.group(1) if found.re.groups else found.group(0)
            return (subject or "").strip(_SUBJECT_SEPARATORS).lower()
        span = self._phrase_span(name)
        if span is None:
            return ""
        return name.lower()[: span[0]].strip(_SUBJECT_SEPARATORS)

    def name_words(self, name: str) -> tuple[str, ...]:
        """*name*'s words (lower case) with its condition phrase taken out:
        what two images' names are compared on."""
        lowered = name.lower()
        span = self._phrase_span(name)
        if span is not None:
            lowered = f"{lowered[: span[0]]} {lowered[span[1]:]}"
        return tuple(re.findall(r"[a-z0-9]+", lowered))

    def _likeness(self, a: str, b: str) -> tuple[int, int]:
        """How alike two names are: words in common from the start, then
        words in common anywhere (phrases taken out)."""
        words_a, words_b = self.name_words(a), self.name_words(b)
        leading = 0
        for word_a, word_b in zip(words_a, words_b):
            if word_a != word_b:
                break
            leading += 1
        shared = sum(min(words_a.count(word), words_b.count(word)) for word in set(words_a))
        return leading, shared

    def _shared_label(self, subject: str, a: str, b: str) -> str:
        """A pair's label within a subject of several pairs: the words its two
        names share from the start."""
        words_a, words_b = self.name_words(a), self.name_words(b)
        common = []
        for word_a, word_b in zip(words_a, words_b):
            if word_a != word_b:
                break
            common.append(word_a)
        return "_".join(common) or subject

    def _match_by_name(
        self, subject: str, a_names: list[str], b_names: list[str]
    ) -> tuple[list[Pair], list[tuple[str, str, str]]]:
        """One subject's images matched one to one, closest names first; a tie
        at the closest is left out, never guessed."""
        where = f"subject {subject}" if subject else "its name"
        pairs: list[Pair] = []
        unpaired: list[tuple[str, str, str]] = []
        a_left, b_left = list(a_names), list(b_names)
        while a_left and b_left:
            scores = {(a, b): self._likeness(a, b) for a in a_left for b in b_left}
            best = max(scores.values())
            closest = [pair for pair, score in scores.items() if score == best]
            unique = [
                (a, b) for a, b in closest
                if sum(a == other_a for other_a, _ in closest) == 1
                and sum(b == other_b for _, other_b in closest) == 1
            ]
            if unique:
                for a, b in unique:
                    pairs.append(Pair(subject, (a,), (b,)))
                    a_left.remove(a)
                    b_left.remove(b)
                continue
            tied = {name for pair in closest for name in pair}
            for name in sorted(tied, key=(a_names + b_names).index):
                unpaired.append(
                    (name, subject, f"names in {where} equally like each other: which pairs is unclear")
                )
            a_left = [name for name in a_left if name not in tied]
            b_left = [name for name in b_left if name not in tied]
        for own, other_label, other_names in (
            (a_left, self.b.label, b_names),
            (b_left, self.a.label, a_names),
        ):
            for name in own:
                reason = (
                    f"no {other_label} image of {where} left once the closest names were paired"
                    if other_names
                    else f"no {other_label} image of {where}"
                )
                unpaired.append((name, subject, reason))
        if len(pairs) > 1 or not subject:
            pairs = [
                Pair(self._shared_label(subject, pair.a[0], pair.b[0]), pair.a, pair.b)
                for pair in pairs
            ]
        return pairs, unpaired

    def pairing(self, names: Iterable[str]) -> Pairing:
        """*names* paired across the two conditions (see the class). An image
        is left out, with its reason, when it is in neither condition or both,
        the subject pattern reads no subject from it, its subject has no image
        of the other condition, or -- matched one to one -- its closest name
        is tied or taken."""
        if not self.compares:
            return Pairing()
        unpaired: list[tuple[str, str, str]] = []
        by_subject: dict[str, dict[str, list[str]]] = {}
        names = list(names)
        for name in names:
            label = self.condition_of(name)
            if label == BOTH_CONDITIONS:
                unpaired.append((name, "", "in both conditions"))
                continue
            if label == NO_CONDITION:
                unpaired.append((name, "", "in neither condition"))
                continue
            subject = self.subject_of(name)
            if not subject and self.subject_pattern:
                unpaired.append((name, "", "no subject ID: the subject pattern finds none in the name"))
                continue
            by_subject.setdefault(subject, {self.a.label: [], self.b.label: []})[label].append(name)
        pairs: list[Pair] = []
        for subject, sides in by_subject.items():
            a_names, b_names = sides[self.a.label], sides[self.b.label]
            if self.aggregate != AGGREGATE_NONE and a_names and b_names:
                pairs.append(Pair(subject, tuple(a_names), tuple(b_names)))
                continue
            matched, left_out = self._match_by_name(subject, a_names, b_names)
            pairs.extend(matched)
            unpaired.extend(left_out)
        order = {name: index for index, name in enumerate(names)}
        pairs.sort(key=lambda pair: order[pair.a[0]])
        unpaired.sort(key=lambda item: order[item[0]])
        return Pairing(tuple(pairs), tuple(unpaired))

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
        if self.paired and not self.compares:
            return "A paired comparison needs both conditions."
        if self.aggregate not in AGGREGATES:
            return (
                f"'{self.aggregate}' is no way to combine a subject's images: "
                f"choose one of {', '.join(AGGREGATES)}."
            )
        if self.subject_pattern:
            try:
                re.compile(self.subject_pattern)
            except re.error as error:
                return f"The subject ID pattern is not a regular expression: {error}."
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
        names = list(names)
        counts = self.counts(names)
        parts = [f"{counts[condition.label]} in {condition.label}" for condition in self.conditions]
        if counts[BOTH_CONDITIONS]:
            parts.append(f"{counts[BOTH_CONDITIONS]} in both")
        if counts[NO_CONDITION]:
            parts.append(f"{counts[NO_CONDITION]} in neither")
        if self.paired and self.compares and not self.problem():
            pairing = self.pairing(names)
            subjects = ", ".join(pair.subject for pair in pairing.pairs)
            parts.append(
                f"{len(pairing.pairs)} pair(s) by subject"
                + (f" ({subjects})" if subjects else "")
                + (f", {len(pairing.unpaired)} image(s) with no partner" if pairing.unpaired else "")
            )
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


#: The largest field the statistics CSVs are read with: a full-mode run writes
#: whole lists (its communities, a node set each) into one Value field, far
#: past csv's default 131,072 characters. Held to a C long, which is 32 bits
#: on Windows, where csv.field_size_limit(sys.maxsize) overflows.
WIDE_FIELD_LIMIT = 2**31 - 1


@contextmanager
def _wide_csv_fields():
    """csv's field limit raised to :data:`WIDE_FIELD_LIMIT` for one read and
    put back after it (the limit is the interpreter's, not the reader's)."""
    previous = csv.field_size_limit(WIDE_FIELD_LIMIT)
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def read_statistics_csv(path: Path | str) -> dict[MetricKey, float]:
    """The numbers in one ``{stem}_statistics.csv``, by (section, metric, unit)."""
    values: dict[MetricKey, float] = {}
    with _wide_csv_fields(), Path(path).open(newline="", encoding="utf-8") as handle:
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
    with _wide_csv_fields(), Path(path).open(newline="", encoding="utf-8") as handle:
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


def smallest_wilcoxon_p(n_pairs: int) -> float:
    """The smallest two-sided p an exact Wilcoxon signed-rank test can give
    with *n_pairs* non-zero differences and no ties: 2 / 2**n (0.5 at 2
    pairs, 0.25 at 3, 0.0625 at 5, 0.031 at 6)."""
    if n_pairs < 1:
        return float("nan")
    return min(1.0, 2.0 / 2.0**n_pairs)


@dataclass(frozen=True)
class PairedTest:
    """Condition B against condition A on one metric, over subjects with an
    image in each; differences are B - A."""

    n_pairs: int = 0
    w_plus: float = float("nan")
    p: float = float("nan")
    smallest_p: float = float("nan")
    rank_biserial: float = float("nan")
    t: float = float("nan")
    t_p: float = float("nan")
    cohen_dz: float = float("nan")
    note: str = ""

    @property
    def tested(self) -> bool:
        return not math.isnan(self.p)


def compare_paired(a: Sequence[float], b: Sequence[float]) -> PairedTest:
    """The Wilcoxon signed-rank test (two-sided, zero differences dropped;
    exact for small samples as scipy chooses) and the paired t test of *b*
    against *a*, ``a[i]`` and ``b[i]`` one subject's two images. W+ is the
    rank sum of the pairs where *b* is larger; the matched-pairs rank-biserial
    r, (W+ - W-) / (W+ + W-), and Cohen's dz, mean / SD of the differences,
    are positive when *b* tends to be larger."""
    import warnings

    from scipy import stats as sp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(a.size, b.size))
    if n < MIN_IMAGES_PER_CONDITION:
        return PairedTest(
            n_pairs=n,
            note=f"not tested: fewer than {MIN_IMAGES_PER_CONDITION} pairs",
        )
    differences = b - a
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return PairedTest(
            n_pairs=n, note="not tested: the same value in both images of every pair"
        )
    ranks = sp.rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    smallest = smallest_wilcoxon_p(int(nonzero.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(sp.wilcoxon(nonzero, alternative="two-sided").pvalue)
    notes = []
    if nonzero.size < n:
        notes.append(f"{n - nonzero.size} pair(s) with no difference left out of the signed-rank test")
    if smallest > 0.05:
        notes.append(
            f"with {nonzero.size} pair(s) no difference can reach p < 0.05 "
            f"(smallest possible p {smallest:.3g})"
        )
    t = t_p = dz = float("nan")
    sd = float(differences.std(ddof=1))
    if sd > 0:
        paired_t = sp.ttest_rel(b, a)
        t, t_p = float(paired_t.statistic), float(paired_t.pvalue)
        dz = float(differences.mean() / sd)
    return PairedTest(
        n_pairs=n,
        w_plus=w_plus,
        p=p,
        smallest_p=smallest,
        rank_biserial=(w_plus - w_minus) / (w_plus + w_minus),
        t=t,
        t_p=t_p,
        cohen_dz=dz,
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
    #: Paired designs only: every pair, and every image left out and why.
    pairs: Path | None = None
    pairing: Pairing | None = None
    images: int = 0
    missing: tuple[str, ...] = ()
    metrics: int = 0
    tested: int = 0
    significant: int = 0
    message: str = ""

    @property
    def written(self) -> tuple[Path, ...]:
        return tuple(
            path for path in (self.by_image, self.summary, self.comparison, self.pairs) if path
        )


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


def _write_paired(
    files: BatchStatisticsFiles,
    output_root: Path,
    images: Sequence[ImageStatistics],
    keys: Sequence[MetricKey],
    conditions: BatchConditions,
    pairing: Pairing,
    alpha: float,
) -> None:
    """The paired comparison: ``batch_condition_pairs.csv`` (every pair with
    the images that make each condition's value, and every image left out
    with its reason) and the comparison CSV over the subjects with an image
    in each condition -- a subject's value in a condition the median or mean
    of its images there when the pairing aggregates, metric by metric over
    the images that have the metric."""
    label_a, label_b = conditions.a.label, conditions.b.label
    by_stem = {image.stem: image for image in images}

    def value_rule(names: Sequence[str]) -> str:
        return "the one image" if len(names) == 1 else f"{conditions.aggregate} of {len(names)} images"

    files.pairs = _write(
        output_root / PAIRS_FILENAME,
        [
            "Subject",
            f"{label_a} images", f"{label_b} images",
            f"{label_a} value", f"{label_b} value",
            "Left out because",
        ],
        [
            *(
                [
                    pair.subject, "; ".join(pair.a), "; ".join(pair.b),
                    value_rule(pair.a), value_rule(pair.b), "",
                ]
                for pair in pairing.pairs
            ),
            *(
                [
                    subject,
                    name if by_stem[name].condition == label_a else "",
                    name if by_stem[name].condition == label_b else "",
                    "", "",
                    reason,
                ]
                if by_stem[name].condition in (label_a, label_b)
                else [subject, "", "", "", "", f"{name}: {reason}"]
                for name, subject, reason in pairing.unpaired
            ),
        ],
    )
    files.pairing = pairing
    combine = np.median if conditions.aggregate == "median" else np.mean

    def subject_value(names: Sequence[str], key: MetricKey) -> float | None:
        values = [by_stem[name].values[key] for name in names if key in by_stem[name].values]
        if not values:
            return None
        return values[0] if len(values) == 1 else float(combine(values))

    tests: list[PairedTest] = []
    rows = []
    for key in keys:
        matched = []
        for pair in pairing.pairs:
            a_value, b_value = subject_value(pair.a, key), subject_value(pair.b, key)
            if a_value is not None and b_value is not None:
                matched.append((a_value, b_value))
        a_values = [a for a, _b in matched]
        b_values = [b for _a, b in matched]
        test = compare_paired(a_values, b_values)
        tests.append(test)
        a, b = summarise(a_values), summarise(b_values)
        differences = summarise([b - a for a, b in matched])
        rows.append(
            [
                *key,
                len(matched), a["Median"], b["Median"], a["Mean"], b["Mean"],
                a["Min"], a["Max"], b["Min"], b["Max"],
                differences["Median"], differences["Mean"],
                test.w_plus, test.p, None, test.smallest_p, test.rank_biserial,
                test.t, test.t_p, test.cohen_dz, test.note,
            ]
        )
    header = [
        "Section", "Metric", "Unit",
        "N pairs",
        f"Median {label_a}", f"Median {label_b}",
        f"Mean {label_a}", f"Mean {label_b}",
        f"Min {label_a}", f"Max {label_a}", f"Min {label_b}", f"Max {label_b}",
        f"Median of paired differences ({label_b} - {label_a})",
        f"Mean of paired differences ({label_b} - {label_a})",
        f"Wilcoxon W+ (pairs with {label_b} larger)",
        "P (Wilcoxon signed-rank, two-sided)",
        "Q (Benjamini-Hochberg)", "Smallest possible P",
        f"Matched-pairs rank-biserial r (+: {label_b} larger)",
        "Paired t", "P (paired t)", f"Cohen dz (+: {label_b} larger)",
        "Note",
    ]
    qvalues = benjamini_hochberg([test.p for test in tests])
    q_column = header.index("Q (Benjamini-Hochberg)")
    for row, q in zip(rows, qvalues):
        row[q_column] = q
    files.comparison = _write(output_root / COMPARISON_FILENAME, header, rows)
    files.tested = sum(test.tested for test in tests)
    files.significant = sum(not math.isnan(q) and q < alpha for q in qvalues)


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
    pairing = (
        conditions.pairing([image.stem for image in images])
        if conditions.paired and conditions.compares
        else Pairing()
    )
    subject_rows = (
        [["", "Subject", "", *(pairing.subject_of(image.stem) for image in images)]]
        if conditions.paired and conditions.compares
        else []
    )

    files.by_image = _write(
        output_root / BY_IMAGE_FILENAME,
        ["Section", "Metric", "Unit", *(image.stem for image in images)],
        [
            ["", "Condition", "", *(image.condition for image in images)],
            *subject_rows,
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

    if conditions.compares and not conditions.paired:
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
    elif conditions.compares and conditions.paired:
        _write_paired(files, output_root, images, keys, conditions, pairing, alpha)

    parts = [f"Batch statistics of {len(images)} image(s), {len(keys)} metrics: {files.summary}"]
    if files.comparison is not None and conditions.paired:
        left_out = ", ".join(name for name, _subject, _reason in pairing.unpaired)
        parts.append(
            f"{conditions.b.label} against {conditions.a.label}, paired by subject"
            + (
                f" ({conditions.aggregate} of each subject's images in a condition)"
                if conditions.aggregate != AGGREGATE_NONE
                else ""
            )
            + f": {len(pairing.pairs)} pair(s)"
            + (f" ({', '.join(pair.subject for pair in pairing.pairs)})" if pairing.pairs else "")
            + f", smallest possible p {smallest_wilcoxon_p(len(pairing.pairs)):.3g}"
            + (f"; left out with no partner: {left_out}" if left_out else "")
            + f"; {files.tested} metrics tested, {files.significant} with q < {alpha:g}: "
            f"{files.comparison} (pairs: {files.pairs})"
        )
    elif files.comparison is not None:
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
    "AGGREGATES",
    "AGGREGATE_NONE",
    "ALL_IMAGES",
    "BOTH_CONDITIONS",
    "BY_IMAGE_FILENAME",
    "BatchConditions",
    "BatchStatisticsFiles",
    "COMPARISON_FILENAME",
    "Condition",
    "DESIGNS",
    "INDEPENDENT",
    "ImageStatistics",
    "NO_CONDITION",
    "PAIRED",
    "PAIRS_FILENAME",
    "Pair",
    "PairedTest",
    "Pairing",
    "SUMMARY_FIELDS",
    "SUMMARY_FILENAME",
    "TwoGroupTest",
    "WIDE_FIELD_LIMIT",
    "benjamini_hochberg",
    "collect_batch_statistics",
    "compare_paired",
    "compare_two",
    "image_statistics_files",
    "metric_keys",
    "parse_condition",
    "read_branch_statistics_csv",
    "read_statistics_csv",
    "smallest_mann_whitney_p",
    "smallest_wilcoxon_p",
    "summarise",
    "write_batch_statistics",
]
