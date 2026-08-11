"""Does a per-specimen quantity separate by group when it should not?

Some quantities in this study are supposed to be properties of the instrument rather than of
the tissue: the segmentation threshold, the foreground fraction at a frozen threshold, the
classifier's mean output probability. If one of those separates cleanly by cohort, part of
the measured group difference is the measuring device, and the biological reading is
contaminated in a way nothing downstream can undo.

Checking it by eye does not work. With n = 3 per group, complete separation happens by chance
with probability 2/C(6,3) = 0.10, so "all the WKY values are below all the SHR values" is
weak evidence on its own and a coin-flip's distance from unremarkable. It is also the exact
floor of a two-sided rank test at this n: no arrangement of three against three can reach
p < 0.10. Reporting a p-value without that context invites reading 0.10 as "nearly
significant" when it is in fact the most extreme result available.

So this reports the separation, the size of the gap relative to the spread, and the exact
permutation p - alongside the floor, so the two are never seen apart.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CohortSplit:
    """Whether a per-specimen quantity separates by group."""

    quantity: str
    values_by_group: Dict[str, Tuple[float, ...]]
    separated: bool
    direction: Optional[str]
    gap: float
    permutation_p: float
    floor_p: float
    concerning: bool
    verdict: str


def _exact_permutation_p(a, b) -> float:
    """Two-sided exact p for the difference in means, over every group assignment."""
    pooled = np.concatenate([a, b])
    n = len(a)
    observed = abs(np.mean(a) - np.mean(b))
    indices = range(len(pooled))
    diffs = []
    for chosen in combinations(indices, n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        diffs.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
    diffs = np.asarray(diffs)
    return float((diffs >= observed - 1e-12).sum() / len(diffs))


def assess_cohort_split(
    values_by_specimen: Dict[str, float],
    quantity: str = "value",
    groups_by_specimen: Optional[Dict[str, str]] = None,
) -> CohortSplit:
    """Report whether ``values_by_specimen`` separates by cohort.

    ``concerning`` is set when the groups separate completely *and* the gap between them
    exceeds the within-group spread. Separation alone is too weak a signal at n = 3 to act
    on; separation with a gap wider than the noise is worth stopping for.
    """
    if groups_by_specimen is None:
        from ..specimens import get_specimen

        groups_by_specimen = {
            sid: get_specimen(sid).group for sid in values_by_specimen
        }

    by_group: Dict[str, list] = {}
    for specimen_id, value in values_by_specimen.items():
        by_group.setdefault(groups_by_specimen[specimen_id], []).append(float(value))

    if len(by_group) != 2 or any(len(v) < 2 for v in by_group.values()):
        return CohortSplit(quantity, {g: tuple(v) for g, v in by_group.items()},
                           False, None, 0.0, 1.0, 1.0, False,
                           "Not enough specimens per group to assess separation.")

    (group_a, values_a), (group_b, values_b) = sorted(by_group.items())
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)

    separated = a.max() < b.min() or b.max() < a.min()
    direction = None
    gap = 0.0
    if separated:
        if a.max() < b.min():
            direction, gap = f"{group_a} < {group_b}", float(b.min() - a.max())
        else:
            direction, gap = f"{group_b} < {group_a}", float(a.min() - b.max())

    spread = float(max(a.max() - a.min(), b.max() - b.min()))
    permutation_p = _exact_permutation_p(a, b)
    floor_p = _exact_permutation_p(np.array([0.0, 1.0, 2.0][:len(a)]),
                                   np.array([10.0, 11.0, 12.0][:len(b)]))
    concerning = bool(separated and gap > spread)

    if not separated:
        verdict = (f"{quantity} does not separate by group; the cohorts overlap. "
                   f"This is the reassuring outcome.")
    elif concerning:
        verdict = (
            f"{quantity} separates completely ({direction}) with a gap of {gap:.4g}, wider "
            f"than the {spread:.4g} spread within either group. At n = 3 the exact p cannot "
            f"go below {floor_p:.2f}, so this is as extreme as the data can show. If this "
            f"quantity is a property of the instrument rather than the tissue, part of the "
            f"group difference is the instrument."
        )
    else:
        verdict = (
            f"{quantity} separates ({direction}) but only by {gap:.4g}, within the "
            f"{spread:.4g} spread inside the groups. Complete separation arises by chance "
            f"with probability {floor_p:.2f} at n = 3, so this is weak on its own."
        )

    return CohortSplit(quantity, {g: tuple(v) for g, v in by_group.items()},
                       separated, direction, gap, permutation_p, floor_p,
                       concerning, verdict)
