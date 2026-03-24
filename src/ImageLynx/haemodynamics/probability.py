"""Probability helpers for stochastic pericyte constriction."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def select_active_pericyte_indices(
    total_pericytes: int,
    constriction_probability: float,
    *,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Randomly select pericyte indices that are active for constriction.

    Parameters
    ----------
    total_pericytes:
        Number of pericytes available.
    constriction_probability:
        Activation probability in [0, 1]. Example: 0.8 means 80% expected active.
    rng:
        Optional random generator. If omitted, uses a fresh default RNG so each
        pipeline run naturally produces a different cohort.
    """
    if total_pericytes < 0:
        raise ValueError(f"total_pericytes must be >= 0, got {total_pericytes}.")
    if not (0.0 <= float(constriction_probability) <= 1.0):
        raise ValueError(
            "constriction_probability must be in [0, 1], "
            f"got {constriction_probability}."
        )
    if total_pericytes == 0:
        return []
    generator = rng if rng is not None else np.random.default_rng()
    active_mask = generator.random(total_pericytes) < float(constriction_probability)
    return np.flatnonzero(active_mask).astype(int).tolist()


def validate_active_pericyte_indices(
    active_pericyte_indices: Iterable[int] | None,
    *,
    total_pericytes: int,
) -> list[int]:
    """Validate and normalize a caller-supplied active cohort."""
    if active_pericyte_indices is None:
        return []
    out: list[int] = []
    for idx in active_pericyte_indices:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= int(total_pericytes):
            raise ValueError(
                f"Active pericyte index {idx_int} outside valid range "
                f"[0, {int(total_pericytes) - 1}]."
            )
        out.append(idx_int)
    return sorted(set(out))
