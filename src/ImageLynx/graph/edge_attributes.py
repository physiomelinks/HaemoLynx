"""Reading a number off a graph edge without inventing one.

The graph-side counterpart of ``BatchRun.numeric_column``: an edge attribute becomes a float
only when it is present and finite. A missing, None, NaN, infinite or text value raises, naming
the edge, rather than being written into a table or a VTK file as NaN or 0.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def edge_float(data: Mapping[str, Any], attr: str, edge: tuple) -> float:
    """``data[attr]`` as a finite float, or raise naming ``edge`` and ``attr``.

    ``data`` is one edge's attribute dict and ``edge`` its ``(u, v, key)``, used only in the
    message. Zero is a value: only absence, non-numbers (booleans, and anything ``float``
    refuses, such as "wide") and non-finite values are refused.
    """
    value = data.get(attr)
    if value is None:
        raise ValueError(f"edge {edge} has no {attr!r}.")
    if isinstance(value, (bool, np.bool_)):
        number = None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
    if number is None:
        raise ValueError(f"edge {edge} has a {attr!r} that is not a number: {value!r}.")
    if not math.isfinite(number):
        raise ValueError(f"edge {edge} has a non-finite {attr!r}: {value!r}.")
    return number
