"""A reproducibility seed shared by every sampled/approximate statistic.

Split out on its own so `shape.py` (path efficiency's random pair sample)
and `network_measures.py` (approximate betweenness) both use the exact same
value without either module depending on the other just for one constant.
"""
from __future__ import annotations

#: Not individually meaningful -- it exists only so a "fast" (sampled)
#: statistics run reproduces the exact same numbers from one run to the
#: next, not because 42 (or any other value) is a better sample than
#: another.
REPRODUCIBILITY_SEED = 42
