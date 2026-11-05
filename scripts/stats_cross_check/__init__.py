"""Cross-check HaemoLynx's morphometric statistics against independent tools.

Every tool reads the same binary mask and reports total centreline length,
segment, branch-point and end-point counts, and per-segment length,
tortuosity and diameter:

``measures``         one record per tool, and the table (pure, unit-tested)
``haemolynx_side``   the run's own stages to the graph, then its statistics CSV
``skan_side``        skan on the Lee skeleton, nothing cleaned
``maskel_side``      maskel (MIT, Python >= 3.13) in an interpreter of its own
``maskel_runner``    the standalone script that runs inside that interpreter
``vesselvio_files``  VesselVio's own result files (GPL-3: run as a program, never imported)

The command-line entry point is ``scripts/stats_cross_check.py``.
"""

__all__ = ["haemolynx_side", "maskel_side", "measures", "skan_side", "vesselvio_files"]
