"""Compare a pipeline run on this checkout against a run on a reference ref.

The pieces are split so the parts worth testing do not need a pipeline run:

``metrics``       artefacts -> comparable numbers (pure, unit-tested)
``report``        numbers -> COMPARISON.md and index.html (pure, unit-tested)
``run_settings``  the settings both sides run with
``runner``        git worktrees, subprocesses, artefact collection
``_run_side``     the standalone script that runs inside one checkout

The command-line entry point is ``scripts/compare_branches.py``.
"""

__all__ = ["metrics", "report", "run_settings", "runner"]
