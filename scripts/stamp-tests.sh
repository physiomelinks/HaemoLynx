#!/usr/bin/env bash
# Run the full suite and, if it passes, stamp the staged tree so the pre-commit
# hook can skip the fast tests while the stamp still matches.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=venv/bin/python
[ -x "$PY" ] || PY=python3
. scripts/git-hooks/_stamp.sh
MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true "$PY" -m pytest -q
stamp_write
if stamp_matches; then echo "stamped $(git write-tree)"; else
    echo "tests passed but no stamp written: tracked files have unstaged edits" >&2
fi
