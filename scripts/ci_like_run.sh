#!/usr/bin/env bash
# Run the suite the way CI does (.github/workflows/pytest-pr.yml): Python 3.10,
# a fresh venv, `pip install -e .[dev]` (so only pyproject.toml's dependencies),
# MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true. Extra arguments go to pytest.
#
# Everything (venv, pip cache, a copy of the tracked files) lives in one temp
# folder that is removed on exit. Nothing is written elsewhere and no
# interpreter or package is installed globally: HOME, the XDG folders and
# matplotlib's config folder point inside it too, so no cache lands in ~.
# As in CI, that home folder is empty, so the specimen data root is not found
# and the realdata tests skip. Set CI_LIKE_TMPDIR to choose the parent folder
# of the temp folder (default: $TMPDIR, else /tmp).
set -euo pipefail
# Drop every GIT_* variable first. A hook (or any git-run parent) leaves GIT_DIR, GIT_INDEX_FILE
# and the like set; left in, they aim every git call below at that repository, and the
# `git init` in the copy then writes `core.bare = true` into its shared config.
while IFS= read -r v; do unset "$v"; done < <(compgen -e | grep '^GIT_' || true)
cd "$(git rev-parse --show-toplevel)"

PY310=$(command -v python3.10 || true)
if [ -z "$PY310" ]; then
    cat >&2 <<'MSG'
python3.10 not found on PATH. This script does not install it.
Get it yourself, for example:
  uv python install 3.10      (puts an interpreter in ~/.local/bin)
  or your package manager (e.g. apt install python3.10-venv)
Then rerun this script.
MSG
    exit 2
fi

parent=${CI_LIKE_TMPDIR:-${TMPDIR:-/tmp}}
work=$(mktemp -d "$parent/ci_like_run.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Install from a copy of the tracked files so `pip install -e` leaves no
# egg-info in the repo. Files with unstaged deletions are skipped.
mkdir "$work/src_copy"
git ls-files -z | while IFS= read -r -d '' f; do
    [ -e "$f" ] && printf '%s\0' "$f"
done | tar --null -T - -cf - | tar -xf - -C "$work/src_copy"

"$PY310" -m venv "$work/venv"
export TMPDIR="$work"  # pip and pytest scratch files stay inside too
export PIP_CACHE_DIR="$work/pip-cache" PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true PYTHONUNBUFFERED=1
export HOME="$work/home" MPLCONFIGDIR="$work/mpl" XDG_CACHE_HOME="$work/cache" \
    XDG_CONFIG_HOME="$work/config" XDG_DATA_HOME="$work/data" XDG_STATE_HOME="$work/state"
mkdir "$HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"
cd "$work/src_copy"
# CI runs in a git checkout, and some tests ask git which files are tracked
# (tests/test_example_main_blocks.py collects with `git ls-files`), so the copy
# is made a repository of the same files. -f: a tracked file may match .gitignore.
git init -q
git add -A -f
git -c user.name=ci_like_run -c user.email=ci_like_run@localhost -c commit.gpgsign=false \
    commit -q -m "tracked files, copied"
"$work/venv/bin/python" -m pip install --upgrade pip
"$work/venv/bin/python" -m pip install -e '.[dev]'
"$work/venv/bin/python" -m pytest "$@"
