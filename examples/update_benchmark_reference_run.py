"""Rewrite the reference run: rerun the pipeline on the benchmark sub-volume and store its parts.

Use it after a deliberate change moves a pipeline output. It refuses a tree with tracked
changes (the manifest records the commit), runs ``carotid_image_to_model.py --placed-input`` on
the sub-volume into a temporary folder with the frozen defaults (no ``--config``), prints how
the new run compares with the existing reference run (or says there is none), then rewrites
``tests/data/benchmark/reference_run/``. The commit that changes the reference run says why.

    venv/bin/python examples/update_benchmark_reference_run.py

The benchmark tests (``tests/integration/test_benchmark_reference_run.py``, ``-m benchmark``)
run the same command line, from ``pipeline_command``.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import make_benchmark_subvolume as subvolume  # noqa: E402
from ImageLynx.reference_run import (  # noqa: E402
    compare_with_reference, format_report, write_reference_run,
)

PIPELINE = REPO / "examples" / "carotid_image_to_model.py"
REFERENCE_RUN_DIR = subvolume.BENCHMARK_DIR / "reference_run"
RUN_ENV = {"MPLBACKEND": "Agg", "PYVISTA_OFF_SCREEN": "true"}
# Stand-ins for machine paths in the command the manifest records.
REPO_TOKEN, RUN_TOKEN = "<REPO>", "<RUN>"


def pipeline_command(out_dir) -> list[str]:
    """The pipeline on the sub-volume with the frozen defaults, writing into ``out_dir``."""
    return [sys.executable, str(PIPELINE), "--placed-input", str(subvolume.SUBVOLUME_PATH),
            "--specimen", subvolume.SPECIMEN_ID, "--output-dir", str(out_dir)]


def recorded_command(command, out_dir) -> list[str]:
    """``command`` as the manifest keeps it: no interpreter, no machine paths."""
    return [part.replace(str(out_dir), RUN_TOKEN).replace(str(REPO), REPO_TOKEN)
            for part in command[1:]]


def run_pipeline(out_dir):
    """Run ``pipeline_command`` from the repo root; a non-zero exit raises with the log's tail."""
    done = subprocess.run(pipeline_command(out_dir), cwd=REPO, env={**os.environ, **RUN_ENV},
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"The pipeline exited {done.returncode} on the benchmark sub-volume:\n"
                           f"{done.stderr[-3000:]}")
    return done


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout


def update(reference_dir=REFERENCE_RUN_DIR, git=_git, run=run_pipeline, out=print) -> dict:
    """Refuse a dirty tree, run, print the comparison, rewrite the reference run. Returns the
    new manifest."""
    dirty = subvolume.tracked_changes(git)
    if dirty:
        raise RuntimeError("Refusing to rewrite the reference run from a tree with tracked "
                           f"changes (its manifest records the commit):\n{dirty}")
    reference_dir = Path(reference_dir)
    with tempfile.TemporaryDirectory(prefix="benchmark_run_") as tmp:
        run_dir = Path(tmp) / "run"
        run(run_dir)
        if reference_dir.is_dir():
            out(format_report(compare_with_reference(run_dir, reference_dir),
                              label=f"New run against {reference_dir}"))
        else:
            out(f"No reference run at {reference_dir}; writing the first one.")
        return write_reference_run(run_dir, reference_dir,
                                   recorded_command(pipeline_command(run_dir), run_dir),
                                   git("rev-parse", "HEAD").strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    manifest = update()
    print(f"Wrote {REFERENCE_RUN_DIR} at commit {manifest['commit']}.")


if __name__ == "__main__":
    main()
