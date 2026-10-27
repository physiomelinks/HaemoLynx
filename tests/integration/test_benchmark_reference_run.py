"""The pipeline on the benchmark sub-volume still gives the committed reference run.

The pipeline runs once per session, through its command line (the update script's
``pipeline_command``: ``--placed-input`` on the sub-volume, frozen defaults, no ``--config``),
into a folder pytest keeps. Then one test per part asks ``reference_run`` to compare it with
``tests/data/benchmark/reference_run/``. A failure prints the per-part report and the run
folder's path; after a deliberate change, ``examples/update_benchmark_reference_run.py``
rewrites the reference run.
"""
import pytest

import update_benchmark_reference_run as update_script
from ImageLynx.reference_run import PARTS, compare_with_reference, format_report

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.benchmark]

REFERENCE = update_script.REFERENCE_RUN_DIR


@pytest.fixture(scope="session")
def benchmark_run(tmp_path_factory):
    """The run folder of one pipeline run on the sub-volume, kept after the session."""
    run_dir = tmp_path_factory.mktemp("benchmark") / "run"
    update_script.run_pipeline(run_dir)
    return run_dir


@pytest.fixture(scope="session")
def comparison(benchmark_run):
    """The comparison, or the error it raised (a blank cell, say), so each test can report it
    with the run folder rather than error in setup."""
    command = update_script.recorded_command(update_script.pipeline_command(benchmark_run),
                                             benchmark_run)
    try:
        return compare_with_reference(benchmark_run, REFERENCE, command)
    except Exception as error:  # reported by every test, with the run folder
        return error


@pytest.mark.parametrize("part", PARTS, ids=lambda part: part.replace(" ", "_"))
def test_the_part_matches_the_reference_run(part, comparison, benchmark_run):
    if isinstance(comparison, Exception):
        pytest.fail(f"Comparing with {REFERENCE} raised {comparison!r}\n"
                    f"Run folder kept: {benchmark_run}")
    assert comparison.verdict(part).same, (
        f"\n{format_report(comparison, label=f'Benchmark run against {REFERENCE}')}"
        f"\nRun folder kept: {benchmark_run}")
