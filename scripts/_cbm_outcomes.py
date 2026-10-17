"""pytest plugin loaded by coverage_by_module.py ``run``: records each test's outcome.

Writes {"outcomes": {node id: "passed" | "skipped" | "failed"}, "realdata": [node ids marked
``realdata``]} as JSON to the path in CBM_OUTCOMES_FILE.
"""
import json
import os

_outcomes: dict[str, str] = {}
_realdata: set[str] = set()


def pytest_runtest_logreport(report):
    if "realdata" in report.keywords:
        _realdata.add(report.nodeid)
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcomes[report.nodeid] = report.outcome
    elif report.when == "teardown" and report.outcome == "failed":
        _outcomes[report.nodeid] = "failed"


def pytest_sessionfinish(session):
    with open(os.environ["CBM_OUTCOMES_FILE"], "w") as f:
        json.dump({"outcomes": _outcomes, "realdata": sorted(_realdata)}, f)
