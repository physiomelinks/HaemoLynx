"""A pytest plugin that writes every collected test's ID and markers as JSON.

``pytest --collect-only -p release_review._collect_plugin``, with the output
path in ``RELEASE_REVIEW_COLLECTED``: one collection gives the counts per
marker that would otherwise take one ``-m`` collection each.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_items: list[dict] = []
_errors: list[str] = []


def pytest_collectreport(report):
    if report.failed:
        _errors.append(report.nodeid)


def pytest_collection_modifyitems(session, config, items):
    _items[:] = [
        {"id": item.nodeid, "markers": sorted({marker.name for marker in item.iter_markers()})}
        for item in items
    ]


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("RELEASE_REVIEW_COLLECTED")
    if path:
        Path(path).write_text(json.dumps({"items": _items, "errors": _errors}), encoding="utf-8")
