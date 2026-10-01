"""cb_h1_th_metrics.py output paths: --all must not overwrite the WKY-only table.

Re-run notes 2026-10-01, item 7 (package P). ``--out`` used to default to the WKY-only file
with or without ``--all``, so the six-specimen run silently replaced it.
"""
import json
from types import SimpleNamespace

import pytest

import cb_h1_th_metrics


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Run main() in tmp_path with the specimens and analysis stubbed out."""
    th_map = tmp_path / "th.h5"
    th_map.touch()
    specimens = [SimpleNamespace(specimen_id=sid, group=sid[:3], th_probabilities_path=th_map)
                 for sid in ("WKY-A", "SHR-A")]
    monkeypatch.setattr(cb_h1_th_metrics, "SPECIMENS", specimens)
    monkeypatch.setattr(cb_h1_th_metrics, "analyse",
                        lambda s, t: SimpleNamespace(as_dict=lambda: {"specimen": s.specimen_id}))
    monkeypatch.setattr(cb_h1_th_metrics, "_table", lambda rows: "")
    monkeypatch.setattr(cb_h1_th_metrics, "_group_summary", lambda rows, field: {})
    monkeypatch.chdir(tmp_path)

    def run(*flags):
        monkeypatch.setattr("sys.argv", ["cb_h1_th_metrics.py", "--th-threshold", "0.5", *flags])
        cb_h1_th_metrics.main()

    return run, tmp_path / cb_h1_th_metrics.WKY_OUT, tmp_path / cb_h1_th_metrics.ALL_OUT


def test_default_run_writes_the_wky_only_file(stubbed):
    run, wky_out, all_out = stubbed
    run()
    payload = json.loads(wky_out.read_text())
    assert payload["shr_included"] is False
    assert [r["specimen"] for r in payload["by_threshold"]["0.5"]] == ["WKY-A"]
    assert not all_out.exists()


def test_all_writes_its_own_file_and_leaves_the_wky_table_alone(stubbed):
    run, wky_out, all_out = stubbed
    wky_out.parent.mkdir(parents=True)
    wky_out.write_text("wky-only table")
    run("--all")
    assert wky_out.read_text() == "wky-only table"
    payload = json.loads(all_out.read_text())
    assert payload["shr_included"] is True
    assert [r["specimen"] for r in payload["by_threshold"]["0.5"]] == ["WKY-A", "SHR-A"]


def test_explicit_out_overrides_the_default(stubbed, tmp_path):
    run, wky_out, all_out = stubbed
    run("--all", "--out", "custom.json")
    assert json.loads((tmp_path / "custom.json").read_text())["shr_included"] is True
    assert not wky_out.exists()
    assert not all_out.exists()
