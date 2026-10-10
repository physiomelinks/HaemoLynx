"""``ImageLynx.driver_snapshots``: the runner and the by-content comparator (C5).

Real drivers are not run here (the slowest takes 40 minutes); a fake driver script in
``tmp_path`` stands in, and the comparator is checked per file type on tiny files the tests
write. ``DRIVERS`` is checked against the real scripts' ``--help`` in the slow test.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from ImageLynx import driver_snapshots as ds

FAKE = '''\
import argparse, json, pathlib, sys
parser = argparse.ArgumentParser()
parser.add_argument("--out")
parser.add_argument("--vtk-dir")
parser.add_argument("--fail", action="store_true")
args = parser.parse_args()
print("wrote", args.out)
print("vtk-dir", args.vtk_dir)
if args.out:
    out = pathlib.Path(args.out)
    target = out if out.suffix else out / "result.json"
    target.write_text(json.dumps({"b": 1.5, "a": [1, 2], "where": str(out)}))
print("to stderr", file=sys.stderr)
sys.exit(3 if args.fail else 0)
'''


@pytest.fixture
def scripts(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    for name in ("fake_dir", "fake_file", "fake_stdout", "fake_reader"):
        (folder / f"{name}.py").write_text(FAKE)
    return folder


REGISTRY = (
    ds.Driver("fake_dir", ds.DIR),
    ds.Driver("fake_file", ds.FILE),
    ds.Driver("fake_stdout", ds.STDOUT_ONLY, slow=True),
    ds.Driver("fake_reader", ds.DIR, inputs=(("--vtk-dir", "fake_dir"),)),
)


def _snap(tmp_path, name, scripts, drivers=None):
    chosen = drivers if drivers is not None else ds.select_drivers(registry=REGISTRY)
    return tmp_path / name, ds.take_snapshot(tmp_path / name, chosen, scripts_dir=scripts)


# --- which drivers run -------------------------------------------------------------------


def test_the_default_set_is_every_fast_driver_and_slow_is_opt_in():
    fast = [d.name for d in ds.select_drivers(registry=REGISTRY)]
    assert fast == ["fake_dir", "fake_file", "fake_reader"]
    both = [d.name for d in ds.select_drivers(slow=True, registry=REGISTRY)]
    assert both == ["fake_dir", "fake_file", "fake_stdout", "fake_reader"]


def test_a_named_slow_driver_runs_without_slow_and_a_reader_pulls_its_source_in_first():
    assert [d.name for d in ds.select_drivers(["fake_stdout"], registry=REGISTRY)] == [
        "fake_stdout"]
    assert [d.name for d in ds.select_drivers(["fake_reader"], registry=REGISTRY)] == [
        "fake_dir", "fake_reader"]


def test_an_unknown_driver_name_raises():
    with pytest.raises(ValueError, match="unknown driver.*nope"):
        ds.select_drivers(["nope"], registry=REGISTRY)


def test_the_registry_holds_every_output_writing_driver_but_not_the_tools():
    names = {d.name for d in ds.DRIVERS}
    tools = {"cb_h1_batch", "cb_compare_batch_runs", "cb_driver_snapshots"}
    on_disk = {p.stem for p in ds.EXAMPLES_DIR.glob("cb_*.py")}
    assert on_disk - names == tools
    assert {d.name for d in ds.DRIVERS if d.slow} == {"cb_h2_hypoxic_fraction", "cb_h2_vtk"}


# --- the runner --------------------------------------------------------------------------


def test_each_driver_runs_with_out_inside_the_snapshot_and_its_output_is_kept(tmp_path,
                                                                              scripts):
    snapshot, manifest = _snap(tmp_path, "before", scripts)
    assert (snapshot / "fake_dir" / "out" / "result.json").is_file()
    assert (snapshot / "fake_file" / "out" / "fake_file.json").is_file()
    stdout = (snapshot / "fake_dir" / "stdout.txt").read_text()
    assert f"wrote {snapshot / 'fake_dir' / 'out'}" in stdout
    assert (snapshot / "fake_dir" / "stderr.txt").read_text().strip() == "to stderr"
    assert [d["exit_code"] for d in manifest["drivers"]] == [0, 0, 0]


def test_a_reader_is_pointed_at_its_sources_out_folder(tmp_path, scripts):
    snapshot, _ = _snap(tmp_path, "before", scripts,
                        ds.select_drivers(["fake_reader"], registry=REGISTRY))
    stdout = (snapshot / "fake_reader" / "stdout.txt").read_text()
    assert f"vtk-dir {snapshot / 'fake_dir' / 'out'}" in stdout


def test_the_manifest_records_commit_versions_and_commands_without_the_snapshot_path(
        tmp_path, scripts):
    snapshot, manifest = _snap(tmp_path, "before", scripts)
    assert json.loads((snapshot / "manifest.json").read_text()) == manifest
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ds.REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert manifest["commit"] == head
    assert isinstance(manifest["dirty"], bool)
    assert set(manifest["packages"]) == set(ds.PACKAGES)
    assert manifest["drivers"][0]["command"] == [
        "fake_dir.py", "--out", "<SNAPSHOT>/fake_dir/out"]


def test_a_failing_driver_stops_the_run_and_is_recorded(tmp_path, scripts):
    failing = ds.Driver("fake_dir", ds.DIR, args=("--fail",))
    later = ds.Driver("fake_file", ds.FILE)
    snapshot, manifest = _snap(tmp_path, "before", scripts, [failing, later])
    assert [(d["name"], d["exit_code"]) for d in manifest["drivers"]] == [("fake_dir", 3)]
    assert ds.failed_driver(manifest) == "fake_dir"
    assert (snapshot / "fake_dir" / "stdout.txt").exists()
    assert not (snapshot / "fake_file").exists()


def test_an_existing_snapshot_folder_is_refused(tmp_path, scripts):
    (tmp_path / "before").mkdir()
    with pytest.raises(FileExistsError):
        ds.take_snapshot(tmp_path / "before", [], scripts_dir=scripts)


# --- comparing two snapshots -------------------------------------------------------------


def test_two_runs_of_the_same_drivers_match_although_each_printed_its_own_path(tmp_path,
                                                                               scripts):
    old, _ = _snap(tmp_path, "before", scripts)
    new, _ = _snap(tmp_path, "after", scripts)
    comparisons = ds.compare_snapshots(old, new)
    assert all(c.same for c in comparisons)
    assert "3 drivers compared: all match." in ds.format_report(comparisons)


def test_a_changed_output_is_reported_by_driver_and_file(tmp_path, scripts):
    old, _ = _snap(tmp_path, "before", scripts)
    new, _ = _snap(tmp_path, "after", scripts)
    target = new / "fake_dir" / "out" / "result.json"
    data = json.loads(target.read_text())
    data["b"] = 1.5000001
    target.write_text(json.dumps(data))
    comparisons = {c.driver: c for c in ds.compare_snapshots(old, new)}
    assert not comparisons["fake_dir"].same
    assert comparisons["fake_file"].same
    verdict = next(v for v in comparisons["fake_dir"].files if not v.same)
    assert (verdict.path, verdict.detail) == ("out/result.json", "b: value differs")
    report = ds.format_report(list(comparisons.values()))
    assert "fake_dir: DIFFERENT, 1 of 2 files differ" in report
    assert "  out/result.json: b: value differs" in report


def test_a_file_only_one_snapshot_has_is_a_difference(tmp_path, scripts):
    old, _ = _snap(tmp_path, "before", scripts)
    new, _ = _snap(tmp_path, "after", scripts)
    (new / "fake_dir" / "out" / "extra.txt").write_text("x")
    (new / "fake_file" / "out" / "fake_file.json").unlink()
    comparisons = {c.driver: c for c in ds.compare_snapshots(old, new)}
    assert [(v.path, v.detail) for v in comparisons["fake_dir"].files if not v.same] == [
        ("out/extra.txt", "only in NEW")]
    assert [(v.path, v.detail) for v in comparisons["fake_file"].files if not v.same] == [
        ("out/fake_file.json", "only in OLD")]


def test_stderr_is_saved_but_never_compared(tmp_path, scripts):
    old, _ = _snap(tmp_path, "before", scripts)
    new, _ = _snap(tmp_path, "after", scripts)
    (new / "fake_dir" / "stderr.txt").write_text("a different warning")
    assert all(c.same for c in ds.compare_snapshots(old, new))


def _edit_manifest(snapshot, change):
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("change, message", [
    (lambda m: m["drivers"].pop(), "different drivers or command lines"),
    (lambda m: m["drivers"][0]["command"].append("--extra"), "different drivers or command"),
    (lambda m: m.update(python="0.0.0"), "different python"),
    (lambda m: m["packages"].update(numpy="0.0"), "different packages"),
    (lambda m: m["drivers"][1].update(exit_code=1), "driver fake_file exited non-zero"),
])
def test_snapshots_that_cannot_be_compared_raise_rather_than_report(tmp_path, scripts,
                                                                    change, message):
    old, _ = _snap(tmp_path, "before", scripts)
    new, _ = _snap(tmp_path, "after", scripts)
    _edit_manifest(new, change)
    with pytest.raises(ds.SnapshotsNotComparable, match=message):
        ds.compare_snapshots(old, new)


def test_a_folder_with_no_manifest_is_not_a_snapshot(tmp_path, scripts):
    old, _ = _snap(tmp_path, "before", scripts)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ds.SnapshotsNotComparable, match="not a snapshot"):
        ds.compare_snapshots(old, tmp_path / "empty")


# --- the comparator, per file type -------------------------------------------------------


def _write_pair(tmp_path, name):
    a, b = tmp_path / "old" / name, tmp_path / "new" / name
    a.parent.mkdir(exist_ok=True)
    b.parent.mkdir(exist_ok=True)
    return a, b


def _problems(tmp_path, a, b):
    return ds._file_problems(a, b, tmp_path / "old", tmp_path / "new")


def test_text_swaps_the_snapshot_path_and_reports_the_differing_line(tmp_path):
    a, b = _write_pair(tmp_path, "s.txt")
    a.write_text(f"one\nsaw {tmp_path / 'old'}/x\nthree\n")
    b.write_text(f"one\nsaw {tmp_path / 'new'}/x\nthree\n")
    assert _problems(tmp_path, a, b) == []
    b.write_text(f"one\nsaw {tmp_path / 'new'}/y\nthree\n")
    assert _problems(tmp_path, a, b) == ["line 2 differ"]


def test_json_ignores_key_order_and_equates_nan_but_not_float_noise_or_types(tmp_path):
    a, b = _write_pair(tmp_path, "d.json")
    a.write_text('{"x": NaN, "y": [1, {"z": 2.0}], "p": "' + str(tmp_path / "old") + '"}')
    b.write_text('{"y": [1, {"z": 2.0}], "p": "' + str(tmp_path / "new") + '", "x": NaN}')
    assert _problems(tmp_path, a, b) == []
    b.write_text('{"y": [1, {"z": 2.0000000001}], "p": "<SNAPSHOT>", "x": NaN}')
    assert _problems(tmp_path, a, b) == ["y[1].z: value differs"]
    b.write_text('{"y": [1, {"z": 2}], "p": "<SNAPSHOT>", "x": NaN}')
    assert _problems(tmp_path, a, b) == ["y[1].z: float in OLD, int in NEW"]
    b.write_text('{"y": [1, {"z": 2.0}], "p": "<SNAPSHOT>"}')
    assert _problems(tmp_path, a, b) == ["x: only in OLD"]


def _png(path, pixels, **info):
    from PIL import Image, PngImagePlugin
    meta = PngImagePlugin.PngInfo()
    for key, value in info.items():
        meta.add_text(key, value)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path, pnginfo=meta)


def test_png_compares_decoded_pixels_and_ignores_metadata(tmp_path):
    a, b = _write_pair(tmp_path, "f.png")
    pixels = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    _png(a, pixels, Software="one")
    _png(b, pixels, Software="two")
    assert a.read_bytes() != b.read_bytes()
    assert _problems(tmp_path, a, b) == []
    changed = pixels.copy()
    changed[0, 0, 1] += 1
    changed[2, 3] += 1
    _png(b, changed)
    assert _problems(tmp_path, a, b) == ["2 pixels differ"]
    _png(b, pixels[:, :4])
    assert "in OLD" in _problems(tmp_path, a, b)[0]


def _polydata(pv, flow=None, dtype=np.float64, points=None):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]) if points is None else points
    mesh = pv.PolyData(points, lines=np.array([2, 0, 1, 2, 1, 2]))
    mesh.cell_data["flow"] = np.array([1.0, np.nan] if flow is None else flow, dtype=dtype)
    mesh.point_data["tag"] = np.array([1, 2, 3], dtype=np.int32)
    return mesh


def test_vtk_compares_points_cells_and_arrays_by_name_dtype_and_value(tmp_path):
    pv = pytest.importorskip("pyvista")
    a, b = _write_pair(tmp_path, "m.vtp")
    _polydata(pv).save(a)
    _polydata(pv).save(b)
    assert _problems(tmp_path, a, b) == []   # NaN equals NaN

    _polydata(pv, flow=[1.0, 2.0]).save(b)
    assert _problems(tmp_path, a, b) == ["cell array 'flow' values differ"]

    _polydata(pv, dtype=np.float32).save(b)
    assert "cell array 'flow' is float64" in _problems(tmp_path, a, b)[0]

    moved = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 1e-9]])
    _polydata(pv, points=moved).save(b)
    assert _problems(tmp_path, a, b) == ["points differ (3 in OLD, 3 in NEW)"]

    extra = _polydata(pv)
    extra.point_data["new"] = np.zeros(3)
    extra.save(b)
    assert _problems(tmp_path, a, b) == ["point array 'new' only in NEW"]


def test_vtk_image_data_compares_the_grid_and_its_arrays(tmp_path):
    pv = pytest.importorskip("pyvista")
    a, b = _write_pair(tmp_path, "g.vti")

    def grid(spacing, fill):
        image = pv.ImageData(dimensions=(3, 3, 3), spacing=spacing)
        image.point_data["mask"] = np.full(27, fill, dtype=np.uint8)
        return image

    grid((1, 1, 1), 1).save(a)
    grid((1, 1, 1), 1).save(b)
    assert _problems(tmp_path, a, b) == []
    grid((1, 1, 2), 1).save(b)
    assert _problems(tmp_path, a, b) == ["grid spacing differs"]
    grid((1, 1, 1), 0).save(b)
    assert _problems(tmp_path, a, b) == ["point array 'mask' values differ"]


# --- the registry against the real scripts -----------------------------------------------


def _flags(driver):
    return [part for part in ds.command_for(driver, ds.REPO_ROOT)[2:] if part.startswith("--")]


# A driver run with no flags is left out: some have no parser, so --help would run them in full
# against the CB data.
@pytest.mark.slow
@pytest.mark.parametrize("driver", [d for d in ds.DRIVERS if _flags(d)], ids=lambda d: d.name)
def test_every_registered_driver_accepts_the_flags_it_is_run_with(driver):
    done = subprocess.run([sys.executable, str(ds.EXAMPLES_DIR / f"{driver.name}.py"), "--help"],
                          cwd=ds.REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    for flag in _flags(driver):
        assert flag in done.stdout, f"{driver.name} --help does not list {flag}"
