"""Guards against reading a raw image larger than memory (``classifier.memory``):
what reading one whole takes, the lazy readers refusing a read past the
memory free, the training window's refusal and preflight's check."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.classifier import memory
from haemolynx.classifier.lazy import LazyTiffChannel
from haemolynx.classifier.sparse import SparseLabels
from haemolynx.gui.classifier_training import memory_refusal

from test_pixel_classifier import tube_volume, write_raw


@pytest.fixture
def little_memory(monkeypatch):
    """Every read weighed, against *free* bytes free of *total*."""

    def set_memory(free, total=None):
        monkeypatch.setattr(memory, "CHECKED_READ_BYTES", 0)
        monkeypatch.setattr(memory, "memory_bytes", lambda: (free, total if total is not None else free))

    return set_memory


def test_reading_a_stack_whole_takes_its_size(tmp_path):
    volume, _ = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    assert memory.raw_image_memory_bytes(raw, (None,)) == volume.nbytes
    # Read in another axis order, the reordered copy is counted too.
    assert memory.raw_image_memory_bytes(raw, (None,), "xyz") == 2 * volume.nbytes


def test_reading_one_channel_of_a_hyperstack_takes_that_channel(tmp_path):
    volume, _ = tube_volume()
    composite = np.stack([volume, volume // 2], axis=1)  # z, c, y, x
    raw = tmp_path / "composite.tif"
    tifffile.imwrite(raw, composite, imagej=True, metadata={"axes": "ZCYX"})
    assert memory.raw_image_memory_bytes(raw, (0,)) == volume.nbytes
    assert memory.raw_image_memory_bytes(raw, (0, 1)) == 2 * volume.nbytes


def test_a_lazy_channel_refuses_a_read_past_the_memory_free(tmp_path, little_memory):
    volume, _ = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    plane = volume[0].nbytes
    little_memory(3 * plane)
    channel = LazyTiffChannel(raw)
    try:
        assert np.array_equal(channel[2], volume[2])
        assert np.array_equal(channel[2:5], volume[2:5])
        with pytest.raises(MemoryError, match="3D view reads a whole image"):
            channel[:]
        with pytest.raises(MemoryError):
            np.asarray(channel)
    finally:
        channel.close()


def test_sparse_labels_refuse_a_read_past_the_memory_free(little_memory):
    labels = SparseLabels((20, 30, 40))
    labels[3, 4, 5] = 2
    little_memory(30 * 40 * 3)
    assert labels[3][4, 5] == 2
    with pytest.raises(MemoryError, match="the painted labels"):
        np.asarray(labels)


def test_with_the_memory_free_unknown_nothing_is_refused(tmp_path, monkeypatch):
    volume, _ = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    monkeypatch.setattr(memory, "CHECKED_READ_BYTES", 0)
    monkeypatch.setattr(memory, "memory_bytes", lambda: (None, None))
    channel = LazyTiffChannel(raw)
    try:
        assert np.array_equal(np.asarray(channel), volume)
    finally:
        channel.close()
    assert memory_refusal(10**12, None, low_ram=False) is None


def test_the_window_refuses_an_image_past_most_of_the_memory_free():
    gb = 10**9
    assert memory_refusal(6 * gb, 10 * gb, low_ram=False) is None
    refusal = memory_refusal(150 * gb, 25 * gb, low_ram=False)
    assert "150.0 GB" in refusal and "25.0 GB" in refusal
    assert '"Low RAM option for large networks (High disk space usage)"' in refusal
    # Under the low-RAM option it is read where it is looked at: no refusal.
    assert memory_refusal(150 * gb, 25 * gb, low_ram=True) is None


def test_preflight_stops_a_read_past_the_machines_memory(tmp_path, monkeypatch):
    from haemolynx.classifier import save_classifier
    from haemolynx.pipeline import default_schema, resolve_settings
    from haemolynx.pipeline.checks import check_pixel_classifiers

    from test_pixel_classifier import trained_classifier

    pytest.importorskip("sklearn")
    volume, truth = tube_volume()
    raw = write_raw(tmp_path / "raw.tif", volume)
    path = save_classifier(trained_classifier(volume, truth), tmp_path / "c.haemoclf")

    def report(low_ram=False):
        settings = resolve_settings({
            "output_dir": tmp_path / "out", "raw_image_path": raw, "use_memmap_loading": low_ram,
            "use_pixel_classifier_segmentation": True, "pixel_classifier_path": path,
        }, schema=default_schema(), config_path=None)
        return check_pixel_classifiers(settings)

    size = volume.nbytes
    monkeypatch.setattr(memory, "memory_bytes", lambda: (size // 4, size // 2))
    errors = report().errors
    assert len(errors) == 1 and "more than this machine's" in errors[0] and "use_memmap_loading" in errors[0]
    monkeypatch.setattr(memory, "memory_bytes", lambda: (size, 100 * size))
    checked = report()
    assert checked.errors == [] and any("most of the" in w for w in checked.warnings)
    monkeypatch.setattr(memory, "memory_bytes", lambda: (size // 4, size // 2))
    checked = report(low_ram=True)
    assert checked.errors == [] and not any("whole takes" in w for w in checked.warnings)
