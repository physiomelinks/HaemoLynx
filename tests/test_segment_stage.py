"""What `segment()` accepts, and what it says when it cannot go on.

The stage settles one question -- which file the rest of the run reads -- and
there are two ways to answer it: name a segmented image, or ask ilastik to make
one. The second is the awkward case, because then `input_path` is the stage's
*output* rather than its input, and the documented way to ask for it is to leave
the setting empty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.pipeline import default_schema, resolve_settings
from haemolynx.pipeline.stages import segment

SCHEMA = default_schema()


def settings_for(tmp_path: Path, **overrides) -> dict:
    values = {"vtk_output_prefix": tmp_path / "run", **overrides}
    return resolve_settings(values, schema=SCHEMA, config_path=None)


def a_segmented_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "mask.tif"
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[1:3, 1:3, :] = 255
    tifffile.imwrite(path, volume)
    return path


def test_an_empty_input_path_is_not_a_crash_when_ilastik_will_fill_it(tmp_path):
    """#127: `Path(None)` before the branch that produces the path.

    `input_path` was converted on the first line, so the one configuration the
    setting exists to support -- leave it empty, let ilastik segment -- died
    with `TypeError: expected str, bytes or os.PathLike object, not NoneType`
    before ilastik was reached. Whatever happens next, it must not be that.
    """
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        ilastik_unsegmented_image_path=tmp_path / "raw.tif",
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )

    with pytest.raises(Exception) as raised:  # noqa: PT011 - the type is the point
        segment(settings)

    assert not isinstance(raised.value, TypeError), raised.value
    # It gets far enough to complain about the thing that is actually missing.
    assert "raw.tif" in str(raised.value)


def test_no_input_path_and_no_ilastik_says_so_plainly(tmp_path):
    """The other half of the same branch: nothing to read and nothing to make.

    This used to reach `Path(None)` too. It is a real mistake either way, so it
    should be named rather than arriving as a TypeError from a conversion.
    """
    settings = settings_for(tmp_path, use_ilastik_segmentation=False, input_path=None)

    with pytest.raises(ValueError, match="input_path must name a segmented image"):
        segment(settings)


def test_a_named_image_is_used_as_it_stands(tmp_path):
    """The ordinary case, so the fix above cannot have quietly changed it."""
    mask = a_segmented_tiff(tmp_path)
    inputs = segment(settings_for(tmp_path, input_path=mask))

    assert inputs.image_path == mask
    assert inputs.input_format == "tif"
    assert inputs.output_dir == tmp_path


def test_a_format_the_pipeline_cannot_read_is_refused(tmp_path):
    other = tmp_path / "mask.png"
    other.write_bytes(b"not really a png")

    with pytest.raises(ValueError, match="Invalid image format"):
        segment(settings_for(tmp_path, input_path=other))


def test_ilastik_without_a_classifier_is_refused(tmp_path):
    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        ilastik_unsegmented_image_path=a_segmented_tiff(tmp_path),
        ilastik_classifier_path=None,
    )

    with pytest.raises(ValueError, match="ilastik_classifier_path"):
        segment(settings)


def test_ilastik_hands_on_the_path_it_produced(tmp_path, monkeypatch):
    """The whole point of leaving `input_path` empty: the stage fills it in."""
    from haemolynx import io

    raw = a_segmented_tiff(tmp_path)
    produced = tmp_path / "segmentations" / "mask_segmented.tif"
    produced.parent.mkdir()
    produced.write_bytes(raw.read_bytes())

    monkeypatch.setattr(
        io, "run_ilastik_headless_segmentation",
        lambda **_kwargs: produced,
    )

    settings = settings_for(
        tmp_path,
        use_ilastik_segmentation=True,
        input_path=None,
        ilastik_unsegmented_image_path=raw,
        ilastik_classifier_path=tmp_path / "classifier.ilp",
        ilastik_output_dir=tmp_path / "segmentations",
    )
    inputs = segment(settings)

    assert inputs.image_path == produced
    assert settings["input_path"] == produced
