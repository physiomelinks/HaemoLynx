"""What a run's outputs are named after, and where they go.

Every file a run writes -- the VTK files, the skeleton and graph pickles, the
checkpoints, the statistics CSVs, the manifest, the citations -- goes into one
folder (``output_dir``, :func:`run_output_dir`) and carries one stem: the
analysed image's (``input_path``'s file name, ilastik's output once that
stage has replaced it), or, for a synthetic network
(``network_source=synthetic``), the template run's file name with
``_synthetic`` after it, or ``_surrounded`` for its own network inside a
synthetic one (``network_source=synthetic_surround``). :func:`run_stem` is the one answer; a leaf module,
so the checks, the provenance and the panel's checkpoints can all ask it
without importing the stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from haemolynx.graph.boundaries import network_is_synthetic, network_surrounds_image

SYNTHETIC_STEM_SUFFIX = "_synthetic"
SURROUNDED_STEM_SUFFIX = "_surrounded"


def synthetic_run_stem(template_path: Path | str, *, surrounded: bool = False) -> str:
    """``{template stem}_synthetic`` for the ``.haemorun`` at *template_path*,
    or ``{template stem}_surrounded`` for its network inside a synthetic one."""
    return f"{Path(template_path).stem}{SURROUNDED_STEM_SUFFIX if surrounded else SYNTHETIC_STEM_SUFFIX}"


#: The file type the HaemoLynx classifier writes its masks as.
CLASSIFIER_OUTPUT_SUFFIX = ".tif"


def classifier_segments(settings: Mapping[str, Any]) -> bool:
    """``use_pixel_classifier_segmentation`` in effect: on, ilastik off (its
    switch is hidden while ilastik's is on), and a raw image to segment."""
    if network_is_synthetic(settings) or settings.get("use_ilastik_segmentation"):
        return False
    return bool(settings.get("use_pixel_classifier_segmentation")) and settings.get(
        "raw_image_path"
    ) not in (None, "")


def segments_raw_image(settings: Mapping[str, Any]) -> bool:
    """Whether the run's segmented image is made from its raw image, by
    ilastik or by the HaemoLynx classifier, rather than read from a file."""
    if network_is_synthetic(settings) or settings.get("raw_image_path") in (None, ""):
        return False
    return bool(settings.get("use_ilastik_segmentation")) or classifier_segments(settings)


def classifier_segmented_path(settings: Mapping[str, Any]) -> Path:
    """Where the classifier writes the main image's mask:
    ``{output_dir}/{raw image stem}_segmented.tif``, beside the run's other
    outputs, so a batch keeps each image's in its own folder."""
    from haemolynx.io.ilastik import ilastik_segmented_stem

    output_dir = Path(settings.get("output_dir") or ".")
    return output_dir / f"{ilastik_segmented_stem(settings['raw_image_path'])}{CLASSIFIER_OUTPUT_SUFFIX}"


def run_input_path(settings: Mapping[str, Any]) -> Optional[Path]:
    """The segmented image a run analyses, or None: unset, or a synthetic
    run, which reads no image whatever the setting holds.

    The Input tab's ``input_path`` -- or, while ilastik segments (its switch
    on and a raw image to segment), ilastik's output from the raw image,
    ``{raw image stem}_segmented`` in ``ilastik_output_dir``: ``input_path``
    counts only once `segment` has put that output there, and is otherwise
    an unrelated file left from before ilastik was switched on. The panel's
    checkpoints, its saved run's name and the plots' folder read it before
    `segment` runs, and named their files after that stale file. While the
    HaemoLynx classifier segments, its output in ``output_dir``
    (:func:`classifier_segmented_path`), by the same rule."""
    if network_is_synthetic(settings):
        return None
    value = settings.get("input_path")
    raw = settings.get("raw_image_path")
    if classifier_segments(settings):
        output = classifier_segmented_path(settings)
        if value not in (None, "") and Path(value) == output:
            return Path(value)
        return output
    if settings.get("use_ilastik_segmentation") and raw not in (None, ""):
        from haemolynx.io.ilastik import ilastik_segmented_path, ilastik_segmented_stem

        output_dir = Path(settings.get("ilastik_output_dir") or ".")
        if (
            value not in (None, "")
            and Path(value).parent == output_dir
            and Path(value).stem == ilastik_segmented_stem(raw)
        ):
            return Path(value)
        return ilastik_segmented_path(output_dir, raw, settings.get("ilastik_output_suffix") or ".tif")
    return None if value in (None, "") else Path(value)


def run_stem(settings: Mapping[str, Any]) -> Optional[str]:
    """The stem a run's outputs are named with, or None when the run names
    neither an image nor a template."""
    if network_is_synthetic(settings):
        template = settings.get("synthetic_template_path")
        return None if template in (None, "") else synthetic_run_stem(template, surrounded=network_surrounds_image(settings))
    image = run_input_path(settings)
    return None if image is None else image.stem


def run_output_dir(settings: Mapping[str, Any]) -> Optional[Path]:
    """The folder a run writes its outputs into (``output_dir``), or None
    when it is unset."""
    value = settings.get("output_dir")
    return None if value in (None, "") else Path(value)


__all__ = [
    "CLASSIFIER_OUTPUT_SUFFIX",
    "SURROUNDED_STEM_SUFFIX",
    "SYNTHETIC_STEM_SUFFIX",
    "classifier_segmented_path",
    "classifier_segments",
    "run_input_path",
    "segments_raw_image",
    "run_output_dir",
    "run_stem",
    "synthetic_run_stem",
]
