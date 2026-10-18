"""What a run's outputs are named after.

Every file a run writes -- the skeleton and graph pickles, the checkpoints,
the statistics CSVs, the manifest, the citations -- carries one stem: the
analysed image's (``input_path``'s file name, ilastik's output once that
stage has replaced it), or, for a synthetic network
(``network_source=synthetic``), the template run's file name with
``_synthetic`` after it. :func:`run_stem` is the one answer; a leaf module,
so the checks, the provenance and the panel's checkpoints can all ask it
without importing the stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from haemolynx.graph.boundaries import network_is_synthetic

SYNTHETIC_STEM_SUFFIX = "_synthetic"


def synthetic_run_stem(template_path: Path | str) -> str:
    """``{template stem}_synthetic`` for the ``.haemorun`` at *template_path*."""
    return f"{Path(template_path).stem}{SYNTHETIC_STEM_SUFFIX}"


def run_input_path(settings: Mapping[str, Any]) -> Optional[Path]:
    """The image a run analyses (``input_path``), or None: unset, or a
    synthetic run, which reads no image whatever the setting holds."""
    if network_is_synthetic(settings):
        return None
    value = settings.get("input_path")
    return None if value in (None, "") else Path(value)


def run_stem(settings: Mapping[str, Any]) -> Optional[str]:
    """The stem a run's outputs are named with, or None when the run names
    neither an image nor a template."""
    if network_is_synthetic(settings):
        template = settings.get("synthetic_template_path")
        return None if template in (None, "") else synthetic_run_stem(template)
    image = run_input_path(settings)
    return None if image is None else image.stem


__all__ = ["SYNTHETIC_STEM_SUFFIX", "run_input_path", "run_stem", "synthetic_run_stem"]
