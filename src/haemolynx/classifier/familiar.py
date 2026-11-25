"""Whether an image is like those a classifier was trained on.

The classifier reads each channel scaled to its own range (``features.intensity_range``),
so an image brighter or dimmer than the training images reads alike. What the
range leaves is the shape of the histogram -- where the background, the
vessels and the bright tail sit on that range -- and an image whose histogram
sits elsewhere (another stain, another depth, saturated, mostly empty) is one
the forest has never seen the like of. Each training image keeps its profile
(``features.PROFILE_PERCENTILES`` on its range, ``TrainingImage.profile``); an
image whose profile lies more than :data:`UNFAMILIAR_DISTANCE` from every one of
them, on a channel, is called unfamiliar. A warning only: nothing is refused.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .features import FeatureSpec, intensity_range_and_profile
from .model import PixelClassifier, TrainingImage

#: How far (the largest difference of any percentile, as a fraction of the
#: channel's range) an image's profile may lie from the nearest training
#: image's before it is called unfamiliar. Not fitted on marked images.
UNFAMILIAR_DISTANCE = 0.2

#: About how many voxels preflight reads a raw image's profile from: fewer
#: than a segmentation's range, to be quick on a large image.
PREFLIGHT_SAMPLE_VOXELS = 1_000_000

#: Profiles read for preflight, by file, its size and time, channel and sample.
_PROFILES: dict[tuple, tuple[float, ...]] = {}


def profile_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """The largest difference between two profiles' percentiles."""
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _unfamiliar(
    channels: Sequence[int | None], images: Sequence[TrainingImage], profiles: Mapping[int | None, Sequence[float]]
) -> list[tuple[int | None, float]]:
    found = []
    for index, channel in enumerate(channels):
        known = [image.profile[index] for image in images if len(image.profile) > index]
        if not known or channel not in profiles:
            continue
        distance = min(profile_distance(profiles[channel], profile) for profile in known)
        if distance > UNFAMILIAR_DISTANCE:
            found.append((channel, distance))
    return found


def unfamiliar_channels(
    classifier: PixelClassifier, profiles: Mapping[int | None, Sequence[float]]
) -> list[tuple[int | None, float]]:
    """``(channel, distance)`` for each channel the classifier reads whose
    profile in *profiles* lies more than :data:`UNFAMILIAR_DISTANCE` from
    every training image's. A channel no training image has a profile of (a
    classifier saved before profiles were kept), or *profiles* lacks, is
    not judged."""
    return _unfamiliar(classifier.spec.channels, classifier.images, profiles)


def unfamiliar_channels_in_record(
    record: Mapping, profiles: Mapping[int | None, Sequence[float]]
) -> list[tuple[int | None, float]]:
    """:func:`unfamiliar_channels` from a classifier file's record
    (``model.read_classifier_record``), without loading its forest."""
    channels = FeatureSpec.from_record(record["spec"]).channels
    images = [TrainingImage.from_record(image) for image in record.get("images") or ()]
    return _unfamiliar(channels, images, profiles)


def raw_image_profiles(
    path: str | Path, channels: Sequence[int | None], *, voxels: int = PREFLIGHT_SAMPLE_VOXELS
) -> dict[int | None, tuple[float, ...]]:
    """Each of *channels*' profile in the raw TIFF at *path*, read a page at a
    time and kept while the file is unchanged. A channel that cannot be read
    so (a file not stored a page per plane) is left out."""
    from .lazy import LazyTiffChannel

    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return {}
    found: dict[int | None, tuple[float, ...]] = {}
    for channel in channels:
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, channel, int(voxels))
        if key not in _PROFILES:
            try:
                reader = LazyTiffChannel(path, channel)
            except (OSError, ValueError):
                continue
            try:
                _PROFILES[key] = intensity_range_and_profile(reader, voxels)[1]
            finally:
                reader.close()
        found[channel] = _PROFILES[key]
    return found


def unfamiliar_note(image_name: str, classifier_name: str, found: Sequence[tuple[int | None, float]]) -> str:
    """What to tell a user of an image the classifier finds unfamiliar."""
    channels = ", ".join(
        ("its channel" if channel is None else f"channel C{channel + 1}") + f" ({distance:.0%} of its range off)"
        for channel, distance in found
    )
    return (
        f"{image_name} is unlike every image {classifier_name} was trained on: the spread of its "
        f"intensities differs ({channels}). Check its masks, or train the classifier on it too."
    )
