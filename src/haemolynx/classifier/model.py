"""A trained pixel classifier, and the file it is kept in.

A classifier is a :class:`~haemolynx.classifier.features.FeatureSpec` (which
channels, which scales), its labels, the :class:`~haemolynx.classifier.forest.Forest`
that votes between them, and the voxels it was trained on: each one's
features, label, image and position. Keeping those lets it be trained further
on another image -- the new image's strokes added to the old ones -- and lets
the training window put an image's own strokes back where they were painted.

The file (``.haemoclf``) is a NumPy ``.npz`` archive: plain arrays and one JSON
record, read with ``allow_pickle=False``, so opening a classifier from someone
else runs no code. Its SHA-256 is what a segmentation's record names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FeatureSpec
from .forest import FOREST_MAX_DEPTH, FOREST_TREES, Forest, train_forest

#: What a classifier is trained to find, and its labels, first the background.
#: ``vessels`` is the main vessel mask; ``vessel_classes`` the arteriole and
#: venule masks (large or small), a venule never painted leaving one mask of
#: every vessel of that size, as an arteriole mask without a venule mask is;
#: ``pericytes`` and ``blocks`` the masks of objects the pericyte placement and
#: measurements, and the capillary block perturbation, read (each blob one).
PURPOSES: dict[str, tuple[str, ...]] = {
    "vessels": ("Background", "Vessel"),
    "vessel_classes": ("Background", "Arteriole", "Venule"),
    "vessels_and_classes": ("Background", "Capillary", "Arteriole", "Venule"),
    "vessels_and_sized_classes": (
        "Background", "Capillary", "Large arteriole", "Small arteriole", "Large venule", "Small venule",
    ),
    "pericytes": ("Background", "Pericyte"),
    "blocks": ("Background", "Block"),
    "cells": ("Background", "Cell"),
}

#: Labels renamed since a classifier may have been saved, per purpose: a file
#: naming the old label reads as naming the new one. The combined job's
#: "Vessel" became "Capillary" (2026-11-04).
RENAMED_LABELS: dict[str, dict[str, str]] = {
    "vessels_and_classes": {"Vessel": "Capillary"},
}

#: The mask each purpose can write, and the labels it holds. A combined
#: classifier (``vessels_and_classes``, a vessel neither arteriole nor venule
#: painted "Capillary") makes the main vessel mask of every vessel label, so an
#: arteriole is in the vessel mask and in the arteriole mask both; one that
#: tells large from small (``vessels_and_sized_classes``) makes the large and
#: the small vessel masks' arteriole and venule masks too.
MASK_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "vessels": {"Vessel": ("Vessel",)},
    "vessel_classes": {"Arteriole": ("Arteriole",), "Venule": ("Venule",)},
    "vessels_and_classes": {
        "Vessel": ("Capillary", "Arteriole", "Venule"),
        "Arteriole": ("Arteriole",),
        "Venule": ("Venule",),
    },
    "vessels_and_sized_classes": {
        "Vessel": ("Capillary", "Large arteriole", "Small arteriole", "Large venule", "Small venule"),
        "Large arteriole": ("Large arteriole",),
        "Small arteriole": ("Small arteriole",),
        "Large venule": ("Large venule",),
        "Small venule": ("Small venule",),
    },
    "pericytes": {"Pericyte": ("Pericyte",)},
    "blocks": {"Block": ("Block",)},
    "cells": {"Cell": ("Cell",)},
}

#: The purposes a classifier for each job may have.
JOB_PURPOSES: dict[str, tuple[str, ...]] = {
    "vessels": ("vessels", "vessels_and_classes", "vessels_and_sized_classes"),
    "vessel_classes": ("vessel_classes", "vessels_and_classes", "vessels_and_sized_classes"),
    "pericytes": ("pericytes",),
    "blocks": ("blocks",),
    "cells": ("cells",),
}

#: The main vessel mask's name (:data:`MASK_LABELS`).
VESSEL_LABEL = "Vessel"

#: The vessel-mask roles (large, small) each vessel purpose makes masks for,
#: from one file: the classifier in the main image's row fills these roles'.
VESSEL_MASK_ROLES: dict[str, tuple[str, ...]] = {
    "vessels_and_classes": ("large",),
    "vessels_and_sized_classes": ("large", "small"),
}


def role_class_masks(purpose: str, role: str) -> tuple[str, str]:
    """``(arteriole mask, venule mask)``: the masks of a vessel-mask *role*
    (``"large"`` or ``"small"``) a classifier for *purpose* writes. One that
    tells large from small writes each role its own; any other, the same."""
    if purpose == "vessels_and_sized_classes":
        size = role.capitalize()
        return f"{size} arteriole", f"{size} venule"
    return "Arteriole", "Venule"

FILE_SUFFIX = ".haemoclf"
FILE_KIND = "haemolynx-pixel-classifier"
#: The file's layout. A file of a newer layout is refused.
FILE_VERSION = 1


@dataclass(frozen=True)
class TrainingImage:
    """An image a classifier was trained on, as it was read."""

    path: str
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx: tuple[float, float, float]
    ranges: tuple[tuple[float, float], ...]
    #: Each channel's profile on its range (``features.intensity_range_and_profile``):
    #: what an image the classifier is later given is compared with
    #: (``classifier.familiar``). Empty for an image recorded before profiles were.
    profile: tuple[tuple[float, ...], ...] = ()

    def to_record(self) -> dict:
        record = {
            "path": self.path,
            "shape_zyx": list(self.shape_zyx),
            "voxel_size_zyx": list(self.voxel_size_zyx),
            "ranges": [list(r) for r in self.ranges],
        }
        if self.profile:
            record["profile"] = [list(p) for p in self.profile]
        return record

    @classmethod
    def from_record(cls, record: dict) -> "TrainingImage":
        return cls(
            path=str(record["path"]),
            shape_zyx=tuple(int(v) for v in record["shape_zyx"]),  # type: ignore[arg-type]
            voxel_size_zyx=tuple(float(v) for v in record["voxel_size_zyx"]),  # type: ignore[arg-type]
            ranges=tuple((float(a), float(b)) for a, b in record["ranges"]),
            profile=tuple(tuple(float(v) for v in p) for p in record.get("profile") or ()),
        )


@dataclass(frozen=True)
class TrainingVoxels:
    """The labelled voxels a classifier was trained on."""

    features: np.ndarray  # (n, n_features) float32
    classes: np.ndarray  # (n,) uint8, index into the labels
    image: np.ndarray  # (n,) int32, index into the images
    points_zyx: np.ndarray  # (n, 3) int32

    @classmethod
    def empty(cls, n_features: int) -> "TrainingVoxels":
        return cls(
            features=np.empty((0, n_features), np.float32),
            classes=np.empty(0, np.uint8),
            image=np.empty(0, np.int32),
            points_zyx=np.empty((0, 3), np.int32),
        )

    def __len__(self) -> int:
        return int(len(self.classes))

    def without_image(self, index: int) -> "TrainingVoxels":
        keep = self.image != index
        return TrainingVoxels(
            self.features[keep], self.classes[keep], self.image[keep], self.points_zyx[keep]
        )

    def joined(self, other: "TrainingVoxels") -> "TrainingVoxels":
        return TrainingVoxels(
            np.concatenate([self.features, other.features]),
            np.concatenate([self.classes, other.classes]),
            np.concatenate([self.image, other.image]),
            np.concatenate([self.points_zyx, other.points_zyx]),
        )


@dataclass(frozen=True)
class PixelClassifier:
    purpose: str
    spec: FeatureSpec
    images: tuple[TrainingImage, ...] = ()
    training: TrainingVoxels | None = None
    forest: Forest | None = None
    trees: int = FOREST_TREES
    max_depth: int = FOREST_MAX_DEPTH
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.purpose not in PURPOSES:
            raise ValueError(f"Unknown classifier purpose {self.purpose!r}; one of {sorted(PURPOSES)}.")
        if self.training is None:
            object.__setattr__(self, "training", TrainingVoxels.empty(self.spec.n_features))

    @property
    def labels(self) -> tuple[str, ...]:
        return PURPOSES[self.purpose]

    def label_index(self, name: str) -> int:
        return self.labels.index(name)

    def mask_labels(self, mask: str) -> tuple[int, ...]:
        """The label indices mask *mask* holds (:data:`MASK_LABELS`), those
        the classifier was trained on; raises for a mask it cannot make."""
        masks = MASK_LABELS[self.purpose]
        if mask not in masks:
            raise ValueError(f"A {self.purpose} classifier makes no {mask} mask; it makes {', '.join(masks)}.")
        trained = set(self.trained_labels())
        found = tuple(self.label_index(name) for name in masks[mask] if name in trained)
        if not found:
            raise ValueError(f"This classifier was never shown any {' or '.join(masks[mask])} voxels.")
        return found

    def trained_labels(self) -> list[str]:
        """The labels with at least one training voxel, in label order."""
        present = set(np.unique(self.training.classes).tolist())
        return [name for index, name in enumerate(self.labels) if index in present]

    def voxels_per_label(self) -> dict[str, int]:
        counts = np.bincount(self.training.classes, minlength=len(self.labels))
        return {name: int(counts[index]) for index, name in enumerate(self.labels)}

    def image_index(self, path: str | Path) -> int | None:
        wanted = _same_path_key(path)
        for index, image in enumerate(self.images):
            if _same_path_key(image.path) == wanted:
                return index
        return None

    def with_image_voxels(self, image: TrainingImage, voxels: TrainingVoxels) -> "PixelClassifier":
        """This classifier with *image*'s labelled voxels replaced by *voxels*
        (whose ``image`` entries are ignored): another image's stay."""
        index = self.image_index(image.path)
        images = list(self.images)
        if index is None:
            index = len(images)
            images.append(image)
        else:
            images[index] = image
        kept = self.training.without_image(index)
        mine = replace(voxels, image=np.full(len(voxels), index, dtype=np.int32))
        return replace(self, images=tuple(images), training=kept.joined(mine), forest=None)

    def trained(self, *, n_jobs: int | None = -1) -> "PixelClassifier":
        """This classifier with its forest grown on its training voxels.
        At least two labels must have voxels."""
        if len(self.trained_labels()) < 2:
            raise ValueError(
                "Paint at least two labels before training: a classifier needs examples of the "
                "background and of what it is to find."
            )
        forest = train_forest(
            self.training.features,
            self.training.classes,
            n_classes=len(self.labels),
            trees=self.trees,
            max_depth=self.max_depth,
            n_jobs=n_jobs,
        )
        return replace(self, forest=forest)


def _same_path_key(path: str | Path) -> str:
    try:
        return os.path.normcase(str(Path(path).resolve()))
    except OSError:
        return os.path.normcase(str(path))


def save_classifier(classifier: PixelClassifier, path: str | Path) -> Path:
    """Write *classifier* to *path* (a trained one: it must have a forest),
    replacing the file only once the new one is complete."""
    if classifier.forest is None:
        raise ValueError("Train the classifier before saving it.")
    path = Path(path)
    from haemolynx import __version__

    record = {
        "kind": FILE_KIND,
        "version": FILE_VERSION,
        "purpose": classifier.purpose,
        "labels": list(classifier.labels),
        "spec": classifier.spec.to_record(),
        "images": [image.to_record() for image in classifier.images],
        "trees": int(classifier.trees),
        "max_depth": int(classifier.max_depth),
        "voxels_per_label": classifier.voxels_per_label(),
        "haemolynx_version": __version__,
        "notes": classifier.notes,
    }
    arrays = dict(classifier.forest.arrays())
    arrays.update(
        record=np.array(json.dumps(record)),
        training_features=classifier.training.features,
        training_classes=classifier.training.classes,
        training_image=classifier.training.image,
        training_points=classifier.training.points_zyx,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)
    return path


def read_classifier_record(path: str | Path) -> dict:
    """The JSON record of the classifier at *path*, without its arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return _record_of(archive, path)


def _record_of(archive, path) -> dict:
    try:
        record = json.loads(str(archive["record"][()]))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{Path(path).name} is not a HaemoLynx classifier.") from error
    if not isinstance(record, dict) or record.get("kind") != FILE_KIND:
        raise ValueError(f"{Path(path).name} is not a HaemoLynx classifier.")
    if int(record.get("version", 0)) > FILE_VERSION:
        raise ValueError(
            f"{Path(path).name} was saved by a newer HaemoLynx (file version {record['version']}); "
            "update HaemoLynx to read it."
        )
    return _with_current_label_names(record)


def _with_current_label_names(record: dict) -> dict:
    """*record* with any label renamed since it was saved (:data:`RENAMED_LABELS`)
    under its current name, in its labels and its voxel counts."""
    renamed = RENAMED_LABELS.get(record.get("purpose"), {})
    if not renamed:
        return record
    record = dict(record)
    record["labels"] = [renamed.get(name, name) for name in record.get("labels", [])]
    counts = record.get("voxels_per_label")
    if isinstance(counts, dict):
        record["voxels_per_label"] = {renamed.get(name, name): n for name, n in counts.items()}
    return record


def load_classifier(path: str | Path) -> PixelClassifier:
    """The classifier saved at *path* (:func:`save_classifier`). Raises
    ``ValueError`` for a file that is not one, or not whole."""
    path = Path(path)
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path.name} cannot be read as a HaemoLynx classifier: {error}") from error
    with archive:
        record = _record_of(archive, path)
        spec = FeatureSpec.from_record(record["spec"])
        purpose = str(record["purpose"])
        if purpose not in PURPOSES or list(record.get("labels", [])) != list(PURPOSES[purpose]):
            raise ValueError(f"{path.name} names labels this HaemoLynx does not know: {record.get('labels')}.")
        try:
            forest = Forest.from_arrays(archive)
            training = TrainingVoxels(
                features=np.asarray(archive["training_features"], dtype=np.float32),
                classes=np.asarray(archive["training_classes"], dtype=np.uint8),
                image=np.asarray(archive["training_image"], dtype=np.int32),
                points_zyx=np.asarray(archive["training_points"], dtype=np.int32),
            )
        except KeyError as error:
            raise ValueError(f"{path.name} is missing part of its classifier ({error}).") from error
    if forest.n_features != spec.n_features or forest.n_classes != len(PURPOSES[purpose]):
        raise ValueError(f"{path.name}'s forest does not match its features or labels.")
    return PixelClassifier(
        purpose=purpose,
        spec=spec,
        images=tuple(TrainingImage.from_record(image) for image in record.get("images", [])),
        training=training,
        forest=forest,
        trees=int(record.get("trees", FOREST_TREES)),
        max_depth=int(record.get("max_depth", FOREST_MAX_DEPTH)),
        notes=dict(record.get("notes") or {}),
    )


def new_classifier(
    purpose: str, *, scales_um: Sequence[float] | None = None, channels=(None,), coarse: bool = True
) -> PixelClassifier:
    """An untrained classifier for *purpose*. A new classifier computes its
    wide scales on a coarse grid (*coarse*, ``FeatureSpec.coarse``)."""
    spec = FeatureSpec(channels=tuple(channels), coarse=coarse) if scales_um is None else FeatureSpec(
        scales_um=tuple(scales_um), channels=tuple(channels), coarse=coarse
    )
    return PixelClassifier(purpose=purpose, spec=spec)
