"""Training a classifier live, one plane at a time, as ilastik does.

While labels are painted, only the plane in view is classified: its features
are computed once (a few planes at a time, in boxes over worker threads) and
kept while memory allows, and each repaint only trains the forest again on the
painted voxels and walks it over that plane. The painted voxels' own features
are computed once each and remembered. Every feature here is the value the
whole volume gives at that voxel (:mod:`.features`), so what the preview shows
is what segmenting the whole image gives there, and the training voxels are
the ones a full segmentation would be trained on.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from .features import BOX_VOXELS, boxes, feature_halo, features_at, features_of_box, intensity_range
from .model import PixelClassifier, TrainingImage, TrainingVoxels

#: Memory the cached planes' features may take.
PLANE_CACHE_BYTES = 1 << 30

#: Planes computed together along the axis in view, so their margin is paid once.
PLANES_PER_SLAB = 4


def region_features(
    volumes: Sequence[np.ndarray],
    ranges: Sequence[tuple[float, float]],
    voxel_size_zyx: Sequence[float],
    spec,
    region: tuple[slice, slice, slice],
    *,
    workers: int = 1,
    box_voxels: int = BOX_VOXELS,
) -> np.ndarray:
    """The features of every voxel of *region* (slices of the volume), in
    boxes over *workers* threads: float32 ``region shape + (n_features,)``,
    the whole volume's values there."""
    shape = tuple(int(n) for n in volumes[0].shape)
    halo = feature_halo(spec, voxel_size_zyx)
    region = tuple(slice(*part.indices(n)[:2]) for part, n in zip(region, shape))
    extent = tuple(part.stop - part.start for part in region)
    out = np.empty(extent + (spec.n_features,), dtype=np.float32)
    tasks = []
    for _padded, _inner, core in boxes(extent, (0, 0, 0), box_voxels=max(1, box_voxels // 4)):
        core_volume = tuple(slice(r.start + c.start, r.start + c.stop) for r, c in zip(region, core))
        padded = tuple(
            slice(max(0, c.start - h), min(n, c.stop + h)) for c, h, n in zip(core_volume, halo, shape)
        )
        inner = tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(core_volume, padded))
        tasks.append((core, padded, inner))

    def one(task):
        core, padded, inner = task
        out[core] = features_of_box(volumes, ranges, voxel_size_zyx, spec, padded, inner)

    if workers <= 1 or len(tasks) == 1:
        for task in tasks:
            one(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, tasks))
    return out


class LiveClassifier:
    """One image being painted, and the classifier being trained on it.

    *volumes* are the image's channels in canonical order, as the classifier
    reads them; *path* names the image in the classifier's record. Not
    thread-safe: the panel calls it from one worker thread at a time.
    """

    def __init__(
        self,
        classifier: PixelClassifier,
        volumes: Sequence[np.ndarray],
        voxel_size_zyx: Sequence[float],
        *,
        path: str,
        workers: int = 4,
        cache_bytes: int = PLANE_CACHE_BYTES,
    ):
        self.classifier = classifier
        self.volumes = tuple(volumes)
        self.voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx)
        self.shape = tuple(int(n) for n in self.volumes[0].shape)
        self.ranges = tuple(intensity_range(volume) for volume in self.volumes)
        self.image = TrainingImage(
            path=str(path), shape_zyx=self.shape, voxel_size_zyx=self.voxel_size_zyx, ranges=self.ranges
        )
        self.workers = max(1, int(workers))
        self.cache_bytes = int(cache_bytes)
        self._planes: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._points: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def spec(self):
        return self.classifier.spec

    def _plane_bytes(self, axis: int) -> int:
        others = [n for a, n in enumerate(self.shape) if a != axis]
        return int(np.prod(others)) * self.spec.n_features * 4

    def plane_features(self, axis: int, index: int) -> np.ndarray:
        """The features of plane *index* across *axis* (``plane shape +
        (n_features,)``), computed with its slab of neighbours and kept."""
        key = (int(axis), int(index))
        with self._lock:
            if key in self._planes:
                self._planes.move_to_end(key)
                return self._planes[key]
        per_plane = self._plane_bytes(axis)
        slab = max(1, min(PLANES_PER_SLAB, self.cache_bytes // max(4 * per_plane, 1)))
        start = (index // slab) * slab
        stop = min(start + slab, self.shape[axis])
        region = [slice(0, n) for n in self.shape]
        region[axis] = slice(start, stop)
        features = region_features(
            self.volumes, self.ranges, self.voxel_size_zyx, self.spec, tuple(region), workers=self.workers
        )
        with self._lock:
            for offset, plane in enumerate(range(start, stop)):
                self._planes[(axis, plane)] = np.take(features, offset, axis=axis)
            while len(self._planes) > 1 and len(self._planes) * per_plane > self.cache_bytes:
                self._planes.popitem(last=False)
            return self._planes.get(key, np.take(features, index - start, axis=axis))

    def features_at(self, points_zyx: np.ndarray) -> np.ndarray:
        """The features at *points_zyx*: from a cached plane, or remembered,
        or computed in boxes round the points not yet seen."""
        points = np.asarray(points_zyx, dtype=np.int64).reshape(-1, 3)
        out = np.empty((len(points), self.spec.n_features), dtype=np.float32)
        flat = np.ravel_multi_index(points.T, self.shape) if len(points) else np.empty(0, np.int64)
        missing = []
        for row, (point, key) in enumerate(zip(points, flat.tolist())):
            known = self._points.get(key)
            if known is None:
                for axis in range(3):
                    plane = self._planes.get((axis, int(point[axis])))
                    if plane is not None:
                        known = plane[tuple(int(point[a]) for a in range(3) if a != axis)]
                        self._points[key] = known
                        break
            if known is None:
                missing.append(row)
            else:
                out[row] = known
        if missing:
            computed = features_at(self.volumes, self.ranges, self.voxel_size_zyx, self.spec, points[missing])
            out[missing] = computed
            for row, values in zip(missing, computed):
                self._points[int(flat[row])] = values
        return out

    def train(self, labels: np.ndarray) -> PixelClassifier:
        """The classifier trained on every voxel *labels* (the painted layer:
        0 unpainted, else 1 + the label's index) marks on this image, beside
        the voxels it holds from other images."""
        if tuple(labels.shape) != self.shape:
            raise ValueError(f"The labels are {tuple(labels.shape)}, the image {self.shape}.")
        points = np.argwhere(np.asarray(labels) > 0).astype(np.int32)
        classes = (np.asarray(labels)[tuple(points.T)].astype(np.int64) - 1)
        known = classes < len(self.classifier.labels)
        points, classes = points[known], classes[known].astype(np.uint8)
        voxels = TrainingVoxels(
            self.features_at(points), classes, np.zeros(len(classes), np.int32), points
        )
        self.classifier = self.classifier.with_image_voxels(self.image, voxels).trained()
        return self.classifier

    def predict_plane(self, axis: int, index: int, classifier: PixelClassifier | None = None) -> np.ndarray:
        """Each voxel of the plane's most probable label (uint8, an index
        into the labels), split over the worker threads."""
        classifier = classifier or self.classifier
        if classifier.forest is None:
            raise ValueError("Train the classifier first.")
        features = self.plane_features(axis, index)
        flat = features.reshape(-1, self.spec.n_features)
        parts = np.array_split(np.arange(len(flat)), self.workers)
        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                labels = np.concatenate(list(pool.map(lambda rows: classifier.forest.predict(flat[rows]), parts)))
        else:
            labels = classifier.forest.predict(flat)
        return labels.reshape(features.shape[:2])

    def painted_labels(self) -> np.ndarray:
        """The labels this image's strokes in the classifier paint back:
        0 unpainted, else 1 + the label's index."""
        out = np.zeros(self.shape, dtype=np.uint8)
        index = self.classifier.image_index(self.image.path)
        if index is None:
            return out
        training = self.classifier.training
        mine = training.image == index
        points = training.points_zyx[mine]
        inside = np.all((points >= 0) & (points < np.array(self.shape)), axis=1)
        out[tuple(points[inside].T)] = training.classes[mine][inside] + 1
        return out
