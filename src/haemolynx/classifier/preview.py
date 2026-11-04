"""Training a classifier live, one plane at a time, as ilastik does.

While labels are painted, only the plane in view is classified -- and of it
only the part on screen (*window*), so a plane of a stack far larger than
memory costs no more than the screen shows. Features are computed in tiles a
few planes deep and kept while memory allows, so moving back to a plane, or
painting again, computes nothing new; each repaint only trains the forest
again on the painted voxels and walks it over the window. The painted
voxels' own features are computed once each and remembered. Every feature
here is the value the whole volume gives at that voxel (:mod:`.features`), so
what the preview shows is what segmenting the whole image gives there, and
the training voxels are the ones a full segmentation would be trained on.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from .features import BOX_VOXELS, boxes, feature_halo, features_of_box, intensity_range
from .model import PixelClassifier, TrainingImage, TrainingVoxels
from .sparse import SparseLabels, painted_points

#: Memory the cached features may take.
PLANE_CACHE_BYTES = 1 << 30

#: Planes computed together along the axis in view, so their margin is paid once.
PLANES_PER_SLAB = 4

#: The side of a cached tile of a plane, in voxels.
TILE = 256

#: The most voxels of a plane the preview classifies at once: a window
#: larger than this is cut down round its centre.
PREVIEW_MAX_VOXELS = 1 << 22


def region_features(
    volumes: Sequence,
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


def clip_window(window, plane_shape: tuple[int, int], limit: int = PREVIEW_MAX_VOXELS):
    """*window* ``((r0, r1), (c0, c1))`` on a plane of *plane_shape* -- the
    whole plane when None -- clipped to it, and cut down round its centre to
    at most *limit* voxels."""
    if window is None:
        window = ((0, plane_shape[0]), (0, plane_shape[1]))
    bounds = []
    for (low, high), n in zip(window, plane_shape):
        low, high = max(0, int(np.floor(low))), min(n, int(np.ceil(high)))
        bounds.append([low, max(high, low + 1)] if low < n else [n - 1, n])
    while (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) > limit:
        axis = 0 if bounds[0][1] - bounds[0][0] >= bounds[1][1] - bounds[1][0] else 1
        low, high = bounds[axis]
        quarter = (high - low) // 4
        bounds[axis] = [low + quarter, high - quarter]
    return tuple(tuple(b) for b in bounds)


class LiveClassifier:
    """One image being painted, and the classifier being trained on it.

    *volumes* are the image's channels in canonical order, as the classifier
    reads them (arrays, memory maps or lazily read channels); *path* names the
    image in the classifier's record. Thread-safe for its cache; the panel
    calls it from one worker thread at a time.
    """

    def __init__(
        self,
        classifier: PixelClassifier,
        volumes: Sequence,
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
        self._tiles: OrderedDict[tuple[int, int, int, int], np.ndarray] = OrderedDict()
        self._points: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def spec(self):
        return self.classifier.spec

    def _plane_axes(self, axis: int) -> tuple[int, int]:
        return tuple(a for a in range(3) if a != axis)  # type: ignore[return-value]

    def _slab(self, axis: int, index: int) -> tuple[int, int]:
        start = (int(index) // PLANES_PER_SLAB) * PLANES_PER_SLAB
        return start, min(start + PLANES_PER_SLAB, self.shape[axis])

    def _tile(self, axis: int, index: int, row: int, column: int) -> np.ndarray:
        """The features of tile (*row*, *column*) of the slab holding plane
        *index* across *axis*: ``(planes, tile rows, tile columns, n_features)``."""
        start, stop = self._slab(axis, index)
        key = (int(axis), start, int(row), int(column))
        with self._lock:
            tile = self._tiles.get(key)
            if tile is not None:
                self._tiles.move_to_end(key)
                return tile
        first, second = self._plane_axes(axis)
        region = [slice(0, n) for n in self.shape]
        region[axis] = slice(start, stop)
        region[first] = slice(row * TILE, min((row + 1) * TILE, self.shape[first]))
        region[second] = slice(column * TILE, min((column + 1) * TILE, self.shape[second]))
        features = region_features(
            self.volumes, self.ranges, self.voxel_size_zyx, self.spec, tuple(region), workers=self.workers
        )
        tile = np.moveaxis(features, axis, 0)
        with self._lock:
            self._tiles[key] = tile
            used = sum(t.nbytes for t in self._tiles.values())
            while len(self._tiles) > 1 and used > self.cache_bytes:
                _key, dropped = self._tiles.popitem(last=False)
                used -= dropped.nbytes
        return tile

    def plane_features(self, axis: int, index: int, window=None) -> np.ndarray:
        """The features of the *window* (``((r0, r1), (c0, c1))`` in the plane's
        own axes, the whole plane when None) of plane *index* across *axis*:
        ``(rows, columns, n_features)``. Tiles are computed side by side on the
        worker threads, and kept."""
        first, second = self._plane_axes(axis)
        (r0, r1), (c0, c1) = clip_window(window, (self.shape[first], self.shape[second]), limit=1 << 62)
        rows = range(r0 // TILE, (r1 - 1) // TILE + 1)
        columns = range(c0 // TILE, (c1 - 1) // TILE + 1)
        start, _stop = self._slab(axis, index)
        wanted = [(r, c) for r in rows for c in columns]
        with self._lock:
            missing = [(r, c) for r, c in wanted if (int(axis), start, r, c) not in self._tiles]
        if len(missing) > 1 and self.workers > 1:
            inner_workers, self.workers = self.workers, 1
            try:
                with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                    list(pool.map(lambda rc: self._tile(axis, index, *rc), missing))
            finally:
                self.workers = inner_workers
        out = np.empty((r1 - r0, c1 - c0, self.spec.n_features), dtype=np.float32)
        for r, c in wanted:
            tile = self._tile(axis, index, r, c)[index - start]
            tr0, tc0 = r * TILE, c * TILE
            a0, a1 = max(r0, tr0), min(r1, tr0 + tile.shape[0])
            b0, b1 = max(c0, tc0), min(c1, tc0 + tile.shape[1])
            out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = tile[a0 - tr0:a1 - tr0, b0 - tc0:b1 - tc0]
        return out

    def _cached_point(self, point) -> np.ndarray | None:
        with self._lock:
            for axis in range(3):
                start, _stop = self._slab(axis, int(point[axis]))
                first, second = self._plane_axes(axis)
                key = (axis, start, int(point[first]) // TILE, int(point[second]) // TILE)
                tile = self._tiles.get(key)
                if tile is not None:
                    return tile[
                        int(point[axis]) - start,
                        int(point[first]) - key[2] * TILE,
                        int(point[second]) - key[3] * TILE,
                    ]
        return None

    def features_at(self, points_zyx: np.ndarray) -> np.ndarray:
        """The features at *points_zyx*: remembered, or from a cached tile,
        or computed in boxes round the points not yet seen."""
        points = np.asarray(points_zyx, dtype=np.int64).reshape(-1, 3)
        out = np.empty((len(points), self.spec.n_features), dtype=np.float32)
        flat = np.ravel_multi_index(points.T, self.shape) if len(points) else np.empty(0, np.int64)
        missing = []
        for row, (point, key) in enumerate(zip(points, flat.tolist())):
            known = self._points.get(key)
            if known is None:
                known = self._cached_point(point)
                if known is not None:
                    self._points[key] = np.array(known)
            if known is None:
                missing.append(row)
            else:
                out[row] = known
        if missing:
            self._tile_points(points, flat, np.asarray(missing), out)
        return out

    def _tile_points(self, points: np.ndarray, flat: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        """The features of *rows* of *points* through the tiles of the
        planes across z they lie on: the tiles the preview would compute,
        side by side on the workers, each tile's points read as soon as it is
        made (and remembered), so the cache never needs to hold them all."""
        groups: dict[tuple[int, int, int], list[int]] = {}
        for row in rows.tolist():
            z, y, x = (int(v) for v in points[row])
            groups.setdefault((self._slab(0, z)[0], y // TILE, x // TILE), []).append(row)

        def one(item):
            (start, tile_row, tile_column), members = item
            tile = self._tile(0, start, tile_row, tile_column)
            for row in members:
                z, y, x = (int(v) for v in points[row])
                values = np.array(tile[z - start, y - tile_row * TILE, x - tile_column * TILE])
                out[row] = values
                self._points[int(flat[row])] = values

        items = list(groups.items())
        if self.workers > 1 and len(items) > 1:
            inner_workers, self.workers = self.workers, 1
            try:
                with ThreadPoolExecutor(max_workers=inner_workers) as pool:
                    list(pool.map(one, items))
            finally:
                self.workers = inner_workers
        else:
            for item in items:
                one(item)

    def train(self, labels) -> PixelClassifier:
        """The classifier trained on every voxel *labels* (the painted layer:
        0 unpainted, else 1 + the label's index; dense or
        :class:`~.sparse.SparseLabels`) marks on this image, beside the
        voxels it holds from other images."""
        if tuple(labels.shape) != self.shape:
            raise ValueError(f"The labels are {tuple(labels.shape)}, the image {self.shape}.")
        points, values = painted_points(labels)
        classes = values.astype(np.int64) - 1
        known = (classes >= 0) & (classes < len(self.classifier.labels))
        points, classes = points[known], classes[known].astype(np.uint8)
        voxels = TrainingVoxels(self.features_at(points), classes, np.zeros(len(classes), np.int32), points)
        self.classifier = self.classifier.with_image_voxels(self.image, voxels).trained()
        return self.classifier

    def predict_plane(
        self, axis: int, index: int, classifier: PixelClassifier | None = None, *, window=None
    ) -> np.ndarray:
        """Each voxel of the plane's *window* (the whole plane when None) its
        most probable label (uint8, an index into the labels), split over the
        worker threads."""
        classifier = classifier or self.classifier
        if classifier.forest is None:
            raise ValueError("Train the classifier first.")
        features = self.plane_features(axis, index, window)
        flat = features.reshape(-1, self.spec.n_features)
        parts = np.array_split(np.arange(len(flat)), self.workers)
        if self.workers > 1 and len(flat) > 4096:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                labels = np.concatenate(list(pool.map(lambda rows: classifier.forest.predict(flat[rows]), parts)))
        else:
            labels = classifier.forest.predict(flat)
        return labels.reshape(features.shape[:2])

    def painted_labels(self, *, sparse: bool = False):
        """The labels this image's strokes in the classifier paint back:
        0 unpainted, else 1 + the label's index; a
        :class:`~.sparse.SparseLabels` with *sparse*."""
        out = SparseLabels(self.shape) if sparse else np.zeros(self.shape, dtype=np.uint8)
        index = self.classifier.image_index(self.image.path)
        if index is None:
            return out
        training = self.classifier.training
        mine = training.image == index
        points = training.points_zyx[mine]
        inside = np.all((points >= 0) & (points < np.array(self.shape)), axis=1)
        if inside.any():
            out[tuple(points[inside].T.astype(np.int64))] = training.classes[mine][inside] + 1
        return out
