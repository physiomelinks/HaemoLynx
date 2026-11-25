"""The random forest a pixel classifier votes with, held as plain arrays.

Trained by scikit-learn (:func:`train_forest`), then copied out of it into
:class:`Forest`: every tree's nodes in one set of arrays, with this module's
own vectorised walk down them (:meth:`Forest.predict_proba`). So a saved
classifier needs neither scikit-learn nor a pickle to be read and applied,
reads the same under any scikit-learn version, and opening one runs no code.
Only training needs scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Trees per forest, and how deep each may grow. Thirty-two trees twelve deep
#: classified a test volume as well as ilastik's hundred unlimited ones, and
#: three times as fast.
FOREST_TREES = 32
FOREST_MAX_DEPTH = 12

#: The seed every forest is grown from, so the same training voxels always
#: give the same classifier.
RANDOM_STATE = 0

#: Voxels walked down the trees at once.
PREDICT_CHUNK = 32_768

SCIKIT_LEARN_MISSING = (
    "Training a classifier needs scikit-learn, which is not installed. Fix: "
    "pip install scikit-learn (it comes with HaemoLynx's napari extra)."
)


@dataclass(frozen=True)
class Forest:
    """Every tree's nodes, one row per node, the trees one after another.

    A node sends a voxel left when its feature is at most the node's
    threshold. A leaf sends every voxel to itself (threshold +inf), so a walk
    of :attr:`depth` steps ends on a leaf whatever path it took. *value* is
    the class probabilities each node's training voxels gave.
    """

    roots: np.ndarray  # (trees,) int64
    left: np.ndarray  # (nodes,) int64
    right: np.ndarray  # (nodes,) int64
    feature: np.ndarray  # (nodes,) int64
    threshold: np.ndarray  # (nodes,) float64
    value: np.ndarray  # (nodes, classes) float32
    depth: int
    n_features: int

    @property
    def n_classes(self) -> int:
        return int(self.value.shape[1])

    @property
    def n_trees(self) -> int:
        return int(len(self.roots))

    def arrays(self) -> dict[str, np.ndarray]:
        """The forest as named arrays, for a file."""
        return {
            "forest_roots": self.roots,
            "forest_left": self.left,
            "forest_right": self.right,
            "forest_feature": self.feature,
            "forest_threshold": self.threshold,
            "forest_value": self.value,
            "forest_shape": np.array([self.depth, self.n_features], dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays) -> "Forest":
        depth, n_features = (int(v) for v in arrays["forest_shape"])
        forest = cls(
            roots=np.asarray(arrays["forest_roots"], dtype=np.int64),
            left=np.asarray(arrays["forest_left"], dtype=np.int64),
            right=np.asarray(arrays["forest_right"], dtype=np.int64),
            feature=np.asarray(arrays["forest_feature"], dtype=np.int64),
            threshold=np.asarray(arrays["forest_threshold"], dtype=np.float64),
            value=np.asarray(arrays["forest_value"], dtype=np.float32),
            depth=depth,
            n_features=n_features,
        )
        forest.check()
        return forest

    def check(self) -> None:
        """Refuse a forest whose arrays could walk out of bounds."""
        nodes = len(self.left)
        if not (len(self.right) == len(self.feature) == len(self.threshold) == len(self.value) == nodes):
            raise ValueError("The classifier's forest arrays do not agree in length.")
        for name, index in (("roots", self.roots), ("left", self.left), ("right", self.right)):
            if index.size and (index.min() < 0 or index.max() >= nodes):
                raise ValueError(f"The classifier's forest has {name} pointing past its nodes.")
        if self.feature.size and (self.feature.min() < 0 or self.feature.max() >= self.n_features):
            raise ValueError("The classifier's forest reads a feature it does not have.")

    def _walk_arrays(self):
        """The arrays the walk reads, made once: each node's two children
        side by side (int32), and its threshold as the largest float32 not
        above it -- for a float32 feature ``x``, ``x <= t`` exactly when
        ``x <= that``, so the walk compares in float32 and still goes the
        way scikit-learn's float64 comparison goes."""
        cached = self.__dict__.get("_walk_cache")
        if cached is None:
            threshold32 = self.threshold.astype(np.float32)
            above = threshold32.astype(np.float64) > self.threshold
            threshold32[above] = np.nextafter(threshold32[above], np.float32(-np.inf))
            cached = (
                self.roots.astype(np.int32),
                np.ascontiguousarray(np.stack([self.left, self.right], axis=1), dtype=np.int32),
                self.feature.astype(np.int32),
                threshold32,
                np.ascontiguousarray(self.value, dtype=np.float32),
            )
            object.__setattr__(self, "_walk_cache", cached)
        return cached

    def _check_features(self, features: np.ndarray) -> np.ndarray:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(
                f"The classifier reads {self.n_features} features, given {features.shape}."
            )
        return features

    def votes(self, features: np.ndarray, *, chunk: int = PREDICT_CHUNK) -> np.ndarray:
        """Each voxel's leaf probabilities summed over the trees, tree by tree
        in order from 0.0 (float32 ``(voxels, classes)``): the same sums in
        the same order however the voxels are chunked."""
        features = self._check_features(features)
        roots, children, feature, threshold, value = self._walk_arrays()
        count = len(features)
        out = np.zeros((count, self.n_classes), dtype=np.float32)
        flat = features.ravel()
        children_flat = children.ravel()
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            rows = np.arange(0, stop - start, dtype=np.int32)[:, None] * np.int32(self.n_features)
            block = flat[start * self.n_features: stop * self.n_features]
            node = np.broadcast_to(roots, (stop - start, self.n_trees)).copy()
            for _ in range(self.depth):
                right = block[rows + feature[node]] > threshold[node]
                node = children_flat[2 * node + right]
            acc = out[start:stop]
            for tree in range(self.n_trees):
                acc += value[node[:, tree]]
        return out

    def predict_proba(self, features: np.ndarray, *, chunk: int = PREDICT_CHUNK) -> np.ndarray:
        """The mean over the trees of each voxel's leaf probabilities:
        float32 ``(voxels, classes)`` for float32 *features* ``(voxels, n_features)``."""
        return self.votes(features, chunk=chunk) / np.float32(self.n_trees)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Each voxel's most probable class (uint8; a tie goes to the lower
        class). Compiled with numba when it is installed, the same sums in the
        same order, so the same answer either way."""
        features = self._check_features(features)
        compiled = _compiled_walk()
        if compiled is not None:
            roots, children, feature, threshold, value = self._walk_arrays()
            out = np.empty(len(features), dtype=np.uint8)
            compiled(features.ravel(), self.n_features, roots, children, feature, threshold, value, self.depth, out)
            return out
        return np.argmax(self.votes(features), axis=1).astype(np.uint8)

    @classmethod
    def from_sklearn(cls, model, n_classes: int) -> "Forest":
        """Copy a fitted ``RandomForestClassifier``'s trees out of it.
        *model*'s classes are class numbers below *n_classes*; a class it
        never saw gets probability 0 everywhere."""
        classes = [int(c) for c in model.classes_]
        roots, lefts, rights, features, thresholds, values = [], [], [], [], [], []
        offset = 0
        depth = 0
        for estimator in model.estimators_:
            tree = estimator.tree_
            nodes = int(tree.node_count)
            leaf = tree.children_left[:nodes] < 0
            own = np.arange(nodes, dtype=np.int64)
            lefts.append(np.where(leaf, own, tree.children_left[:nodes]).astype(np.int64) + offset)
            rights.append(np.where(leaf, own, tree.children_right[:nodes]).astype(np.int64) + offset)
            features.append(np.where(leaf, 0, tree.feature[:nodes]).astype(np.int64))
            thresholds.append(np.where(leaf, np.inf, tree.threshold[:nodes]).astype(np.float64))
            raw = np.asarray(tree.value[:nodes, 0, :], dtype=np.float64)
            totals = raw.sum(axis=1, keepdims=True)
            probabilities = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
            full = np.zeros((nodes, int(n_classes)), dtype=np.float32)
            full[:, classes] = probabilities
            values.append(full)
            roots.append(offset)
            depth = max(depth, int(tree.max_depth))
            offset += nodes
        forest = cls(
            roots=np.asarray(roots, dtype=np.int64),
            left=np.concatenate(lefts),
            right=np.concatenate(rights),
            feature=np.concatenate(features),
            threshold=np.concatenate(thresholds),
            value=np.concatenate(values),
            depth=depth,
            n_features=int(model.n_features_in_),
        )
        forest.check()
        return forest


_COMPILED: list = []

#: Set to turn the compiled walk off (the tests compare the two).
DISABLE_NUMBA_ENVIRONMENT = "HAEMOLYNX_CLASSIFIER_NO_NUMBA"


def _compiled_walk():
    """The walk compiled by numba (releasing the GIL, so box workers run it
    side by side), or None when numba is not installed or is turned off."""
    import os

    if os.environ.get(DISABLE_NUMBA_ENVIRONMENT):
        return None
    if _COMPILED:
        return _COMPILED[0]
    try:
        from numba import njit
    except ImportError:
        _COMPILED.append(None)
        return None

    @njit(cache=True, nogil=True)
    def walk(flat, n_features, roots, children, feature, threshold, value, depth, out):  # pragma: no cover - compiled
        # Tree by tree, so one tree's nodes stay in cache across the voxels;
        # each voxel still adds its trees' votes in tree order from 0.0.
        classes = value.shape[1]
        voxels = out.shape[0]
        acc = np.zeros((voxels, classes), dtype=np.float32)
        for tree in range(roots.shape[0]):
            root = roots[tree]
            for voxel in range(voxels):
                base = voxel * n_features
                node = root
                for _ in range(depth):
                    left = children[node, 0]
                    if left == node:
                        break
                    if flat[base + feature[node]] > threshold[node]:
                        node = children[node, 1]
                    else:
                        node = left
                for c in range(classes):
                    acc[voxel, c] += value[node, c]
        for voxel in range(voxels):
            best = 0
            for c in range(1, classes):
                if acc[voxel, c] > acc[voxel, best]:
                    best = c
            out[voxel] = best

    _COMPILED.append(walk)
    return walk


def scikit_learn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def train_forest(
    features: np.ndarray,
    classes: np.ndarray,
    *,
    n_classes: int,
    trees: int = FOREST_TREES,
    max_depth: int = FOREST_MAX_DEPTH,
    n_jobs: int | None = -1,
) -> Forest:
    """A forest fitted to *features* (``(voxels, n_features)``) labelled
    *classes* (class numbers below *n_classes*), grown from
    :data:`RANDOM_STATE`. Needs scikit-learn."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as error:
        raise ImportError(SCIKIT_LEARN_MISSING) from error
    features = np.ascontiguousarray(features, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64).ravel()
    if len(features) != len(classes) or not len(classes):
        raise ValueError("Training needs one class per labelled voxel, and at least one voxel.")
    if classes.min() < 0 or classes.max() >= n_classes:
        raise ValueError(f"Classes must be 0 to {n_classes - 1}, got {sorted(set(classes.tolist()))}.")
    model = RandomForestClassifier(
        n_estimators=int(trees), max_depth=int(max_depth), random_state=RANDOM_STATE, n_jobs=n_jobs
    )
    model.fit(features, classes)
    return Forest.from_sklearn(model, n_classes)
