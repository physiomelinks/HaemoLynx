"""Connected components of a volume, labelled a slab of z-planes at a time.

``scipy.ndimage.label`` of a whole volume writes an int32 for every voxel --
22 GB for a 5.5 G-voxel stack -- before anything is read back from it. Here
each slab of whole z-planes is labelled on its own, and two slabs are joined
wherever a voxel in one's last plane touches one in the next's first through
*structure*. That finds the same components: every step of a path between
two voxels lies inside one slab or crosses between the two planes a join
compares. Components are numbered the way ``label`` numbers them, by their
first voxel in C order, so either can stand in for the other.

Nothing is kept per voxel. A caller that needs something at some voxels --
their component, or which components hold a marked voxel -- collects it from
each slab's own labels as the slabs go by (*visit*), and turns those
slab-local labels into component numbers afterwards
(:meth:`SlabLabels.numbers`). Labelling the slabs again
(:meth:`SlabLabels.replay`) gives the same slab labels, for a second pass
that writes something per voxel.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import memmap_support

#: ``foreground(z0, z1)``: the volume's planes ``z0:z1``, as booleans.
Planes = Callable[[int, int], np.ndarray]
#: ``visit(z0, z1, labels, offset)``, called with each slab's own labels.
Visit = Callable[[int, int, np.ndarray, int], None]


@dataclass(frozen=True)
class SlabLabels:
    """The components :func:`label_by_slab` found."""

    #: How many components there are, numbered 1 to *count*.
    count: int
    #: Component number of each slab label once offset (``label + offset``);
    #: entry 0 is the background.
    number: np.ndarray
    #: ``(count, 3)`` first voxel and one past the last of each component's
    #: tight box -- what ``find_objects`` gives for component ``n`` at row
    #: ``n - 1``.
    lo: np.ndarray
    hi: np.ndarray
    structure: np.ndarray
    planes_per_slab: int
    shape: tuple[int, int, int]
    #: Each slab's offset, in z order: what :meth:`label_slab` adds.
    offsets: tuple[int, ...] = ()
    #: How many labels each slab has, in z order.
    slab_counts: tuple[int, ...] = ()

    def box(self, n: int) -> tuple[slice, slice, slice]:
        """Component *n*'s tight box, as ``find_objects`` gives it."""
        return tuple(slice(int(a), int(b)) for a, b in zip(self.lo[n - 1], self.hi[n - 1]))

    def numbers(self, labels: np.ndarray, offset: int) -> np.ndarray:
        """Component number of each of a slab's *labels* (0 stays 0)."""
        labels = np.asarray(labels)
        out = np.zeros(labels.shape, dtype=self.number.dtype)
        inside = labels > 0
        out[inside] = self.number[labels[inside].astype(np.int64) + int(offset)]
        return out

    def number_dtype(self) -> np.dtype:
        """int32 while every component number fits one, else int64."""
        return np.dtype(np.int32) if self.count < np.iinfo(np.int32).max else np.dtype(np.int64)

    def label_slab(self, index: int, foreground: Planes) -> tuple[int, int, np.ndarray, int]:
        """``(z0, z1, labels, offset)`` of slab *index* alone, labelled again
        as :meth:`replay` gives it -- *foreground* must give the planes it
        gave when they were labelled."""
        z0 = int(index) * self.planes_per_slab
        z1 = min(z0 + self.planes_per_slab, self.shape[0])
        labels, _n = label(np.asarray(foreground(z0, z1), dtype=bool), structure=self.structure)
        return z0, z1, labels, int(self.offsets[index])

    def slabs(self) -> Iterator[tuple[int, int, int, int]]:
        """``(z0, z1, offset, n)`` per slab: its planes, and its labels'
        offset and count -- ``offset + 1`` to ``offset + n`` index
        :attr:`number`. A slab whose components all answer alike needs no
        labelling again."""
        offset = 0
        for i, z0 in enumerate(range(0, self.shape[0], self.planes_per_slab)):
            n = int(self.slab_counts[i])
            yield z0, min(z0 + self.planes_per_slab, self.shape[0]), offset, n
            offset += n

    def replay(self, foreground: Planes) -> Iterator[tuple[int, int, np.ndarray, int]]:
        """``(z0, z1, labels, offset)`` per slab, labelled again exactly as
        :func:`label_by_slab` labelled them -- *foreground* must give the
        same planes it gave then."""
        offset = 0
        for z0 in range(0, self.shape[0], self.planes_per_slab):
            z1 = min(z0 + self.planes_per_slab, self.shape[0])
            labels, n = label(np.asarray(foreground(z0, z1), dtype=bool), structure=self.structure)
            yield z0, z1, labels, offset
            offset += int(n)


def _face_links(
    before: np.ndarray, after: np.ndarray, before_offset: int, after_offset: int, steps
) -> np.ndarray:
    """``(n, 2)`` pairs of offset labels joined between the last plane of one
    slab (*before*) and the first of the next (*after*), each pair once."""
    rows, cols = before.shape
    # Each pair as one int64 of the two slab-local labels: neither slab can
    # hold more labels than voxels, so their product stays in range.
    width = int(after.max()) + 1
    found = []
    for dy, dx in steps:
        b = before[max(0, -dy):rows - max(0, dy), max(0, -dx):cols - max(0, dx)]
        a = after[max(0, dy):rows - max(0, -dy), max(0, dx):cols - max(0, -dx)]
        both = (b > 0) & (a > 0)
        if both.any():
            found.append(b[both].astype(np.int64) * width + a[both].astype(np.int64))
    if not found:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(np.concatenate(found))
    return np.column_stack([pairs // width + before_offset, pairs % width + after_offset])


def label_by_slab(
    foreground: Planes,
    shape: tuple[int, int, int],
    *,
    structure: np.ndarray,
    block_voxels: int | None = None,
    visit: Visit | None = None,
) -> SlabLabels:
    """The connected components (through *structure*, a 3x3x3 ``label``
    structure) of a volume of *shape* read through *foreground*, a slab of
    about *block_voxels* voxels at a time (``memmap_support.LOW_MEMORY_BLOCK_VOXELS``
    when not given; a whole volume's worth labels it in one piece).

    *visit*, when given, is called with each slab's ``label`` output and the
    offset that makes its labels unique across slabs: ``labels + offset``
    (where ``labels > 0``) indexes :attr:`SlabLabels.number`.
    """
    shape, structure = _checked(shape, structure)
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    step = memmap_support.slab_step(shape, block_voxels)
    joined = _Join(shape, structure, step)
    for z0 in range(0, shape[0], step):
        z1 = min(z0 + step, shape[0])
        labels, n = label(np.asarray(foreground(z0, z1), dtype=bool), structure=structure)
        n = int(n)
        offset = joined.add(z0, n, _boxes(labels) if n else None, labels[0], labels[-1].copy())
        if visit is not None:
            visit(z0, z1, labels, offset)
    return joined.finish()


def _checked(shape, structure) -> tuple[tuple[int, int, int], np.ndarray]:
    shape = tuple(int(s) for s in shape)
    structure = np.asarray(structure, dtype=bool)
    if len(shape) != 3 or structure.shape != (3, 3, 3):
        raise ValueError("label_by_slab labels 3D volumes with a 3x3x3 structure")
    return shape, structure


def _boxes(labels: np.ndarray) -> np.ndarray:
    """``(n, 6)`` each label's ``find_objects`` box, starts then stops, in the slab."""
    return np.array(
        [[s.start for s in box] + [s.stop for s in box] for box in find_objects(labels)],
        dtype=np.int64,
    )


class _Join:
    """The slabs' labels joined into components, fed one slab at a time in z
    order: its label count, boxes, and first and last planes."""

    def __init__(self, shape, structure, step):
        self.shape, self.structure, self.step = shape, structure, step
        self.steps = [(int(dy) - 1, int(dx) - 1) for dy, dx in np.argwhere(structure[2])]
        self.links: list[np.ndarray] = []
        self.lo_parts: list[np.ndarray] = []
        self.hi_parts: list[np.ndarray] = []
        self.counts: list[int] = []
        self.offset = 0
        self.previous = None  # the slab before's last plane and offset; None when it had no labels

    def add(self, z0: int, n: int, boxes, first, last) -> int:
        """Join one slab; returns its labels' offset."""
        offset = self.offset
        if n:
            boxes = np.array(boxes, dtype=np.int64)
            boxes[:, 0] += z0
            boxes[:, 3] += z0
            self.lo_parts.append(boxes[:, :3])
            self.hi_parts.append(boxes[:, 3:])
            if self.previous is not None:
                self.links.append(_face_links(self.previous[0], first, self.previous[1], offset, self.steps))
        self.previous = (last, offset) if n else None
        self.counts.append(int(n))
        self.offset += int(n)
        return offset

    def finish(self) -> SlabLabels:
        offset, structure, step, shape = self.offset, self.structure, self.step, self.shape
        counts = tuple(self.counts)
        offsets = tuple(int(v) for v in np.cumsum((0,) + counts[:-1])) if counts else ()
        slabs = dict(offsets=offsets, slab_counts=counts)
        if offset == 0:
            empty = np.empty((0, 3), dtype=np.int64)
            return SlabLabels(0, np.zeros(1, dtype=np.int64), empty, empty, structure, step, shape, **slabs)
        pairs = np.concatenate(self.links) if self.links else np.empty((0, 2), dtype=np.int64)
        if len(pairs):
            graph = coo_matrix(
                (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                shape=(offset + 1, offset + 1),
            )
            _n_groups, group = connected_components(graph, directed=False)
        else:
            group = np.arange(offset + 1)
        # Number the groups by their first slab label: slabs come in z order and
        # ``label`` numbers each slab's in C order, so that is the order in which
        # their first voxels come -- ``label``'s own numbering of the whole volume.
        groups, first = np.unique(group[1:], return_index=True)
        rank = np.empty(len(groups), dtype=np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(len(groups))
        number = np.zeros(offset + 1, dtype=np.int64)
        number[1:] = rank[np.searchsorted(groups, group[1:])] + 1
        count = len(groups)
        lo = np.full((count, 3), np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.zeros((count, 3), dtype=np.int64)
        np.minimum.at(lo, number[1:] - 1, np.concatenate(self.lo_parts))
        np.maximum.at(hi, number[1:] - 1, np.concatenate(self.hi_parts))
        return SlabLabels(count, number, lo, hi, structure, step, shape, **slabs)


#: Bytes of RAM a worker holds per voxel of the slab it labels: the slab
#: read and as booleans, scipy's int32 labels, and the int64 indexes a
#: replay looks its answers up with.
SLAB_BYTES_PER_VOXEL = 24


def label_by_slab_in_workers(
    read,
    volumes: tuple[np.ndarray, ...],
    shape: tuple[int, int, int],
    *,
    structure: np.ndarray,
    collect=None,
    params=None,
    block_voxels: int | None = None,
    workers: int = 1,
    memmap_directory=None,
    on_low_memory=None,
) -> tuple[SlabLabels, list[tuple[int, object]]]:
    """:func:`label_by_slab` with each slab labelled in one of *workers*
    processes, then joined here in z order -- so the same components,
    numbered the same way, however many workers there are.

    ``read(volumes, z0, z1, params)`` gives the planes' foreground, and
    ``collect(volumes, z0, z1, labels, params)``, when given, what the
    caller keeps of a slab's own labels (what a *visit* would gather); both
    module-level, so workers can be sent them. Returns the components and,
    per slab, ``(offset, collected)``: ``labels + offset`` indexes
    :attr:`SlabLabels.number` as a visit's do. *volumes* are shared as
    :func:`~.shared_volume_tasks.iter_volume_tasks` shares them.
    """
    from .shared_volume_tasks import iter_volume_tasks

    shape, structure = _checked(shape, structure)
    if block_voxels is None:
        block_voxels = memmap_support.LOW_MEMORY_BLOCK_VOXELS
    step = memmap_support.slab_step(shape, block_voxels)
    starts = list(range(0, shape[0], step))
    payloads = [(read, collect, z0, min(z0 + step, shape[0]), structure, params) for z0 in starts]
    plane = int(shape[1]) * int(shape[2])
    joined = _Join(shape, structure, step)
    collected = []
    results = iter_volume_tasks(
        _label_slab, payloads, volume=tuple(volumes), workers=workers,
        memmap_directory=memmap_directory, task_bytes=step * plane * SLAB_BYTES_PER_VOXEL,
        on_low_memory=on_low_memory,
    )
    for z0, (n, boxes, first, last, kept) in zip(starts, results):
        collected.append((joined.add(z0, n, boxes, first, last), kept))
    return joined.finish(), collected


def _label_slab(volumes, payload):
    """One slab of :func:`label_by_slab_in_workers`: its label count, boxes,
    first and last planes (in the smallest type their labels fit), and what
    *collect* keeps."""
    read, collect, z0, z1, structure, params = payload
    labels, n = label(np.asarray(read(volumes, z0, z1, params), dtype=bool), structure=structure)
    n = int(n)
    small = np.min_scalar_type(n)
    first = labels[0].astype(small)
    # One plane: the same array twice, which pickling sends once.
    last = first if z1 - z0 == 1 else labels[-1].astype(small)
    kept = None if collect is None else collect(volumes, z0, z1, labels, params)
    return n, (_boxes(labels) if n else None), first, last, kept


def replay_in_workers(
    components: SlabLabels,
    read,
    write,
    volumes: tuple[np.ndarray, ...],
    *,
    relabel,
    params=None,
    workers: int = 1,
    memmap_directory=None,
    on_low_memory=None,
) -> list:
    """``write(volumes, z0, z1, labels, offset, n, params)`` for every slab, in
    *workers* processes, and its results in z order. *labels* is the slab
    labelled again as :func:`label_by_slab_in_workers` labelled it -- read
    through the same *read* -- for a slab where ``relabel(offset, n)`` says
    its components do not all answer alike, else None. *write* writes the
    slab's own planes of an output (a whole memmap it opens itself, so
    workers can), so the slabs need nothing from each other."""
    from .shared_volume_tasks import iter_volume_tasks

    slabs = list(components.slabs())
    payloads = [
        (read, write, z0, z1, offset, n, bool(n and relabel(offset, n)), components.structure, params)
        for z0, z1, offset, n in slabs
    ]
    plane = int(components.shape[1]) * int(components.shape[2])
    return list(
        iter_volume_tasks(
            _replay_slab, payloads, volume=tuple(volumes), workers=workers,
            memmap_directory=memmap_directory,
            task_bytes=components.planes_per_slab * plane * SLAB_BYTES_PER_VOXEL,
            on_low_memory=on_low_memory,
        )
    )


def _replay_slab(volumes, payload):
    read, write, z0, z1, offset, n, again, structure, params = payload
    labels = None
    if again:
        labels, _n = label(np.asarray(read(volumes, z0, z1, params), dtype=bool), structure=structure)
    return write(volumes, z0, z1, labels, offset, n, params)


def offset_lookup(labels: np.ndarray, offset: int, table: np.ndarray) -> np.ndarray:
    """``table[labels + offset]`` where a slab's *labels* are set, ``table[0]``
    elsewhere: a per-offset-label answer, read off at each voxel."""
    index = np.array(labels, dtype=np.int64)
    index[index > 0] += int(offset)
    return np.asarray(table)[index]


def voxels_by_component(
    foreground: Planes,
    shape: tuple[int, int, int],
    *,
    structure: np.ndarray,
    block_voxels: int | None = None,
) -> tuple[SlabLabels, np.ndarray, np.ndarray]:
    """``(components, keys, numbers)``: every foreground voxel's flat C-order
    index into *shape* and its component number, in C order."""
    plane = int(shape[1]) * int(shape[2])
    keys = []
    ids = []

    def visit(z0, _z1, labels, offset):
        flat = labels.reshape(-1)
        at = np.flatnonzero(flat)
        keys.append(at.astype(np.int64) + int(z0) * plane)
        ids.append(flat[at].astype(np.int64) + int(offset))

    components = label_by_slab(
        foreground, shape, structure=structure, block_voxels=block_voxels, visit=visit
    )
    if not keys:
        return components, np.empty(0, np.int64), np.empty(0, np.int64)
    return components, np.concatenate(keys), components.number[np.concatenate(ids)]


def numbers_at(
    foreground: Planes,
    query: Planes,
    shape: tuple[int, int, int],
    *,
    structure: np.ndarray,
    block_voxels: int | None = None,
) -> tuple[SlabLabels, "ComponentLookup"]:
    """The components of *foreground*, and a lookup of the component number
    at every voxel where *query* is True (0 where that voxel is background)."""
    plane = int(shape[1]) * int(shape[2])
    keys = []
    ids = []

    def visit(z0, z1, labels, offset):
        wanted = np.flatnonzero(np.asarray(query(z0, z1), dtype=bool).reshape(-1))
        here = labels.reshape(-1)[wanted].astype(np.int64)
        keys.append(wanted.astype(np.int64) + int(z0) * plane)
        ids.append(np.where(here > 0, here + int(offset), 0))

    components = label_by_slab(
        foreground, shape, structure=structure, block_voxels=block_voxels, visit=visit
    )
    keys_all = np.concatenate(keys) if keys else np.empty(0, np.int64)
    ids_all = np.concatenate(ids) if ids else np.empty(0, np.int64)
    numbers = components.number[ids_all].astype(components.number_dtype(), copy=False)
    return components, ComponentLookup(keys_all, numbers, shape)


class ComponentLookup:
    """Component numbers at a fixed set of voxels, looked up by coordinate."""

    def __init__(self, keys: np.ndarray, numbers: np.ndarray, shape):
        self._keys = keys
        self._numbers = numbers
        self._plane = int(shape[1]) * int(shape[2])
        self._row = int(shape[2])

    def keys_of(self, voxels) -> np.ndarray:
        z, y, x = voxels
        return (
            np.asarray(z, dtype=np.int64) * self._plane
            + np.asarray(y, dtype=np.int64) * self._row
            + np.asarray(x, dtype=np.int64)
        )

    def found(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(numbers, found)`` at *keys* (flat C-order indices): each
        number where *found*, anything elsewhere."""
        keys = np.asarray(keys, dtype=np.int64)
        if len(self._keys) == 0:
            return np.zeros(keys.shape, dtype=self._numbers.dtype), np.zeros(keys.shape, dtype=bool)
        at = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        return self._numbers[at], self._keys[at] == keys

    def __getitem__(self, voxels) -> np.ndarray | int:
        """The component number at ``(z, y, x)`` -- three ints, or three
        index arrays as for fancy indexing -- which must be among the voxels
        looked up."""
        numbers, found = self.found(self.keys_of(voxels))
        if not np.all(found):
            raise KeyError("a voxel that was not looked up")
        return int(numbers) if np.ndim(numbers) == 0 else numbers


class NumbersAnywhere:
    """Component numbers at any voxel: from *known* (a
    :class:`ComponentLookup` at the voxels asked about most) where it has
    them, else from the voxel's slab labelled again -- through *foreground*,
    which must give the planes as they were labelled -- the last
    *slabs_kept* of them kept as component numbers for the next lookups.

    For the low-RAM option's thin-arm join: the skeleton's voxels are
    looked up from the start, and a bridge drawn later is read where it
    lands. Looked up at every voxel of the fat region instead, the numbers
    were 16 bytes a fat voxel (2.3 GB for 144 M), twice over."""

    def __init__(self, components: SlabLabels, known: ComponentLookup, foreground: Planes, *, slabs_kept: int = 2):
        self.components = components
        self.known = known
        self.foreground = foreground
        self.slabs_kept = max(1, int(slabs_kept))
        self._slabs: OrderedDict[int, tuple[int, np.ndarray]] = OrderedDict()

    def _slab_numbers(self, index: int) -> tuple[int, np.ndarray]:
        cached = self._slabs.get(index)
        if cached is not None:
            self._slabs.move_to_end(index)
            return cached
        z0, _z1, labels, offset = self.components.label_slab(index, self.foreground)
        number = self.components.number[offset:offset + int(labels.max()) + 1].astype(
            self.components.number_dtype()
        )
        number[0] = 0
        entry = (z0, number[labels])
        del labels
        self._slabs[index] = entry
        while len(self._slabs) > self.slabs_kept:
            self._slabs.popitem(last=False)
        return entry

    def __getitem__(self, voxels) -> np.ndarray | int:
        z, y, x = (np.asarray(v, dtype=np.int64) for v in voxels)
        scalar = z.ndim == 0
        z, y, x = np.atleast_1d(z), np.atleast_1d(y), np.atleast_1d(x)
        numbers, found = self.known.found(self.known.keys_of((z, y, x)))
        numbers = np.array(numbers, dtype=self.components.number_dtype())
        missing = np.flatnonzero(~found)
        if missing.size:
            slab = z[missing] // self.components.planes_per_slab
            for index in np.unique(slab):
                z0, slab_numbers = self._slab_numbers(int(index))
                these = missing[slab == index]
                numbers[these] = slab_numbers[z[these] - z0, y[these], x[these]]
        return int(numbers[0]) if scalar else numbers


class ForegroundLabels:
    """A label volume held at its foreground voxels only: each one's flat
    C-order key and component number, in C order -- 12 bytes a foreground
    voxel (the numbers int32 while they fit), where ``scipy.ndimage.label``'s
    own volume is 4 bytes a voxel of the whole stack, and 8 once the stack
    has 2**31 voxels: 44 GB for a 5.5 G-voxel mask. A volume is labelled a
    slab of planes at a time (:func:`voxels_by_component`); numbered as
    ``label(mask, structure)`` numbers it, so each method answers what the
    same read of that label volume would.

    *structure* is ``label``'s (its default, face connectivity, when None).
    """

    def __init__(self, mask: np.ndarray, structure: np.ndarray | None = None, *, block_voxels: int | None = None):
        from scipy.ndimage import generate_binary_structure

        if not (hasattr(mask, "shape") and hasattr(mask, "__getitem__")):
            mask = np.asanyarray(mask)  # else read in place (a memmap_support.MaskUnion, say)
        self.shape = tuple(int(s) for s in mask.shape)
        structure = (
            generate_binary_structure(mask.ndim, 1) if structure is None else np.asarray(structure, dtype=bool)
        )
        if mask.ndim == 3 and structure.shape == (3, 3, 3):
            components, keys, numbers = voxels_by_component(
                lambda z0, z1: np.asarray(mask[z0:z1], dtype=bool), self.shape,
                structure=structure, block_voxels=block_voxels,
            )
            self.count = int(components.count)
            self._boxes = [components.box(n) for n in range(1, self.count + 1)]
        else:
            labels, count = label(np.asarray(mask[...], dtype=bool), structure=structure)
            flat = labels.reshape(-1)
            keys = np.flatnonzero(flat).astype(np.int64)
            numbers = flat[keys]
            self.count = int(count)
            self._boxes = list(find_objects(labels))
            del labels, flat
        dtype = np.int32 if self.count < np.iinfo(np.int32).max else np.int64
        self.keys = keys
        self.numbers = np.asarray(numbers).astype(dtype, copy=False)
        #: Voxels per component, as ``np.bincount(labels.ravel())``: index 0
        #: the background's.
        self.sizes = np.bincount(self.numbers, minlength=self.count + 1).astype(np.int64)
        self.sizes[0] = int(np.prod(self.shape, dtype=np.int64)) - len(self.keys)
        self._order = None

    def find_objects(self) -> list[tuple[slice, ...]]:
        """``find_objects(labels)``: each component's tight box."""
        return list(self._boxes)

    def box(self, number: int) -> tuple[slice, ...]:
        return self._boxes[int(number) - 1]

    def coords(self) -> np.ndarray:
        """``np.argwhere(labels)``: the foreground voxels in C order -- and
        laid out as ``np.argwhere`` lays them out (Fortran order, the
        transpose of the index arrays), since a BLAS product of them (a
        covariance, ``x.T @ x``) rounds by their layout."""
        return np.transpose(np.unravel_index(self.keys, self.shape))

    def component_coords(self, number: int) -> np.ndarray:
        """``np.argwhere(labels[box] == number) + box start``: component
        *number*'s voxels in C order, laid out as that expression gives them
        (see :meth:`coords`)."""
        return np.transpose(np.unravel_index(self.keys[self.members(number)], self.shape))

    def at(self, voxels) -> np.ndarray:
        """``labels[voxels]`` for a tuple of index arrays (in range)."""
        keys = np.ravel_multi_index(tuple(np.asarray(v, dtype=np.intp) for v in voxels), self.shape)
        out = np.zeros(np.shape(keys), dtype=self.numbers.dtype)
        if len(self.keys):
            at = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
            found = self.keys[at] == keys
            out[found] = self.numbers[at[found]]
        return out

    def __getitem__(self, box) -> np.ndarray:
        """``labels[box]`` for a box of step-1 slices: :meth:`read`, so a
        reader of boxes of a label volume can be handed this instead."""
        box = box if isinstance(box, tuple) else (box,)
        box = box + (slice(None),) * (len(self.shape) - len(box))
        bounds = []
        for s, n in zip(box, self.shape):
            if not isinstance(s, slice):
                raise TypeError("ForegroundLabels is read by boxes of slices")
            start, stop, step = s.indices(n)
            if step != 1:
                raise ValueError("ForegroundLabels is read by step-1 slices")
            bounds.append(slice(start, max(start, stop)))
        return self.read(tuple(bounds))

    def read(self, box: tuple[slice, ...]) -> np.ndarray:
        """``labels[box]`` (in-range slices, step 1), from the voxels in it."""
        starts = [int(s.start) for s in box]
        extents = [int(s.stop) - int(s.start) for s in box]
        out = np.zeros(extents, dtype=self.numbers.dtype)
        if not len(self.keys) or not all(extents):
            return out
        # Each row along the last axis is one run of keys.
        lead = np.indices(extents[:-1]).reshape(len(extents) - 1, -1) + np.array(starts[:-1])[:, None]
        row = np.ravel_multi_index(tuple(lead) + (np.full(lead.shape[1], starts[-1]),), self.shape)
        lo = np.searchsorted(self.keys, row)
        hi = np.searchsorted(self.keys, row + extents[-1])
        counts = hi - lo
        if not counts.sum():
            return out
        which = np.repeat(np.arange(len(row)), counts)
        index = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(lo, counts)
        flat_out = out.reshape(len(row), extents[-1])
        flat_out[which, self.keys[index] - row[which]] = self.numbers[index]
        return out

    def members(self, number: int) -> np.ndarray:
        """Indices into :attr:`keys` of component *number*'s voxels, C order."""
        if self._order is None:
            self._order = np.argsort(self.numbers, kind="stable")
            self._bounds = np.searchsorted(self.numbers[self._order], np.arange(self.count + 2))
        return self._order[self._bounds[int(number)]:self._bounds[int(number) + 1]]

    def write(self, out: np.ndarray, lut: np.ndarray, *, block_voxels: int | None = None) -> np.ndarray:
        """``out[...] = lut[labels]``, a slab of planes at a time."""
        lut = np.asarray(lut)
        plane = int(np.prod(self.shape[1:], dtype=np.int64)) if len(self.shape) > 1 else 1
        step = memmap_support.slab_step(self.shape, block_voxels)
        for z0 in range(0, self.shape[0], step):
            z1 = min(z0 + step, self.shape[0])
            slab = np.full((z1 - z0,) + self.shape[1:], lut[0], dtype=out.dtype)
            lo, hi = np.searchsorted(self.keys, [z0 * plane, z1 * plane])
            slab.reshape(-1)[self.keys[lo:hi] - z0 * plane] = lut[self.numbers[lo:hi]]
            out[z0:z1] = slab
        return out
