"""Helpers for large-vessel mask preprocessing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)

logger = logging.getLogger(__name__)


def _copy_of(mask: np.ndarray, *, low_memory: bool, memmap_directory=None) -> np.ndarray:
    """A boolean copy of *mask* to edit: on disk under the low-RAM option,
    copied a slab of planes at a time."""
    from haemolynx.preprocessing.memmap_support import new_memmap_array, slab_step

    if not low_memory:
        return np.array(mask, dtype=bool)
    out = new_memmap_array(mask.shape, bool, directory=memmap_directory)
    step = slab_step(mask.shape)
    for start in range(0, mask.shape[0], step):
        out[start:start + step] = np.asarray(mask[start:start + step], dtype=bool)
    return out


def _set_at(volume: np.ndarray, labels, keys: np.ndarray, value: bool) -> None:
    """``volume`` set to *value* at the flat C-order *keys*."""
    if len(keys):
        volume[np.unravel_index(keys, labels.shape)] = value


#: Face neighbours only: a voxel is on a component's surface when one of its six
#: face neighbours is outside it.
_FACE_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_FACE_STRUCTURE[1, 1, :] = _FACE_STRUCTURE[1, :, 1] = _FACE_STRUCTURE[:, 1, 1] = True
_FULL_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def dilate_binary_mask_by_microns(
    mask: np.ndarray,
    dilation_microns: float,
    voxel_size_zyx: tuple[float, float, float],
    *,
    low_memory: bool = False,
    memmap_directory=None,
) -> np.ndarray:
    """Dilate a binary mask by a physical radius in microns.

    Uses an EDT threshold in physical units, which supports anisotropic voxels and
    is typically faster than repeated binary dilation for larger radii.
    *voxel_size_zyx* is the spacing of each array axis in canonical ``(z, y, x)``
    order, not the ``(x, y, z)`` order reported by image metadata.

    *low_memory* (the low-RAM option) makes no distance map of the volume:
    each block is transformed with a halo holding every mask voxel within
    the radius (``preprocessing.blockwise_edt.near_distances_by_block``),
    which compares with the radius as the whole transform does, and the mask
    returned is on disk in *memmap_directory*. The whole transform was a
    float64 volume, and scipy ~70 bytes a voxel making it.
    """
    if dilation_microns <= 0:
        return mask.astype(bool, copy=False)

    binary_mask = mask.astype(bool, copy=False)
    if low_memory and binary_mask.ndim == 3:
        return _dilate_by_block(binary_mask, float(dilation_microns), voxel_size_zyx, memmap_directory)
    distance_from_mask = distance_transform_edt(~binary_mask, sampling=voxel_size_zyx)
    return binary_mask | (distance_from_mask <= float(dilation_microns))


def _dilate_by_block(
    mask: np.ndarray, dilation_microns: float, voxel_size_zyx, memmap_directory
) -> np.ndarray:
    """:func:`dilate_binary_mask_by_microns` a block at a time, on disk."""
    from haemolynx.preprocessing import blockwise_edt
    from haemolynx.preprocessing.memmap_support import combine_by_slab

    sampling = tuple(float(v) for v in voxel_size_zyx)
    out = combine_by_slab(lambda m: m.astype(bool), mask, directory=memmap_directory)
    if not _any_by_slab(mask):
        # scipy's transform of a volume with no feature reads every voxel as
        # that far from one at index (-1, 0, 0), so the whole transform
        # "dilates" an empty mask into a corner of the volume: the same here.
        reach = np.minimum(blockwise_edt.reach_voxels(dilation_microns, sampling), mask.shape)
        index = np.indices(tuple(int(r) for r in reach)).reshape(3, -1).T
        displacement = np.column_stack([-1 - index[:, 0], -index[:, 1], -index[:, 2]])
        near = blockwise_edt.displacement_distances(displacement, sampling) <= dilation_microns
        out[tuple(slice(0, int(r)) for r in reach)] = near.reshape(tuple(int(r) for r in reach))
        return out

    from scipy.ndimage import maximum_filter1d

    halo = blockwise_edt.reach_voxels(dilation_microns, sampling)

    def outside(box):
        return ~np.asarray(mask[box], dtype=bool)

    def within_reach(box):
        # Only a voxel with mask inside the reach's box round it can be within
        # the reach of the mask; the rest stay out without being measured.
        padded = tuple(
            slice(max(0, c.start - int(h)), min(int(n), c.stop + int(h)))
            for c, h, n in zip(box, halo, mask.shape)
        )
        near = np.asarray(mask[padded], dtype=np.uint8)
        for axis, h in enumerate(halo):
            near = maximum_filter1d(near, size=2 * int(h) + 1, axis=axis, mode="constant", cval=0)
        inner = tuple(slice(c.start - p.start, c.stop - p.start) for c, p in zip(box, padded))
        return near[inner].astype(bool) & outside(box)

    for core, where, distance in blockwise_edt.near_distances_by_block(
        outside, within_reach, mask.shape, sampling=sampling, reach=dilation_microns
    ):
        near = np.zeros(where.shape, dtype=bool)
        near[where] = distance <= dilation_microns
        out[core] = np.asarray(mask[core], dtype=bool) | near
    return out


def _any_by_slab(mask: np.ndarray) -> bool:
    from haemolynx.preprocessing.memmap_support import slab_step

    step = slab_step(mask.shape)
    return any(bool(np.asarray(mask[z:z + step]).any()) for z in range(0, mask.shape[0], step))


def dilate_large_vessel_masks_by_microns(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    dilation_microns: float,
    voxel_size_zyx: tuple[float, float, float],
    *,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dilate optional arteriole/venule masks by a micron distance (on disk,
    a block at a time, under *low_memory*: :func:`dilate_binary_mask_by_microns`)."""
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask

    if dilation_microns <= 0:
        return (
            large_arteriole_mask.astype(bool, copy=False),
            large_venule_mask.astype(bool, copy=False),
        )

    return (
        dilate_binary_mask_by_microns(
            large_arteriole_mask,
            dilation_microns=dilation_microns,
            voxel_size_zyx=voxel_size_zyx,
            low_memory=low_memory,
            memmap_directory=memmap_directory,
        ),
        dilate_binary_mask_by_microns(
            large_venule_mask,
            dilation_microns=dilation_microns,
            voxel_size_zyx=voxel_size_zyx,
            low_memory=low_memory,
            memmap_directory=memmap_directory,
        ),
    )


def exclude_smaller_overlapping_large_vessel_components(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    *,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Remove overlap voxels from the smaller component in each overlap pair.

    For each overlapping arteriole/venule component pair, the smaller full
    component is identified, and only the voxels in the *overlap region* are
    removed from that smaller component. This suppresses class-leakage overlap
    while preserving non-overlapping parts of the component.
    """
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask

    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    return _exclude_smaller_overlapping_mask_components(
        large_arteriole_mask,
        large_venule_mask,
        low_memory=low_memory,
        memmap_directory=memmap_directory,
    )


def exclude_smaller_overlapping_small_vessel_components(
    small_arteriole_mask: np.ndarray | None,
    small_venule_mask: np.ndarray | None,
    *,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Small-vessel equivalent of overlap cleanup used for large-vessel masks."""
    if small_arteriole_mask is None or small_venule_mask is None:
        return small_arteriole_mask, small_venule_mask

    if small_arteriole_mask.shape != small_venule_mask.shape:
        raise ValueError(
            "small_arteriole_mask and small_venule_mask must share a shape. "
            f"Got {small_arteriole_mask.shape} and {small_venule_mask.shape}."
        )
    return _exclude_smaller_overlapping_mask_components(
        small_arteriole_mask,
        small_venule_mask,
        low_memory=low_memory,
        memmap_directory=memmap_directory,
    )


def swap_minority_touching_vessel_components(
    arteriole_mask: np.ndarray | None,
    venule_mask: np.ndarray | None,
    *,
    max_size_ratio: float = 1.0,
    min_contact_fraction: float = 0.3,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Move each minority arteriole/venule component into the class it touches.

    A vessel segmented partly as arteriole and partly as venule comes out of the
    classifier as two touching components, one of each class. The smaller one
    (the minority) is relabelled -- moved whole into the other mask and cleared
    from its own -- so the vessel is one continuous class. Two components touch
    when they share a voxel or sit 26-adjacent, which is how masks exported one
    label per voxel meet: they never overlap, so removing overlap does nothing.

    A component is swapped when the touching components of the other class that
    are larger than it, and at least ``1 / max_size_ratio`` times its size,
    face at least ``min_contact_fraction`` of its surface area (see
    :func:`_facing_area_fraction`). The contact requirement is what keeps a
    genuine vessel that merely runs beside or across a larger one of the other
    class: a tube lying against a neighbour faces it over a few percent of its
    surface, while a vessel whose labels split lengthwise across its
    cross-section faces the other half over its flat side, about a third. A
    short piece joined end to end onto a longer vessel faces it only with its
    end (about 0.1-0.15), so catching those needs a lower fraction.

    Components are decided largest first, each against its neighbours' labels
    as they stand after the larger decisions, so a chain of alternating pieces
    ends up one consistent class rather than every piece flipping at once.

    The fraction is of surface *area*, each voxel face weighted by its area on
    *voxel_size_zyx*: counted as surface voxels within 26-adjacency of the
    other class, the same tube split lengthwise on 2 x 0.5 x 0.5 um voxels read
    0.54 (swapped) or 0.21 (kept) depending only on which way the split ran --
    a face normal to z covers a quarter of one normal to y, and 26-adjacency
    reaches 2 um in z but 0.5 um in-plane. As facing area both read 0.36.

    The components are labelled a slab of planes at a time and held at their
    voxels only (``preprocessing.slab_labels.ForegroundLabels``): two whole
    label volumes were 8 bytes a voxel each on a stack of 2**31 voxels or
    more. *low_memory* (the low-RAM option) writes the masks returned on disk
    in *memmap_directory*. The same masks either way.
    """
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    stats: dict[str, Any] = {
        "swapped_to_venule_component_count": 0,
        "swapped_to_arteriole_component_count": 0,
        "swapped_to_venule_voxel_count": 0,
        "swapped_to_arteriole_voxel_count": 0,
        "kept_touching_component_count": 0,
        "max_kept_contact_fraction": 0.0,
        "components": [],
    }
    if arteriole_mask is None or venule_mask is None:
        return arteriole_mask, venule_mask, stats
    if arteriole_mask.shape != venule_mask.shape:
        raise ValueError(
            "arteriole and venule masks must share a shape. "
            f"Got {arteriole_mask.shape} and {venule_mask.shape}."
        )
    ratio = float(max_size_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"max_size_ratio must be in (0, 1], got {max_size_ratio!r}.")
    min_contact = float(min_contact_fraction)
    if not 0.0 <= min_contact <= 1.0:
        raise ValueError(
            f"min_contact_fraction must be in [0, 1], got {min_contact_fraction!r}."
        )

    arteriole = arteriole_mask.astype(bool, copy=False)
    venule = venule_mask.astype(bool, copy=False)
    if not np.any(arteriole) or not np.any(venule):
        return arteriole, venule, stats

    labels = {
        "arteriole": ForegroundLabels(arteriole, _FULL_STRUCTURE),
        "venule": ForegroundLabels(venule, _FULL_STRUCTURE),
    }
    sizes = {kind: labels[kind].sizes for kind in labels}
    slices = {kind: labels[kind].find_objects() for kind in labels}
    swapped = {kind: np.zeros(sizes[kind].size, dtype=bool) for kind in labels}
    opposite = {"arteriole": "venule", "venule": "arteriole"}
    largest = {kind: int(sizes[kind][1:].max()) for kind in labels}

    order = sorted(
        (
            (int(sizes[kind][component_id]), kind, component_id)
            for kind in labels
            for component_id in range(1, sizes[kind].size)
            if sizes[kind][component_id] > 0
        ),
        reverse=True,
    )
    for size, kind, component_id in order:
        other = opposite[kind]
        min_partner_size = size / ratio
        if largest[other] <= size or largest[other] < min_partner_size:
            continue
        box = _padded_box(slices[kind][component_id - 1], labels[kind].shape)
        component = labels[kind].read(box) == component_id
        other_labels = labels[other].read(box)
        partner_sizes = sizes[other]
        eligible = (
            (partner_sizes > size)
            & (partner_sizes >= min_partner_size)
            & ~swapped[other]
        )
        eligible[0] = False
        partners = eligible[other_labels]
        touching_partners = partners & binary_dilation(component, _FULL_STRUCTURE)
        if not np.any(touching_partners):
            continue

        contact_fraction = _facing_area_fraction(component, partners, voxel_size_zyx)
        if contact_fraction is None:
            continue
        largest_partner = int(
            partner_sizes[np.unique(other_labels[touching_partners])].max()
        )
        is_swapped = contact_fraction >= min_contact
        stats["components"].append(
            {
                "from": kind,
                "to": other,
                "voxel_count": size,
                "largest_partner_voxel_count": largest_partner,
                "contact_fraction": contact_fraction,
                "swapped": is_swapped,
            }
        )
        if is_swapped:
            swapped[kind][component_id] = True
            stats[f"swapped_to_{other}_component_count"] += 1
            stats[f"swapped_to_{other}_voxel_count"] += size
        else:
            stats["kept_touching_component_count"] += 1
            stats["max_kept_contact_fraction"] = max(
                stats["max_kept_contact_fraction"], contact_fraction
            )

    if not (np.any(swapped["arteriole"]) or np.any(swapped["venule"])):
        return arteriole, venule, stats
    # (arteriole & ~to_venule) | to_arteriole and its mirror: each swapped
    # component's voxels cleared from its own mask and set in the other.
    to_venule = labels["arteriole"].keys[swapped["arteriole"][labels["arteriole"].numbers]]
    to_arteriole = labels["venule"].keys[swapped["venule"][labels["venule"].numbers]]
    out_arteriole = _copy_of(arteriole, low_memory=low_memory, memmap_directory=memmap_directory)
    out_venule = _copy_of(venule, low_memory=low_memory, memmap_directory=memmap_directory)
    _set_at(out_arteriole, labels["arteriole"], to_venule, False)
    _set_at(out_arteriole, labels["arteriole"], to_arteriole, True)
    _set_at(out_venule, labels["venule"], to_arteriole, False)
    _set_at(out_venule, labels["venule"], to_venule, True)
    return out_arteriole, out_venule, stats


def _facing_area_fraction(
    component: np.ndarray,
    partners: np.ndarray,
    voxel_size_zyx: tuple[float, float, float] | None,
) -> float | None:
    """The share of *component*'s surface area whose faces open onto a voxel of
    *partners*, each face weighted by its area on *voxel_size_zyx*; ``None``
    when it has no exposed face. A face on the array's edge is not exposed."""
    spacing = np.ones(3) if voxel_size_zyx is None else np.asarray(voxel_size_zyx, float)
    face_area = (spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1])
    inside = np.pad(component, 1, mode="constant", constant_values=True)
    other = np.pad(partners, 1, mode="constant", constant_values=False)
    core = tuple(slice(1, -1) for _ in range(3))
    total = facing = 0.0
    for axis in range(3):
        for step in (-1, 1):
            neighbour = list(core)
            neighbour[axis] = slice(1 + step, inside.shape[axis] - 1 + step)
            neighbour = tuple(neighbour)
            open_face = component & ~inside[neighbour]
            total += float(np.count_nonzero(open_face)) * face_area[axis]
            facing += float(np.count_nonzero(open_face & other[neighbour])) * face_area[axis]
    return facing / total if total > 0.0 else None


def _padded_box(
    component_slice: tuple[slice, ...], shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """A component's bounding box grown by one voxel, for 26-neighbour tests."""
    return tuple(
        slice(max(0, axis.start - 1), min(size, axis.stop + 1))
        for axis, size in zip(component_slice, shape)
    )


def remove_small_opposite_attached_large_vessel_components(
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    *,
    voxel_size_xyz: tuple[float, float, float],
    max_component_volume_um3: float,
    max_attach_distance_microns: float,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, dict[str, float]]]:
    """Remove tiny opposite-type large-vessel components attached near mask surfaces.

    A component is removed when:
    - its physical volume is <= ``max_component_volume_um3``, and
    - its minimum distance to the opposite mask is <= ``max_attach_distance_microns``.

    Distances are read at each component's surface voxels
    (:func:`_component_distances_to_mask`), not from a distance transform of
    the whole volume. Components are labelled a slab of planes at a time and
    held at their voxels (``preprocessing.slab_labels.ForegroundLabels``);
    *low_memory* (the low-RAM option) measures the distances from tiles of
    the other mask's surface and writes the masks returned on disk in
    *memmap_directory*. The same masks either way.
    """
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    stats = {
        "arteriole": {
            "removed_component_count": 0.0,
            "removed_voxel_count": 0.0,
            "removed_volume_um3": 0.0,
            "candidate_component_count": 0.0,
            "total_component_count": 0.0,
            "too_large_component_count": 0.0,
            "near_opposite_component_count": 0.0,
            "near_opposite_too_large_component_count": 0.0,
            "min_component_distance_microns": float("inf"),
        },
        "venule": {
            "removed_component_count": 0.0,
            "removed_voxel_count": 0.0,
            "removed_volume_um3": 0.0,
            "candidate_component_count": 0.0,
            "total_component_count": 0.0,
            "too_large_component_count": 0.0,
            "near_opposite_component_count": 0.0,
            "near_opposite_too_large_component_count": 0.0,
            "min_component_distance_microns": float("inf"),
        },
    }
    if large_arteriole_mask is None or large_venule_mask is None:
        return large_arteriole_mask, large_venule_mask, stats
    if large_arteriole_mask.shape != large_venule_mask.shape:
        raise ValueError(
            "large_arteriole_mask and large_venule_mask must share a shape. "
            f"Got {large_arteriole_mask.shape} and {large_venule_mask.shape}."
        )
    max_volume = float(max_component_volume_um3)
    max_dist = float(max_attach_distance_microns)
    if max_volume <= 0 or max_dist < 0:
        return (
            large_arteriole_mask.astype(bool, copy=False),
            large_venule_mask.astype(bool, copy=False),
            stats,
        )

    voxel_volume_um3 = (
        float(voxel_size_xyz[0])
        * float(voxel_size_xyz[1])
        * float(voxel_size_xyz[2])
    )
    # Image metadata is (x, y, z); EDT sampling follows array axes (z, y, x).
    sampling_zyx = (
        float(voxel_size_xyz[2]),
        float(voxel_size_xyz[1]),
        float(voxel_size_xyz[0]),
    )
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    def _clean_one_side(
        mask_raw: np.ndarray,
        opposite_mask_raw: np.ndarray,
        *,
        side_key: str,
    ) -> np.ndarray:
        mask = mask_raw.astype(bool, copy=False)
        opposite_mask = opposite_mask_raw.astype(bool, copy=False)
        labels = ForegroundLabels(mask, structure)
        component_count = labels.count
        stats[side_key]["total_component_count"] = float(component_count)
        if component_count <= 0:
            stats[side_key]["min_component_distance_microns"] = 0.0
            return mask
        component_sizes = labels.sizes
        candidate_ids: list[int] = []
        too_large_ids: list[int] = []
        for component_id in range(1, int(component_count) + 1):
            voxel_count = int(component_sizes[int(component_id)])
            volume = float(voxel_count) * float(voxel_volume_um3)
            if volume <= max_volume:
                candidate_ids.append(int(component_id))
            else:
                too_large_ids.append(int(component_id))
        stats[side_key]["candidate_component_count"] = float(len(candidate_ids))
        stats[side_key]["too_large_component_count"] = float(len(too_large_ids))

        component_distance = _component_distances_to_mask(
            labels, int(component_count), opposite_mask, sampling_zyx=sampling_zyx,
            mask=mask, low_memory=low_memory,
        )
        near_opposite_all = 0
        near_opposite_too_large = 0
        min_component_distance = float("inf")
        for component_id in range(1, int(component_count) + 1):
            min_distance = float(component_distance[component_id])
            if not np.isfinite(min_distance):
                continue
            if min_distance < min_component_distance:
                min_component_distance = min_distance
            if min_distance <= max_dist:
                near_opposite_all += 1
                if component_id in too_large_ids:
                    near_opposite_too_large += 1
        if not np.isfinite(min_component_distance):
            min_component_distance = 0.0
        stats[side_key]["near_opposite_component_count"] = float(near_opposite_all)
        stats[side_key]["near_opposite_too_large_component_count"] = float(
            near_opposite_too_large
        )
        stats[side_key]["min_component_distance_microns"] = float(
            min_component_distance
        )

        if not candidate_ids:
            return mask

        cleaned = _copy_of(mask, low_memory=low_memory, memmap_directory=memmap_directory)
        removed_components = 0
        removed_voxels = 0
        for component_id in candidate_ids:
            members = labels.members(int(component_id))
            if not len(members):
                continue
            min_distance = float(component_distance[component_id])
            if min_distance <= max_dist:
                removed_components += 1
                removed_voxels += int(len(members))
                _set_at(cleaned, labels, labels.keys[members], False)

        stats[side_key]["removed_component_count"] = float(removed_components)
        stats[side_key]["removed_voxel_count"] = float(removed_voxels)
        stats[side_key]["removed_volume_um3"] = (
            float(removed_voxels) * float(voxel_volume_um3)
        )
        return cleaned

    cleaned_arteriole = _clean_one_side(
        large_arteriole_mask,
        large_venule_mask,
        side_key="arteriole",
    )
    cleaned_venule = _clean_one_side(
        large_venule_mask,
        large_arteriole_mask,
        side_key="venule",
    )
    return cleaned_arteriole, cleaned_venule, stats


def _component_distances_to_mask(
    labels,
    component_count: int,
    other_mask: np.ndarray,
    *,
    sampling_zyx: tuple[float, float, float],
    mask: np.ndarray | None = None,
    low_memory: bool = False,
) -> np.ndarray:
    """Each labelled component's distance to the nearest voxel of *other_mask*,
    in microns, indexed by component id (``inf`` for an empty *other_mask*).

    Read at the components' surface voxels only, from a KD-tree over the
    other mask's surface (:class:`~haemolynx.preprocessing.pointwise_distance.
    FeatureDistance`): the voxel of a component nearest to anything outside
    it is on its surface, so this is the minimum a whole-volume distance
    transform would give, without the float64 volume (eight bytes a voxel,
    twice over) that transform costs.

    *labels* is a label volume, or a ``slab_labels.ForegroundLabels`` of
    *mask* (the components' own mask); *low_memory* reads the distances from
    tiles of the other mask's surface (``FeatureDistance(by_block=True)``).
    """
    from haemolynx.preprocessing.pointwise_distance import (
        FeatureDistance,
        surface_feature_voxels,
    )

    distances = np.full(int(component_count) + 1, np.inf)
    other = np.asarray(other_mask, dtype=bool)
    if component_count <= 0 or not other.any():
        return distances
    if other.all():
        distances[1:] = 0.0
        return distances
    surface = surface_feature_voxels(labels > 0 if mask is None else np.asarray(mask, dtype=bool))
    if not len(surface):  # the mask fills the volume: every component touches
        distances[1:] = 0.0
        return distances
    measured = FeatureDistance(
        other, feature_value=True, sampling=sampling_zyx, by_block=low_memory
    ).at(surface)
    at = labels[tuple(surface.T)] if isinstance(labels, np.ndarray) else labels.at(tuple(surface.T))
    np.minimum.at(distances, at, measured)
    return distances


def _exclude_smaller_overlapping_mask_components(
    arteriole_mask_raw: np.ndarray,
    venule_mask_raw: np.ndarray,
    *,
    low_memory: bool = False,
    memmap_directory=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove overlap voxels from the smaller component in each overlap pair.

    The components are labelled a slab of planes at a time and read at the
    overlap's voxels only (``preprocessing.slab_labels.ForegroundLabels``);
    two whole label volumes and a mask of each pair's overlap were about
    130 GB on a 5.5 G-voxel stack. *low_memory* writes the masks returned
    on disk in *memmap_directory*. The same masks either way."""
    from haemolynx.preprocessing.memmap_support import slab_step
    from haemolynx.preprocessing.slab_labels import ForegroundLabels

    arteriole_mask = arteriole_mask_raw.astype(bool, copy=False)
    venule_mask = venule_mask_raw.astype(bool, copy=False)
    # The overlap's voxels, in C order, a slab at a time.
    shape = arteriole_mask.shape
    plane = int(np.prod(shape[1:], dtype=np.int64))
    step = slab_step(shape)
    parts = []
    for start in range(0, shape[0], step):
        both = np.asarray(arteriole_mask[start:start + step]) & np.asarray(venule_mask[start:start + step])
        parts.append(np.flatnonzero(both).astype(np.int64) + start * plane)
    overlap_keys = np.concatenate(parts) if parts else np.empty(0, np.int64)
    if not len(overlap_keys):
        return arteriole_mask, venule_mask
    initial_overlap_voxels = int(len(overlap_keys))
    overlap = np.unravel_index(overlap_keys, shape)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    arteriole_labels = ForegroundLabels(arteriole_mask, structure)
    venule_labels = ForegroundLabels(venule_mask, structure)

    arteriole_sizes = arteriole_labels.sizes
    venule_sizes = venule_labels.sizes

    overlap_arteriole_labels = arteriole_labels.at(overlap)
    overlap_venule_labels = venule_labels.at(overlap)
    overlapping_pairs = np.unique(
        np.column_stack([overlap_arteriole_labels, overlap_venule_labels]),
        axis=0,
    )

    overlap_pairs_to_remove_from_arteriole: list[tuple[int, int]] = []
    overlap_pairs_to_remove_from_venule: list[tuple[int, int]] = []
    for arteriole_label, venule_label in overlapping_pairs:
        if arteriole_label == 0 or venule_label == 0:
            continue
        arteriole_size = int(arteriole_sizes[int(arteriole_label)])
        venule_size = int(venule_sizes[int(venule_label)])
        if arteriole_size < venule_size:
            overlap_pairs_to_remove_from_arteriole.append(
                (int(arteriole_label), int(venule_label))
            )
        elif venule_size < arteriole_size:
            overlap_pairs_to_remove_from_venule.append(
                (int(arteriole_label), int(venule_label))
            )
        else:
            overlap_pairs_to_remove_from_venule.append(
                (int(arteriole_label), int(venule_label))
            )

    # Each overlap voxel belongs to one pair: removed from a side where its
    # pair is one that side loses.
    pair_keys = overlap_arteriole_labels.astype(np.int64) * (int(venule_labels.count) + 1) + overlap_venule_labels

    def _in_pairs(pairs: list[tuple[int, int]]) -> np.ndarray:
        if not pairs:
            return np.zeros(len(pair_keys), dtype=bool)
        wanted = np.array([a * (int(venule_labels.count) + 1) + v for a, v in pairs], dtype=np.int64)
        return np.isin(pair_keys, wanted)

    from_arteriole = _in_pairs(overlap_pairs_to_remove_from_arteriole)
    from_venule = _in_pairs(overlap_pairs_to_remove_from_venule)
    cleaned_arteriole_mask = _copy_of(arteriole_mask, low_memory=low_memory, memmap_directory=memmap_directory)
    cleaned_venule_mask = _copy_of(venule_mask, low_memory=low_memory, memmap_directory=memmap_directory)
    _set_at(cleaned_arteriole_mask, arteriole_labels, overlap_keys[from_arteriole], False)
    _set_at(cleaned_venule_mask, venule_labels, overlap_keys[from_venule], False)

    # Outside the overlap the masks were left as they were.
    remaining_overlap_voxels = int(np.count_nonzero(~from_arteriole & ~from_venule))
    removed_overlap_voxels = int(initial_overlap_voxels - remaining_overlap_voxels)
    logger.info(
        "Mask overlap cleanup summary: "
        f"initial_overlap_voxels={initial_overlap_voxels}, "
        f"removed_overlap_voxels={removed_overlap_voxels}, "
        f"remaining_overlap_voxels={remaining_overlap_voxels}."
    )

    return cleaned_arteriole_mask, cleaned_venule_mask
