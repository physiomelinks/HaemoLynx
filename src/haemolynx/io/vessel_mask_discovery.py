"""Find a run's vessel masks beside its input image, by file name.

A user who keeps the large and small arteriole and venule masks in the folder
of the image they were segmented from need not browse for each one: a mask's
name is the image's name, or part of it, with its size and vessel type added
-- ``E14_MCA_large_arterioles.tif`` beside ``E14_MCA_segmented.tif``.
:func:`find_vessel_masks` reads the folder for them, and the panel fills the
mask rows from what it finds (``gui/vessel_mask_search.py``).

A name says which mask it is through whole words, in any case and split at
punctuation, spaces and capitals (``LargeArterioles`` is two words): ``large``
or ``small`` for the size, and for the type ``arteriole``, ``arterioles``,
``artery`` or ``arteries``, or ``venule``, ``venules``, ``vein`` or ``veins``.
A size with no type (``E14_MCA_large_vessels.tif``), or with both, is a global
mask of every vessel that size. A type with no size (``E14_MCA_arteriole.tif``)
is a large mask, but only while no file beside the image says ``large``: one
that does decides the large masks alone (Harvey, 2026-10-18; his E14.5 exports
are ``..._Simple Segmentation_arteriole.tiff`` and ``..._Vein.tiff``). It is
never a small mask.

Only the words a mask's name adds to the image's are read for its role, so an
image called ``large_field.tif`` does not make every file beside it a large
mask. Everything else in a mask's name must be in the image's name too, apart
from words that say nothing about which image it is (``mask``, ``segmented``,
ilastik's ``Simple Segmentation`` ...): ``E15_large_arteriole.tif`` is another
image's mask, not ``E14``'s. Of several masks for one role the one sharing
the image's name furthest from its start is taken; a tie is left unchosen.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

from .vessel_masks import VESSEL_MASK_SETTINGS

#: The file types a vessel mask loads from (`load_large_vessel_masks`).
MASK_SUFFIXES = (".tif", ".tiff", ".h5")

#: The words naming each mask size.
SIZE_WORDS: dict[str, frozenset[str]] = {
    "large": frozenset({"large"}),
    "small": frozenset({"small"}),
}

#: The words naming each vessel type. A size with neither is a global mask.
TYPE_WORDS: dict[str, frozenset[str]] = {
    "arteriole": frozenset({"arteriole", "arterioles", "artery", "arteries"}),
    "venule": frozenset({"venule", "venules", "vein", "veins"}),
}

_ROLE_WORDS = frozenset().union(*SIZE_WORDS.values(), *TYPE_WORDS.values())

#: Words a mask's name may add to its image's without naming another image.
_UNIDENTIFYING_WORDS = frozenset(
    {
        "mask",
        "masks",
        "segmented",
        "segmentation",
        "seg",
        "simple",
        "binary",
        "label",
        "labels",
        "labelled",
        "labeled",
        "vessel",
        "vessels",
        "and",
        "raw",
        "unsegmented",
    }
)

#: Words that, added to the image's name, mark a file that is not a mask: an
#: ilastik probability map, or the raw image a mask was segmented from.
_NOT_A_MASK_WORDS = frozenset({"probabilities", "probability", "raw", "unsegmented"})

_FORMAT = {".tif": "tiff", ".tiff": "tiff", ".h5": "h5"}

#: A word: a run of capitals not starting a capitalised word, a capitalised
#: or lower-case word, or a number.
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@lru_cache(maxsize=1 << 16)
def _stem_words(stem: str) -> tuple[str, ...]:
    return tuple(word.lower() for word in _WORD.findall(stem))


def file_name_words(stem: str) -> list[str]:
    """The words of a file name (no extension), lower case, in order."""
    return list(_stem_words(stem))


@dataclass(frozen=True)
class FoundVesselMasks:
    """What :func:`find_vessel_masks` found for one mask size.

    ``arteriole``, ``venule`` and ``global_mask`` are the file chosen for each
    kind of mask, or None; ``ambiguous`` holds each kind whose best files tied,
    with them, none of which was chosen. ``size_named`` is False when the
    files name only a vessel type, read as large since none says ``large``."""

    size: str
    arteriole: Path | None = None
    venule: Path | None = None
    global_mask: Path | None = None
    ambiguous: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    size_named: bool = True

    @property
    def is_pair(self) -> bool:
        return self.arteriole is not None and self.venule is not None

    @property
    def is_global(self) -> bool:
        """The masks to read are one global mask: one was found, and no pair."""
        return self.global_mask is not None and not self.is_pair

    @property
    def found_any(self) -> bool:
        return any(path is not None for path in (self.arteriole, self.venule, self.global_mask))

    def settings(self) -> dict[str, Path | None]:
        """The mask-path settings to fill from what was found.

        A pair fills both rows; a global mask the arteriole/global row and
        empties the venule row, which is what makes the run read it as a
        global mask (`vessel_mask_role_is_global`). A lone arteriole or venule
        mask fills its own row and leaves the other as it is."""
        names = VESSEL_MASK_SETTINGS[self.size]
        arteriole_row, venule_row = names["arteriole_mask_path"], names["venule_mask_path"]
        if self.is_pair:
            return {arteriole_row: self.arteriole, venule_row: self.venule}
        if self.global_mask is not None:
            return {arteriole_row: self.global_mask, venule_row: None}
        if self.arteriole is not None:
            return {arteriole_row: self.arteriole}
        if self.venule is not None:
            return {venule_row: self.venule}
        return {}


def _remove_last(words: list[str], counts: Mapping[str, int]) -> list[str]:
    """*words* without the last ``counts[w]`` occurrences of each word ``w``."""
    left = dict(counts)
    kept: list[str] = []
    for word in reversed(words):
        if left.get(word, 0) > 0:
            left[word] -= 1
            continue
        kept.append(word)
    return kept[::-1]


def _identity(words: Iterable[str]) -> list[str]:
    return [word for word in words if word not in _UNIDENTIFYING_WORDS]


def _common_prefix(first: list[str], second: list[str]) -> int:
    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


class MaskName(NamedTuple):
    """Which mask a file is (:func:`read_mask_name`)."""

    size: str
    #: ``"arteriole"``, ``"venule"`` or ``"global"``.
    kind: str
    #: ``(words shared from the start, words shared, same file format)`` with
    #: the image's name, larger nearer.
    rank: tuple[int, int, int]
    #: False for a type with no size, read as large.
    size_named: bool = True


def read_mask_name(candidate: Path | str, reference: Path | str) -> MaskName | None:
    """Which mask *candidate* is, beside the image *reference*, and how close;
    None when its name is no mask of this image's."""
    candidate, reference = Path(candidate), Path(reference)
    words = file_name_words(candidate.stem)
    reference_words = file_name_words(reference.stem)
    # What the full reading below would refuse anyway, read cheaply first: a
    # batch reads every file against every image. A name with no role word
    # adds none, and every word saying which image it is must be the image's.
    word_set = set(words)
    if not word_set & _ROLE_WORDS:
        return None
    if not word_set - _ROLE_WORDS - _UNIDENTIFYING_WORDS <= set(reference_words):
        return None
    added = Counter(words) - Counter(reference_words)
    if added.keys() & _NOT_A_MASK_WORDS:
        return None
    sizes = [size for size, terms in SIZE_WORDS.items() if added.keys() & terms]
    kinds = [kind for kind, terms in TYPE_WORDS.items() if added.keys() & terms]
    if len(sizes) > 1 or not (sizes or kinds):
        return None
    size_named = bool(sizes)
    size = sizes[0] if size_named else "large"
    kind = kinds[0] if len(kinds) == 1 else "global"

    role_words = {word: count for word, count in added.items() if word in _ROLE_WORDS}
    identity = _identity(_remove_last(words, role_words))
    reference_identity = _identity(reference_words)
    if Counter(identity) - Counter(reference_identity):
        return None
    same_format = int(
        _FORMAT.get(candidate.suffix.lower()) == _FORMAT.get(reference.suffix.lower())
    )
    rank = (_common_prefix(identity, reference_identity), len(identity), same_format)
    return MaskName(size, kind, rank, size_named)


@lru_cache(maxsize=1 << 16)
def _mask_name_words(path: Path) -> tuple[bool, frozenset[str]]:
    """Whether *path*'s name has a role word, and the words in it that say
    which image it is: :func:`read_mask_name`'s first refusals, kept per file
    so a batch reading every file against every image does not parse each
    name again."""
    words = set(_stem_words(path.stem))
    return bool(words & _ROLE_WORDS), frozenset(words - _ROLE_WORDS - _UNIDENTIFYING_WORDS)


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


@lru_cache(maxsize=1024)
def _resolved_folder(folder: Path) -> Path:
    return _resolved(folder)


def file_key(path: Path | str) -> Path:
    """*path* with its folder resolved: what tells two spellings of one file
    apart, one resolve per folder rather than per file."""
    path = Path(path)
    return _resolved_folder(path.parent) / path.name


def _same_file(first: Path, second: Path) -> bool:
    return _resolved(first) == _resolved(second)


def files_in(folder: Path, suffixes: Iterable[str]) -> list[Path]:
    """The files in *folder* with one of *suffixes*, in name order; none for a
    folder that cannot be read. One directory scan, no stat per file."""
    suffixes = tuple(suffixes)
    try:
        with os.scandir(folder) as entries:
            names = [
                entry.name
                for entry in entries
                if os.path.splitext(entry.name)[1].lower() in suffixes and entry.is_file()
            ]
    except OSError:
        return []
    return [folder / name for name in sorted(names)]


def list_mask_files(folders: Iterable[Path | str], exclude: Iterable[Path | str] = ()) -> list[Path]:
    """The mask files (:data:`MASK_SUFFIXES`) in *folders*, each folder once,
    in name order per folder, without the files in *exclude*."""
    excluded = {file_key(path) for path in exclude}
    seen: set[Path] = set()
    files: list[Path] = []
    for folder in folders:
        if folder in (None, ""):
            continue
        folder = Path(folder)
        key = _resolved(folder)
        if key in seen:
            continue
        seen.add(key)
        files.extend(
            candidate
            for candidate in files_in(folder, MASK_SUFFIXES)
            if key / candidate.name not in excluded
        )
    return files


def find_vessel_masks(
    references: Iterable[Path | str | None],
    sizes: Iterable[str] = tuple(SIZE_WORDS),
    folders: Iterable[Path | str] | None = None,
    exclude: Iterable[Path | str] = (),
) -> dict[str, FoundVesselMasks]:
    """The vessel masks beside the images *references*, for each of *sizes*.

    *references* are the run's input images (its segmented image and raw
    image); unset ones are skipped. Every mask file (:data:`MASK_SUFFIXES`) in
    their folders -- or in *folders* instead, when given (a batch run's mask
    folders) -- is read against each of them (:func:`read_mask_name`), at
    its nearest; files in *exclude* (a batch's other images) are never masks.
    The large masks come from the files naming their size, or, with none,
    from those naming only a vessel type. Returns one
    :class:`FoundVesselMasks` per size asked for, empty when nothing was found."""
    sizes = tuple(sizes)
    for size in sizes:
        if size not in SIZE_WORDS:
            raise ValueError(f"size must be 'large' or 'small', got {size!r}.")
    images = [Path(reference) for reference in references if reference not in (None, "")]
    if folders is None:
        folders = [image.parent for image in images]
    return masks_among(list_mask_files(folders, exclude=[*images, *exclude]), images, sizes)


def masks_among(
    candidates: Iterable[Path],
    images: Iterable[Path | str],
    sizes: Iterable[str] = tuple(SIZE_WORDS),
) -> dict[str, FoundVesselMasks]:
    """:func:`find_vessel_masks` over files already listed (:func:`list_mask_files`):
    each of *candidates* read against each of *images*, at its nearest."""
    sizes = tuple(sizes)
    for size in sizes:
        if size not in SIZE_WORDS:
            raise ValueError(f"size must be 'large' or 'small', got {size!r}.")
    images = [Path(image) for image in images]
    image_words = [set(_stem_words(image.stem)) for image in images]
    pools: dict[tuple[str, str, bool], list[tuple[tuple[int, int, int], Path]]] = {}
    for candidate in candidates:
        role, identifying = _mask_name_words(candidate)
        if not role:
            continue
        readings = [
            reading
            for reading in (
                read_mask_name(candidate, image)
                for image, held in zip(images, image_words)
                if identifying <= held
            )
            if reading is not None
        ]
        if not readings:
            continue
        name = max(readings, key=lambda reading: (reading.size_named, reading.rank))
        if name.size in sizes:
            pools.setdefault((name.size, name.kind, name.size_named), []).append(
                (name.rank, candidate)
            )

    found: dict[str, FoundVesselMasks] = {}
    for size in sizes:
        size_named = any(named for pool_size, _kind, named in pools if pool_size == size)
        chosen: dict[str, Path | None] = {}
        ambiguous: dict[str, tuple[Path, ...]] = {}
        for kind in ("arteriole", "venule", "global"):
            pool = sorted(
                pools.get((size, kind, size_named), []), key=lambda item: item[0], reverse=True
            )
            if not pool:
                continue
            best = [path for rank, path in pool if rank == pool[0][0]]
            if len(best) == 1:
                chosen[kind] = best[0]
            else:
                ambiguous[kind] = tuple(best)
        found[size] = FoundVesselMasks(
            size=size,
            arteriole=chosen.get("arteriole"),
            venule=chosen.get("venule"),
            global_mask=chosen.get("global"),
            ambiguous=ambiguous,
            size_named=size_named or not (chosen or ambiguous),
        )
    return found
