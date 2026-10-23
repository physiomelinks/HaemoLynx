"""Match a batch run's images to their raw images and vessel masks, by file name.

A batch run (``PLAN_batch_run.md``) is given up to four folders: the
segmented images, the raw images, the large vessel masks and the small vessel
masks. Each segmented image is one run, and its raw image and masks are found
by name, through the rule the panel already uses to find masks beside an image
(:mod:`haemolynx.io.vessel_mask_discovery`). A folder left blank is the
segmented images' folder, so a user keeping everything in one folder gives
only that.

**The batch's images** (:func:`batch_inputs`) are the image files in the
segmented folder that are neither a mask of another file there
(:func:`~haemolynx.io.vessel_mask_discovery.read_mask_name`), nor a
probability map, nor the raw image of another file there (:func:`is_raw_of`):
``E14_MCA.tif`` beside ``E14_MCA_Simple Segmentation.tiff``.

**A raw image** (:func:`read_raw_name`) is read against a segmented image as a
mask is, with no role words: what its name says, apart from the words that say
nothing about which image it is (``raw``, ``segmented``, ``Simple
Segmentation`` ...) and channel names (ImageJ's ``C1-``, ``_C2``, ``ch1``), must
all be in the segmented image's name -- a segmented image is named after the
raw image it came from, with words added. The nearest wins (words shared from
the start, words shared, the segmented image's channel, the same file format);
a tie is left unchosen and named, for the user to choose
(:meth:`BatchMatch.choose`, :func:`choose_alike`).

**The masks**, read from the segmented folder, are found by
:func:`~haemolynx.io.vessel_mask_discovery.find_vessel_masks`'s rule on the
segmented image and its raw image, the batch's other images never taken for
masks. **A mask folder of its own** says what its files are, so their names
need not (:func:`mask_owners`): each file is the mask of the batch image whose
name it shares the most words with from the start, and a global mask unless
it names a vessel type -- ilastik's ``..._Simple Segmentation_intermediate.h5``
in a small-vessel folder is that image's small mask.

Nothing here reads an image: names in, matches out.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence

from .vessel_masks import VESSEL_MASK_SETTINGS
from .vessel_mask_discovery import (
    MASK_SUFFIXES,
    SIZE_WORDS,
    TYPE_WORDS,
    FoundVesselMasks,
    _FORMAT,
    _common_prefix,
    _UNIDENTIFYING_WORDS,
    _identity,
    _same_file,
    file_key,
    file_name_words,
    files_in,
    list_mask_files,
    masks_among,
    read_mask_name,
)

#: The file types a batch image (segmented or raw) is read from.
IMAGE_SUFFIXES = MASK_SUFFIXES

#: Words saying a file is a segmentation, of an image named without them.
SEGMENTATION_WORDS = frozenset(
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
    }
)

#: Words saying a file is a raw image.
RAW_WORDS = frozenset({"raw", "unsegmented"})

#: Words saying a file is an ilastik probability map, never an image to run.
PROBABILITY_WORDS = frozenset({"probabilities", "probability"})

#: Words that, followed by a number, name a channel: ImageJ's ``C1-``, ``ch2``.
CHANNEL_WORDS = frozenset({"c", "ch", "channel"})

_ROLE_WORDS = frozenset().union(*SIZE_WORDS.values(), *TYPE_WORDS.values())

#: The roles a match can leave a tie in: the raw image, and each mask kind.
RAW_ROLE = "raw"
MASK_KINDS = ("arteriole", "venule", "global")


def mask_role(size: str, kind: str) -> str:
    """The role key of a *size* mask of *kind*: ``"large_arteriole"`` ..."""
    return f"{size}_{kind}"


def _split_channels(words: Sequence[str]) -> tuple[list[str], list[str]]:
    """*words* without channel names (a channel word and the number after it),
    and the channel names taken out, each as ``"c1"``."""
    kept: list[str] = []
    channels: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if (
            word in CHANNEL_WORDS
            and index + 1 < len(words)
            and words[index + 1].isdigit()
        ):
            channels.append(f"c{int(words[index + 1])}")
            index += 2
            continue
        kept.append(word)
        index += 1
    return kept, channels


def _image_files(folder: Path) -> list[Path]:
    return files_in(folder, IMAGE_SUFFIXES)


class RawName(NamedTuple):
    """How close a raw image's name is to its segmented image's (larger nearer)."""

    #: Words shared from the start, channel names left out.
    shared_start: int
    #: The raw name's words, channel names left out.
    shared: int
    #: 1 when the raw image's channel names are all in the segmented image's.
    same_channel: int
    #: 1 for the segmented image's file format.
    same_format: int


def read_raw_name(candidate: Path | str, segmented: Path | str) -> RawName | None:
    """Whether *candidate* can be the raw image *segmented* was made from, and
    how close; None when its name says it is another image's, or a
    segmentation, mask or probability map."""
    candidate, segmented = Path(candidate), Path(segmented)
    words = file_name_words(candidate.stem)
    word_set = set(words)
    if word_set & (PROBABILITY_WORDS | SEGMENTATION_WORDS):
        return None
    segmented_words = file_name_words(segmented.stem)
    plain, channels = _split_channels(words)
    identity = _identity(plain)
    if not set(identity) <= set(segmented_words):
        return None
    if (word_set - set(segmented_words)) & _ROLE_WORDS:
        return None
    segmented_plain, segmented_channels = _split_channels(segmented_words)
    segmented_identity = _identity(segmented_plain)
    if Counter(identity) - Counter(segmented_identity):
        return None
    return RawName(
        _common_prefix(identity, segmented_identity),
        len(identity),
        int(not Counter(channels) - Counter(segmented_channels)),
        int(_FORMAT.get(candidate.suffix.lower()) == _FORMAT.get(segmented.suffix.lower())),
    )


def _says_raw(words: Iterable[str]) -> bool:
    """A name saying ``raw`` and no segmentation word: ``E14_raw.tif``, not
    ilastik's ``E14_raw_Simple Segmentation.tiff``."""
    words = set(words)
    return bool(words & RAW_WORDS) and not words & SEGMENTATION_WORDS


def _raw_of(candidate: Path, image: Path) -> bool:
    if read_raw_name(candidate, image) is None:
        return False
    words = file_name_words(candidate.stem)
    image_adds = Counter(file_name_words(image.stem)) - Counter(words)
    return bool(image_adds.keys() & SEGMENTATION_WORDS) or _says_raw(words)


def is_raw_of(candidate: Path | str, image: Path | str) -> bool:
    """Whether *candidate*, in one folder with *image*, is *image*'s raw image
    rather than an image of its own: its name reads as *image*'s raw image
    (:func:`read_raw_name`), and *image*'s name adds a segmentation word to it
    or it says ``raw`` itself."""
    candidate, image = Path(candidate), Path(image)
    return not _same_file(candidate, image) and _raw_of(candidate, image)


def _is_mask_of(candidate: Path, image: Path) -> bool:
    """*candidate* reads as one of *image*'s masks, and not the other way round:
    two images named ``large_field`` and ``small_field`` are not each other's
    masks."""
    return read_mask_name(candidate, image) is not None and read_mask_name(image, candidate) is None


class _WordIndex:
    """Which of a folder's files hold given words in their names: a file can
    only be another's mask or raw image if the other's name holds every word
    saying which image it is, so a batch reads only those pairs."""

    def __init__(self, words: Sequence[Sequence[str]]) -> None:
        self._everyone = frozenset(range(len(words)))
        self._holding: dict[str, set[int]] = {}
        for index, names in enumerate(words):
            for word in set(names):
                self._holding.setdefault(word, set()).add(index)

    def holding(self, required: Iterable[str]) -> list[int]:
        """The files whose names hold every word in *required*, in order."""
        result: set[int] | None = None
        for word in sorted(set(required), key=lambda word: len(self._holding.get(word, ()))):
            files = self._holding.get(word, set())
            result = set(files) if result is None else result & files
            if not result:
                return []
        return sorted(self._everyone if result is None else result)


def batch_inputs(folder: Path | str) -> tuple[Path, ...]:
    """The segmented images in *folder* a batch runs, in name order: every
    image file that is not a probability map, a mask of another file there or
    the raw image of another file there."""
    files = _image_files(Path(folder))
    words = [file_name_words(path.stem) for path in files]
    index = _WordIndex(words)
    inputs: list[Path] = []
    for position, candidate in enumerate(files):
        names = words[position]
        word_set = set(names)
        if word_set & PROBABILITY_WORDS or _says_raw(names):
            continue
        if word_set & _ROLE_WORDS:
            required = word_set - _ROLE_WORDS - _UNIDENTIFYING_WORDS
            if any(
                _is_mask_of(candidate, files[other])
                for other in index.holding(required)
                if other != position
            ):
                continue
        required = set(_identity(_split_channels(names)[0]))
        if any(
            _raw_of(candidate, files[other])
            for other in index.holding(required)
            if other != position
        ):
            continue
        inputs.append(candidate)
    return tuple(inputs)


@dataclass(frozen=True)
class BatchFolders:
    """A batch run's folders. A blank raw, large or small folder is the
    segmented images' folder."""

    segmented: Path
    raw: Path | None = None
    large: Path | None = None
    small: Path | None = None

    def __post_init__(self) -> None:
        for name in ("segmented", "raw", "large", "small"):
            value = getattr(self, name)
            if value in (None, ""):
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(self, name, Path(value))
        if self.segmented is None:
            raise ValueError("A batch run needs a folder of segmented images.")

    def folder(self, role: str) -> Path:
        """The folder *role* (``"segmented"``, ``"raw"``, ``"large"``,
        ``"small"``) is read from."""
        if role not in ("segmented", "raw", "large", "small"):
            raise ValueError(f"Unknown batch folder {role!r}.")
        return getattr(self, role) or self.segmented

    def own_folder(self, size: str) -> Path | None:
        """The mask folder *size* was given apart from the segmented images,
        or None when its masks are read from the segmented folder."""
        folder = getattr(self, size)
        if folder is None or _same_file(folder, self.segmented):
            return None
        return folder


def _names(paths: Iterable[Path]) -> str:
    return ", ".join(path.name for path in paths)


@dataclass(frozen=True)
class BatchMatch:
    """One segmented image of a batch, and what was matched to it."""

    segmented: Path
    folders: BatchFolders
    raw: Path | None = None
    #: The raw images that tied, none chosen (empty once one is chosen).
    raw_ties: tuple[Path, ...] = ()
    #: What was found per mask size searched (the roles the run uses).
    masks: Mapping[str, FoundVesselMasks] = field(default_factory=dict)
    #: Other images of the batch with this one's stem: they would share a folder.
    same_stem: tuple[Path, ...] = ()

    @property
    def stem(self) -> str:
        return self.segmented.stem

    def ties(self, role: str) -> tuple[Path, ...]:
        """The files tied for *role* (:data:`RAW_ROLE` or :func:`mask_role`)."""
        if role == RAW_ROLE:
            return self.raw_ties
        size, _, kind = role.partition("_")
        found = self.masks.get(size)
        return tuple(found.ambiguous.get(kind, ())) if found is not None else ()

    @property
    def tied_roles(self) -> tuple[str, ...]:
        roles = [RAW_ROLE] if self.raw_ties else []
        for size, found in self.masks.items():
            roles.extend(mask_role(size, kind) for kind in MASK_KINDS if kind in found.ambiguous)
        return tuple(roles)

    def choose(self, role: str, path: Path | str) -> "BatchMatch":
        """This match with *path*, one of the files tied for *role*, chosen."""
        path = Path(path)
        tied = self.ties(role)
        if not any(_same_file(path, other) for other in tied):
            raise ValueError(f"{path.name} is not one of the files tied for {role}.")
        if role == RAW_ROLE:
            return replace(self, raw=path, raw_ties=())
        size, _, kind = role.partition("_")
        found = self.masks[size]
        ambiguous = {key: value for key, value in found.ambiguous.items() if key != kind}
        chosen = replace(
            found,
            ambiguous=ambiguous,
            **{"global_mask" if kind == "global" else kind: path},
        )
        return replace(self, masks={**self.masks, size: chosen})

    @property
    def problems(self) -> tuple[str, ...]:
        """What stops this image running as matched."""
        problems: list[str] = []
        if self.same_stem:
            problems.append(
                f"{_names(self.same_stem)} has the same name, and the two would "
                "share an output folder."
            )
        if self.raw_ties:
            problems.append(f"Several files could be the raw image ({_names(self.raw_ties)}): choose one.")
        for size, found in self.masks.items():
            folder = self.folders.folder(size)
            for kind, tied in found.ambiguous.items():
                problems.append(
                    f"Several files could be the {size} {kind} mask ({_names(tied)}): choose one."
                )
            if found.ambiguous:
                continue
            if not found.found_any:
                problems.append(f"No {size} vessel masks found in {folder}.")
            elif found.venule is not None and found.arteriole is None and found.global_mask is None:
                problems.append(
                    f"Found a {size} venule mask, {found.venule.name}, but no {size} "
                    "arteriole mask to go with it."
                )
        return tuple(problems)

    @property
    def notes(self) -> tuple[str, ...]:
        """What the user should know about the match, which stops nothing."""
        notes: list[str] = []
        if self.raw is None and not self.raw_ties:
            notes.append(f"No raw image found in {self.folders.folder('raw')}.")
        for size, found in self.masks.items():
            if found.is_global:
                notes.append(
                    f"{found.global_mask.name} names no vessel type: read as a global "
                    f"mask of every {size} vessel."
                )
            elif found.arteriole is not None and found.venule is None and not found.ambiguous:
                notes.append(
                    f"No {size} venule mask: {found.arteriole.name} is read as a global "
                    f"mask of every {size} vessel."
                )
            if not found.size_named and found.found_any:
                notes.append(
                    "No file says 'large' or 'small', so the masks naming only a "
                    "vessel type are read as the large ones."
                )
        return tuple(notes)

    @property
    def ok(self) -> bool:
        return not self.problems

    def settings(self) -> dict[str, Path | None]:
        """The run settings this match sets: the segmented and raw images, and
        both mask rows of every size searched -- a row nothing was found for is
        emptied, so no file the panel held for a single run carries over."""
        values: dict[str, Path | None] = {
            "input_path": self.segmented,
            "raw_image_path": self.raw,
        }
        for size, found in self.masks.items():
            names = VESSEL_MASK_SETTINGS[size]
            values[names["arteriole_mask_path"]] = None
            values[names["venule_mask_path"]] = None
            values.update(found.settings())
        return values


def _nearest(
    readings: Sequence[tuple[RawName, Path]],
) -> tuple[Path | None, tuple[Path, ...]]:
    if not readings:
        return None, ()
    best_rank = max(rank for rank, _path in readings)
    best = [path for rank, path in readings if rank == best_rank]
    return (best[0], ()) if len(best) == 1 else (None, tuple(best))


def _plain_identity(words: Sequence[str]) -> list[str]:
    """A name's words saying which image it is: no role, channel or
    unidentifying words."""
    plain, _channels = _split_channels([word for word in words if word not in _ROLE_WORDS])
    return _identity(plain)


def mask_owners(
    candidates: Sequence[Path], inputs: Sequence[Path], size: str
) -> dict[Path, list[tuple[str, Path]]]:
    """Each mask file in *size*'s own folder, given to the batch image it
    belongs to, as ``image -> [(kind, file), ...]``.

    The folder says the size, so a file need not: ilastik's
    ``{stem}_Simple Segmentation_intermediate.h5`` in a small-vessel folder is a
    small mask. A file is the mask of the image whose name shares the most
    words with it from the start (then the most words), when one image does;
    a file naming the other size is not this folder's. Its kind is the vessel
    type it names beyond that image's name, or global with none (or both)."""
    other_sizes = frozenset().union(
        *(words for name, words in SIZE_WORDS.items() if name != size)
    )
    own_size = SIZE_WORDS[size]
    image_identities = [_plain_identity(file_name_words(image.stem)) for image in inputs]
    owners: dict[Path, list[tuple[str, Path]]] = {}
    for candidate in candidates:
        words = file_name_words(candidate.stem)
        if set(words) & (PROBABILITY_WORDS | RAW_WORDS):
            continue
        identity = _plain_identity(words)
        ranks = []
        for image, image_identity in zip(inputs, image_identities):
            shared = _common_prefix(identity, image_identity)
            if shared:
                ranks.append(
                    ((shared, sum((Counter(identity) & Counter(image_identity)).values())), image)
                )
        if not ranks:
            continue
        best = max(rank for rank, _image in ranks)
        winners = [image for rank, image in ranks if rank == best]
        if len(winners) != 1:
            continue
        owner = winners[0]
        added = set(words) - set(file_name_words(owner.stem))
        if added & other_sizes and not added & own_size:
            continue
        kinds = [kind for kind, terms in TYPE_WORDS.items() if added & terms]
        kind = kinds[0] if len(kinds) == 1 else "global"
        owners.setdefault(owner, []).append((kind, candidate))
    return owners


def masks_owned_by(
    owners: Mapping[Path, Sequence[tuple[str, Path]]], image: Path, size: str
) -> FoundVesselMasks:
    """*image*'s *size* masks from :func:`mask_owners`; several files of one
    kind tie."""
    by_kind: dict[str, list[Path]] = {}
    for kind, path in owners.get(image, ()):
        by_kind.setdefault(kind, []).append(path)
    chosen = {kind: paths[0] for kind, paths in by_kind.items() if len(paths) == 1}
    ambiguous = {kind: tuple(paths) for kind, paths in by_kind.items() if len(paths) > 1}
    return FoundVesselMasks(
        size=size,
        arteriole=chosen.get("arteriole"),
        venule=chosen.get("venule"),
        global_mask=chosen.get("global"),
        ambiguous=ambiguous,
    )


def match_batch(
    folders: BatchFolders,
    sizes: Iterable[str] = (),
) -> tuple[BatchMatch, ...]:
    """Each segmented image in *folders*' segmented folder, matched to its raw
    image and to its masks of each of *sizes* (the vessel-mask roles the run
    uses: ``"large"``, ``"small"``). Raises ValueError when the segmented
    folder is not a folder."""
    sizes = tuple(sizes)
    for size in sizes:
        if size not in SIZE_WORDS:
            raise ValueError(f"size must be 'large' or 'small', got {size!r}.")
    if not folders.segmented.is_dir():
        raise ValueError(f"The segmented images folder {folders.segmented} is not a folder.")
    inputs = batch_inputs(folders.segmented)
    input_files = {file_key(image) for image in inputs}
    raw_candidates = [
        candidate
        for candidate in _image_files(folders.folder("raw"))
        if file_key(candidate) not in input_files
    ]
    # Each mask folder is listed once, the batch's images left out of it.
    mask_files = {size: list_mask_files([folders.folder(size)], exclude=inputs) for size in sizes}
    stems = Counter(image.stem.lower() for image in inputs)
    # A mask folder of its own says what its files are: each is the mask of
    # the image it shares its name with.
    owned = {
        size: mask_owners(mask_files[size], inputs, size)
        for size in sizes
        if folders.own_folder(size) is not None
    }

    matches: list[BatchMatch] = []
    # The words each raw candidate needs the segmented image's name to hold,
    # so only those that could match are read in full.
    raw_required = [
        frozenset(_identity(_split_channels(file_name_words(candidate.stem))[0]))
        for candidate in raw_candidates
    ]
    for image in inputs:
        held = set(file_name_words(image.stem))
        readings = [
            (rank, candidate)
            for candidate, required in zip(raw_candidates, raw_required)
            if required <= held and (rank := read_raw_name(candidate, image)) is not None
        ]
        raw, raw_ties = _nearest(readings)
        references = [image] if raw is None else [image, raw]
        masks = {
            size: (
                masks_owned_by(owned[size], image, size)
                if size in owned
                else masks_among(mask_files[size], references, [size])[size]
            )
            for size in sizes
        }
        others = [other for other in inputs if other is not image]
        same_stem = (
            tuple(other for other in others if other.stem.lower() == image.stem.lower())
            if stems[image.stem.lower()] > 1
            else ()
        )
        matches.append(
            BatchMatch(
                segmented=image,
                folders=folders,
                raw=raw,
                raw_ties=raw_ties,
                masks=masks,
                same_stem=same_stem,
            )
        )
    return tuple(matches)


def _distinguishing(path: Path, tied: Sequence[Path]) -> tuple[tuple[str, ...], str]:
    """What sets *path* apart from the other files it tied with: its words
    that are not in every one of them, and its file type."""
    shared = Counter(file_name_words(tied[0].stem))
    for other in tied[1:]:
        shared &= Counter(file_name_words(other.stem))
    own = Counter(file_name_words(path.stem)) - shared
    return tuple(sorted(own.elements())), path.suffix.lower()


def choose_alike(
    matches: Sequence[BatchMatch],
    role: str,
    example: Path | str,
    example_tied: Sequence[Path | str],
) -> tuple[BatchMatch, ...]:
    """*matches* with every tie in *role* chosen as *example* was chosen from
    *example_tied*: the file that differs from the others it tied with as
    *example* differs from its own -- choosing ``C1-E14.tif`` over
    ``C2-E14.tif`` chooses ``C1-E15.tif`` over ``C2-E15.tif``. A tie with no
    such file, or more than one, stays."""
    example = Path(example)
    key = _distinguishing(example, [Path(path) for path in example_tied])
    chosen: list[BatchMatch] = []
    for match in matches:
        tied = match.ties(role)
        if tied:
            alike = [path for path in tied if _distinguishing(path, tied) == key]
            if len(alike) == 1:
                match = match.choose(role, alike[0])
        chosen.append(match)
    return tuple(chosen)


__all__ = [
    "BatchFolders",
    "BatchMatch",
    "CHANNEL_WORDS",
    "IMAGE_SUFFIXES",
    "MASK_KINDS",
    "RAW_ROLE",
    "RawName",
    "batch_inputs",
    "choose_alike",
    "is_raw_of",
    "mask_role",
    "match_batch",
    "read_raw_name",
]
