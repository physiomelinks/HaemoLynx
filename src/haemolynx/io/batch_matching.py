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

**A batch of raw images** (``BatchFolders(from_raw=True)``: ilastik segments
each image) is listed from the raw folder instead (:func:`batch_inputs` with
``raw=True``: every image file there that is no segmentation, probability map
or mask of another), and its masks are read against the raw images' names. Each
image's stem, and so its folder, is that of the segmentation ilastik will make
(``{raw stem}_segmented``, :func:`~haemolynx.io.ilastik.ilastik_segmented_stem`).

**The other files a run reads beside its image** -- a pericyte, cell or 3D
distance vessel mask, a tissue image, a mask of capillary blocks
(:data:`FILE_ROLES`) -- each come from a folder of their own, by the rule a
mask folder of its own follows (:func:`file_owners`): a file is the image's
whose name it shares the most words with from the start.

Nothing here reads an image: names in, matches out.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence

from .ilastik import ilastik_segmented_stem
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


class FileRole(NamedTuple):
    """A file a run reads beside its image, matched per image from a folder
    of its own."""

    #: Its :class:`BatchFolders` field and role key.
    key: str
    #: The run setting it fills.
    setting: str
    #: What it is, for messages.
    label: str
    #: What the run reads with none ("": an image with none is a problem).
    fallback: str = ""
    #: Other settings naming the same file, filled with it too.
    also: tuple[str, ...] = ()

    @property
    def settings(self) -> tuple[str, ...]:
        """Every run setting the file fills."""
        return (self.setting, *self.also)


#: The other files a run reads beside its image, a folder each.
FILE_ROLES = (
    # The perturbations' pericyte mask and the Pericyte measurements box's.
    FileRole(
        "pericyte", "pericyte_mask_path", "pericyte mask",
        also=("pericyte_measurement_mask_path",),
    ),
    FileRole("cell", "cell_mask_path", "cell mask"),
    FileRole(
        "vessel_3d", "measurement_3d_vessel_mask_path", "vessel mask for the 3D distances",
        "the run's segmented image",
    ),
    FileRole("tissue", "tissue_raw_tiff_path", "tissue image", "the raw image"),
    FileRole("capillary_block", "capillary_block_mask_path", "mask of blocks"),
)
FILE_ROLE_KEYS = tuple(role.key for role in FILE_ROLES)


def file_role(key: str) -> FileRole:
    """The :data:`FILE_ROLES` entry *key* names."""
    for role in FILE_ROLES:
        if role.key == key:
            return role
    raise ValueError(f"Unknown batch file role {key!r}.")


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


def batch_inputs(folder: Path | str, *, raw: bool = False) -> tuple[Path, ...]:
    """The segmented images in *folder* a batch runs, in name order: every
    image file that is not a probability map, a mask of another file there or
    the raw image of another file there. With *raw*, the raw images ilastik
    is to segment: every image file that is not a probability map, a
    segmentation (its name says so) or a mask of another file there."""
    files = _image_files(Path(folder))
    words = [file_name_words(path.stem) for path in files]
    index = _WordIndex(words)
    inputs: list[Path] = []
    for position, candidate in enumerate(files):
        names = words[position]
        word_set = set(names)
        if word_set & PROBABILITY_WORDS:
            continue
        if (word_set & SEGMENTATION_WORDS) if raw else _says_raw(names):
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
        if not raw and any(
            _raw_of(candidate, files[other])
            for other in index.holding(required)
            if other != position
        ):
            continue
        inputs.append(candidate)
    return tuple(inputs)


#: The folders a batch reads its images, raw images and vessel masks from.
IMAGE_FOLDER_ROLES = ("segmented", "raw", "large", "small")


@dataclass(frozen=True)
class BatchFolders:
    """A batch run's folders. A blank raw, large or small folder is the
    folder the images are listed from: the segmented folder, or with
    *from_raw* (ilastik segments each raw image) the raw folder, and the
    segmented folder is not read. A blank folder of one of the other files
    (:data:`FILE_ROLES`) is none: the run's own setting is read."""

    segmented: Path | None = None
    raw: Path | None = None
    large: Path | None = None
    small: Path | None = None
    pericyte: Path | None = None
    cell: Path | None = None
    vessel_3d: Path | None = None
    tissue: Path | None = None
    capillary_block: Path | None = None
    #: The images are the raw folder's, each segmented by ilastik.
    from_raw: bool = False

    def __post_init__(self) -> None:
        for name in IMAGE_FOLDER_ROLES + FILE_ROLE_KEYS:
            value = getattr(self, name)
            if value in (None, ""):
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(self, "from_raw", bool(self.from_raw))
        if self.from_raw and self.raw is None:
            raise ValueError(
                "A batch that ilastik segments needs a folder of raw images: its images are those."
            )
        if not self.from_raw and self.segmented is None:
            raise ValueError("A batch run needs a folder of segmented images.")

    @property
    def images(self) -> Path:
        """The folder the batch's images are listed from."""
        return self.raw if self.from_raw else self.segmented

    def folder(self, role: str) -> Path:
        """The folder *role* (``"segmented"``, ``"raw"``, ``"large"``,
        ``"small"``) is read from."""
        if role not in IMAGE_FOLDER_ROLES:
            raise ValueError(f"Unknown batch folder {role!r}.")
        return getattr(self, role) or self.images

    def file_folder(self, key: str) -> Path | None:
        """The folder the other file *key* (:data:`FILE_ROLE_KEYS`) is matched
        from, or None when none was given."""
        return getattr(self, file_role(key).key)

    @property
    def file_keys(self) -> tuple[str, ...]:
        """The other files given a folder."""
        return tuple(key for key in FILE_ROLE_KEYS if getattr(self, key) is not None)

    def own_folder(self, size: str) -> Path | None:
        """The mask folder *size* was given apart from the images, or None
        when its masks are read from the images' folder."""
        folder = getattr(self, size)
        if folder is None or _same_file(folder, self.images):
            return None
        return folder


def _names(paths: Iterable[Path]) -> str:
    return ", ".join(path.name for path in paths)


@dataclass(frozen=True)
class BatchMatch:
    """One image of a batch, and what was matched to it. In a batch of raw
    images (``folders.from_raw``) *segmented* is None until ilastik makes it,
    and the image is its *raw* image."""

    segmented: Path | None
    folders: BatchFolders
    raw: Path | None = None
    #: The raw images that tied, none chosen (empty once one is chosen).
    raw_ties: tuple[Path, ...] = ()
    #: What was found per mask size searched (the roles the run uses).
    masks: Mapping[str, FoundVesselMasks] = field(default_factory=dict)
    #: Other images of the batch with this one's stem: they would share a folder.
    same_stem: tuple[Path, ...] = ()
    #: The other files matched (:data:`FILE_ROLES`), by role key: the file,
    #: or None when the role's folder holds none for this image.
    files: Mapping[str, Path | None] = field(default_factory=dict)
    #: The other files that tied, by role key, none chosen.
    file_ties: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)

    @property
    def image(self) -> Path:
        """The file the batch lists: the segmented image, or the raw image
        ilastik segments."""
        return self.segmented if self.segmented is not None else self.raw

    @property
    def stem(self) -> str:
        """The stem the image's run names its outputs and folder with: the
        segmented image's, or that of the segmentation ilastik makes."""
        if self.segmented is not None:
            return self.segmented.stem
        return ilastik_segmented_stem(self.raw)

    def ties(self, role: str) -> tuple[Path, ...]:
        """The files tied for *role* (:data:`RAW_ROLE`, :func:`mask_role` or
        one of :data:`FILE_ROLE_KEYS`)."""
        if role == RAW_ROLE:
            return self.raw_ties
        if role in FILE_ROLE_KEYS:
            return tuple(self.file_ties.get(role, ()))
        size, _, kind = role.partition("_")
        found = self.masks.get(size)
        return tuple(found.ambiguous.get(kind, ())) if found is not None else ()

    @property
    def tied_roles(self) -> tuple[str, ...]:
        roles = [RAW_ROLE] if self.raw_ties else []
        for size, found in self.masks.items():
            roles.extend(mask_role(size, kind) for kind in MASK_KINDS if kind in found.ambiguous)
        roles.extend(key for key in FILE_ROLE_KEYS if self.file_ties.get(key))
        return tuple(roles)

    def choose(self, role: str, path: Path | str) -> "BatchMatch":
        """This match with *path*, one of the files tied for *role*, chosen."""
        path = Path(path)
        tied = self.ties(role)
        if not any(_same_file(path, other) for other in tied):
            raise ValueError(f"{path.name} is not one of the files tied for {role}.")
        if role == RAW_ROLE:
            return replace(self, raw=path, raw_ties=())._not_another_file(path)
        if role in FILE_ROLE_KEYS:
            ties = {key: value for key, value in self.file_ties.items() if key != role}
            return replace(self, files={**self.files, role: path}, file_ties=ties)
        size, _, kind = role.partition("_")
        found = self.masks[size]
        ambiguous = {key: value for key, value in found.ambiguous.items() if key != kind}
        chosen = replace(
            found,
            ambiguous=ambiguous,
            **{"global_mask" if kind == "global" else kind: path},
        )
        return replace(self, masks={**self.masks, size: chosen})._not_another_file(path)

    def _not_another_file(self, path: Path) -> "BatchMatch":
        """This match with *path*, now its raw image or a mask, taken out of
        the other files' ties: the one file left in a tie is chosen."""
        files, file_ties = dict(self.files), dict(self.file_ties)
        for key, tied in self.file_ties.items():
            left = tuple(other for other in tied if not _same_file(other, path))
            if len(left) > 1:
                file_ties[key] = left
                continue
            del file_ties[key]
            files[key] = left[0] if left else None
        return replace(self, files=files, file_ties=file_ties)

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
        for key in self.files:
            role = file_role(key)
            tied = self.file_ties.get(key)
            if tied:
                problems.append(f"Several files could be the {role.label} ({_names(tied)}): choose one.")
            elif self.files[key] is None and not role.fallback:
                problems.append(f"No {role.label} found in {self.folders.file_folder(key)}.")
        return tuple(problems)

    @property
    def notes(self) -> tuple[str, ...]:
        """What the user should know about the match, which stops nothing."""
        notes: list[str] = []
        if self.raw is None and not self.raw_ties:
            notes.append(f"No raw image found in {self.folders.folder('raw')}.")
        for key, path in self.files.items():
            role = file_role(key)
            if path is None and role.fallback and not self.file_ties.get(key):
                notes.append(
                    f"No {role.label} found in {self.folders.file_folder(key)}: "
                    f"{role.fallback} is read instead."
                )
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
        """The run settings this match sets: the segmented and raw images (no
        segmented image while ilastik is to make it), both mask rows of every
        size searched, and each other file given a folder -- a row nothing was
        found for is emptied, so no file the panel held for a single run
        carries over."""
        values: dict[str, Path | None] = {
            "input_path": self.segmented,
            "raw_image_path": self.raw,
        }
        for size, found in self.masks.items():
            names = VESSEL_MASK_SETTINGS[size]
            values[names["arteriole_mask_path"]] = None
            values[names["venule_mask_path"]] = None
            values.update(found.settings())
        for key, path in self.files.items():
            for setting in file_role(key).settings:
                values[setting] = path
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


class _Owners:
    """Which batch image a file in a folder of its own belongs to: the image
    whose name shares the most words with it from the start, then the most
    words, then the fewest words of its own the file lacks, when one image
    does (``E14_pericytes`` is ``E14``'s, not ``E14_2``'s)."""

    def __init__(self, inputs: Sequence[Path]) -> None:
        self._inputs = list(inputs)
        self._identities = [_plain_identity(file_name_words(image.stem)) for image in inputs]

    def owner(self, words: Sequence[str]) -> Path | None:
        identity = _plain_identity(words)
        ranks = []
        for image, image_identity in zip(self._inputs, self._identities):
            shared_start = _common_prefix(identity, image_identity)
            if shared_start:
                shared = sum((Counter(identity) & Counter(image_identity)).values())
                ranks.append(((shared_start, shared, shared - len(image_identity)), image))
        if not ranks:
            return None
        best = max(rank for rank, _image in ranks)
        winners = [image for rank, image in ranks if rank == best]
        return winners[0] if len(winners) == 1 else None


def file_owners(candidates: Sequence[Path], inputs: Sequence[Path]) -> dict[Path, list[Path]]:
    """Each file in a folder of its own (one of :data:`FILE_ROLES`'), given
    to the batch image it belongs to, as ``image -> [file, ...]``: the image
    whose name it shares the most words with from the start (:class:`_Owners`).
    Probability maps belong to no image."""
    owners = _Owners(inputs)
    owned: dict[Path, list[Path]] = {}
    for candidate in candidates:
        words = file_name_words(candidate.stem)
        if set(words) & PROBABILITY_WORDS:
            continue
        owner = owners.owner(words)
        if owner is not None:
            owned.setdefault(owner, []).append(candidate)
    return owned


def mask_owners(
    candidates: Sequence[Path], inputs: Sequence[Path], size: str
) -> dict[Path, list[tuple[str, Path]]]:
    """Each mask file in *size*'s own folder, given to the batch image it
    belongs to, as ``image -> [(kind, file), ...]``.

    The folder says the size, so a file need not: ilastik's
    ``{stem}_Simple Segmentation_intermediate.h5`` in a small-vessel folder is a
    small mask. A file is the mask of the image whose name shares the most
    words with it from the start (:class:`_Owners`); a file naming the other
    size is not this folder's. Its kind is the vessel type it names beyond
    that image's name, or global with none (or both)."""
    other_sizes = frozenset().union(
        *(words for name, words in SIZE_WORDS.items() if name != size)
    )
    own_size = SIZE_WORDS[size]
    owners: dict[Path, list[tuple[str, Path]]] = {}
    ownership = _Owners(inputs)
    for candidate in candidates:
        words = file_name_words(candidate.stem)
        if set(words) & (PROBABILITY_WORDS | RAW_WORDS):
            continue
        owner = ownership.owner(words)
        if owner is None:
            continue
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
    roles: Iterable[str] = (),
) -> tuple[BatchMatch, ...]:
    """Each image in *folders*' segmented folder (its raw folder, for a batch
    ilastik segments), matched to its raw image, to its masks of each of
    *sizes* (the vessel-mask roles the run uses: ``"large"``, ``"small"``) and
    to each other file of *roles* (:data:`FILE_ROLE_KEYS`: those the run
    reads) given a folder. Raises ValueError when the images' folder is not a
    folder."""
    sizes = tuple(sizes)
    for size in sizes:
        if size not in SIZE_WORDS:
            raise ValueError(f"size must be 'large' or 'small', got {size!r}.")
    roles = tuple(key for key in FILE_ROLE_KEYS if key in set(roles))
    if not folders.images.is_dir():
        kind = "raw" if folders.from_raw else "segmented"
        raise ValueError(f"The {kind} images folder {folders.images} is not a folder.")
    inputs = batch_inputs(folders.images, raw=folders.from_raw)
    input_files = {file_key(image) for image in inputs}
    raw_candidates = (
        []
        if folders.from_raw
        else [
            candidate
            for candidate in _image_files(folders.folder("raw"))
            if file_key(candidate) not in input_files
        ]
    )
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
    # Each other file's folder, the batch's images left out of it, its files
    # given to the images they share their names with.
    file_owned = {
        key: file_owners(
            [
                candidate
                for candidate in _image_files(folders.file_folder(key))
                if file_key(candidate) not in input_files
            ],
            inputs,
        )
        for key in roles
        if folders.file_folder(key) is not None
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
        raw, raw_ties = (image, ()) if folders.from_raw else _nearest(readings)
        references = [image] if raw is None or raw is image else [image, raw]
        masks = {
            size: (
                masks_owned_by(owned[size], image, size)
                if size in owned
                else masks_among(mask_files[size], references, [size])[size]
            )
            for size in sizes
        }
        # A file already matched to the image (its raw image, its masks) is
        # not one of its other files, in a folder shared with them.
        matched = {file_key(path) for path in (raw, *_mask_files(masks)) if path is not None}
        files: dict[str, Path | None] = {}
        file_ties: dict[str, tuple[Path, ...]] = {}
        for key, owners in file_owned.items():
            own = [path for path in owners.get(image, ()) if file_key(path) not in matched]
            files[key] = own[0] if len(own) == 1 else None
            if len(own) > 1:
                file_ties[key] = tuple(own)
        others = [other for other in inputs if other is not image]
        same_stem = (
            tuple(other for other in others if other.stem.lower() == image.stem.lower())
            if stems[image.stem.lower()] > 1
            else ()
        )
        matches.append(
            BatchMatch(
                segmented=None if folders.from_raw else image,
                folders=folders,
                raw=raw,
                raw_ties=raw_ties,
                masks=masks,
                same_stem=same_stem,
                files=files,
                file_ties=file_ties,
            )
        )
    return tuple(matches)


def _mask_files(masks: Mapping[str, FoundVesselMasks]) -> list[Path]:
    """Every file *masks* chose or left tied."""
    paths: list[Path] = []
    for found in masks.values():
        paths.extend(path for path in (found.arteriole, found.venule, found.global_mask) if path)
        for tied in found.ambiguous.values():
            paths.extend(tied)
    return paths


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
    "FILE_ROLES",
    "FILE_ROLE_KEYS",
    "FileRole",
    "IMAGE_FOLDER_ROLES",
    "IMAGE_SUFFIXES",
    "MASK_KINDS",
    "RAW_ROLE",
    "RawName",
    "batch_inputs",
    "choose_alike",
    "file_owners",
    "file_role",
    "is_raw_of",
    "mask_role",
    "match_batch",
    "read_raw_name",
]
