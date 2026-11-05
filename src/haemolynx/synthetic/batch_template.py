"""A batch condition's runs as one synthetic template: the ``.haemobatch``
file (``PLAN_batch_templates.md``; Harvey, 2026-10-25).

A batch run writes one per condition, or one for the whole batch with none
(``gui.batch_run``): gzip JSON holding, for each image run of it that can be a
template, its stem, its ``.haemorun`` path and SHA-256 and its two templates
(``NetworkTemplate.to_dict``, by the statistics report and by the graph), with
the condition, the runs left out and why, and the batch's settings (its
``batch_settings.yaml`` as written). :func:`load_batch_template`
averages the members' templates of the asked source
(``synthetic.average.average_templates``); ``synthetic.load_template`` hands
a ``.haemobatch`` to it, so everything that reads a template reads one.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .average import average_templates, common_box_members
from .template import TEMPLATE_SOURCES, NetworkTemplate, TemplateError

logger = logging.getLogger(__name__)

BATCH_TEMPLATE_SUFFIX = ".haemobatch"
FORMAT = "haemolynx-batch-template"
VERSION = 1
#: The file's stem when the batch has no conditions: every image is in it.
WHOLE_BATCH = "batch"


@dataclass(frozen=True)
class BatchMember:
    stem: str
    haemorun: str
    sha256: str
    #: ``{source: NetworkTemplate.to_dict() or None}`` for each of
    #: ``template.TEMPLATE_SOURCES`` (None where that source refused the run).
    templates: dict


@dataclass(frozen=True)
class BatchTemplateFile:
    condition: Optional[str]
    members: tuple
    #: ``(stem, why)`` for each run that could not be a template.
    left_out: tuple
    created: Optional[str] = None
    haemolynx_version: Optional[str] = None
    #: The batch's ``batch_settings.yaml`` as it was written, or None.
    batch_settings: Optional[str] = None


def is_batch_template(path: Path | str) -> bool:
    return Path(path).suffix.lower() == BATCH_TEMPLATE_SUFFIX


def batch_template_filename(condition: Optional[str]) -> str:
    """``{condition}.haemobatch`` (its label made safe as a file name), or
    ``batch.haemobatch`` for the whole batch."""
    if not condition:
        return WHOLE_BATCH + BATCH_TEMPLATE_SUFFIX
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", condition.strip()).strip("._") or "condition"
    return safe + BATCH_TEMPLATE_SUFFIX


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_batch_template(
    path: Path | str,
    runs: Sequence[tuple[str, Path | str]],
    *,
    condition: Optional[str] = None,
    batch_settings: Optional[str] = None,
) -> Optional[Path]:
    """Write the ``.haemobatch`` of *runs* (``(stem, .haemorun path)`` pairs)
    to *path*: each run's templates by both sources, a run neither source
    can read left out with why, and *batch_settings* (the batch's
    ``batch_settings.yaml``) as given. None, and nothing written, when no run
    can be a template."""
    from haemolynx.gui.run_snapshot import RunSnapshotError

    from .template import load_template

    members: list[dict] = []
    left_out: list[list[str]] = []
    for stem, run in runs:
        run = Path(run)
        templates: dict[str, Any] = {}
        reasons: list[str] = []
        for source in TEMPLATE_SOURCES:
            try:
                templates[source] = load_template(run, source).to_dict()
            except (TemplateError, RunSnapshotError, OSError, ValueError) as error:
                templates[source] = None
                reasons.append(f"{source}: {error}")
        if not any(templates.values()):
            # One reason when both sources refused the run alike.
            why = "; ".join(dict.fromkeys(reason.split(": ", 1)[1] for reason in reasons))
            left_out.append([stem, why])
            logger.warning("Batch template %s: %s left out (%s)", path, stem, why)
            continue
        members.append({"stem": stem, "haemorun": str(run), "sha256": _sha256(run), "templates": templates})
    if not members:
        logger.warning("Batch template %s not written: none of its %d runs can be a template", path, len(runs))
        return None
    try:
        import haemolynx

        version = getattr(haemolynx, "__version__", None)
    except Exception:  # noqa: BLE001
        version = None
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "condition": condition,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "haemolynx_version": version,
        "members": members,
        "left_out": left_out,
        "batch_settings": batch_settings,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with gzip.open(part, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle)
    os.replace(part, target)
    logger.info(
        "Batch template %s: %d run%s%s", target, len(members), "s" if len(members) != 1 else "",
        f" of condition '{condition}'" if condition else "",
    )
    return target


def read_batch_template(path: Path | str) -> BatchTemplateFile:
    """The ``.haemobatch`` at *path*; :class:`~.template.TemplateError` for a
    file that is not one."""
    try:
        with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, EOFError) as error:
        raise TemplateError(f"{path} is not a batch template ({error}).", reason="not_a_run") from error
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise TemplateError(f"{path} is not a batch template.", reason="not_a_run")
    members = tuple(
        BatchMember(m["stem"], m["haemorun"], m.get("sha256", ""), dict(m.get("templates") or {}))
        for m in data.get("members", ())
    )
    return BatchTemplateFile(
        condition=data.get("condition"),
        members=members,
        left_out=tuple(tuple(item) for item in data.get("left_out", ())),
        created=data.get("created"),
        haemolynx_version=data.get("haemolynx_version"),
        batch_settings=data.get("batch_settings"),
    )


def batch_member_templates(path: Path | str, source: str) -> list[NetworkTemplate]:
    """The templates by *source* of the members of the ``.haemobatch`` at
    *path* that have one."""
    file = read_batch_template(path)
    return [NetworkTemplate.from_dict(m.templates[source]) for m in file.members if m.templates.get(source)]


def load_batch_template(path: Path | str, source: str) -> NetworkTemplate:
    """The average of the ``.haemobatch``'s members' templates by *source*
    (``synthetic.average``): the condition's network as one template."""
    file = read_batch_template(path)
    members = [(m.stem, NetworkTemplate.from_dict(m.templates[source])) for m in file.members if m.templates.get(source)]
    if not members:
        raise TemplateError(
            f"{path} holds no run that can be a template by its {source}. Fix: choose the other source, "
            "or run the batch again with Statistics on.",
            reason="no_statistics",
        )
    template = average_templates(
        [t for _s, t in members],
        template_path=str(path),
        source=source,
        condition=file.condition,
        member_names=[s for s, _t in members],
    )
    if file.left_out:
        notes = template.provenance.notes + tuple(f"left out of the batch template: {stem} ({why})" for stem, why in file.left_out)
        from dataclasses import replace

        template = replace(template, provenance=replace(template.provenance, notes=notes))
    return template


def batch_members_in_common_box(path: Path | str, source: str) -> list[NetworkTemplate]:
    """The members' templates scaled to the average's box: what the
    comparison reads each characteristic's spread across the condition from
    (B4b)."""
    return common_box_members(batch_member_templates(path, source))


__all__ = [
    "BATCH_TEMPLATE_SUFFIX",
    "BatchMember",
    "BatchTemplateFile",
    "WHOLE_BATCH",
    "batch_member_templates",
    "batch_members_in_common_box",
    "batch_template_filename",
    "is_batch_template",
    "load_batch_template",
    "read_batch_template",
    "write_batch_template",
]
