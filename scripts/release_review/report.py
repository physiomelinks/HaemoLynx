"""The scorecard as Markdown and JSON, and what moved since the last one."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .ladder import DOCUMENT, PASS, UNKNOWN, CheckResult
from .rubric import ASPECTS, CHECKS, CHECKS_BY_ID, checks_of
from .scorecard import Scorecard

_SYMBOL = {"pass": "✓", "fail": "✗", "unknown": "?"}


def to_json(card: Scorecard, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    return {
        "commit": card.commit,
        "date": card.date,
        "tiers": list(card.tiers),
        "scores": card.scores,
        "document_scores": card.document_scores,
        "lowest": card.lowest,
        "mean": card.mean,
        "gate_short": card.gate_short,
        "results": {check_id: result.as_dict() for check_id, result in card.results.items()},
        "disagreements": [list(each) for each in card.disagreements],
        "notes": card.notes,
        **(extra or {}),
    }


def write_json(path: Path, card: Scorecard, extra: Optional[Mapping[str, Any]] = None) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_json(card, extra), indent=2, default=str), encoding="utf-8")
    return path


def changes(previous: Optional[Mapping[str, Any]], card: Scorecard) -> dict[str, Any]:
    """Scores and check statuses that differ from *previous* (a scorecard
    JSON), as ``{"scores": [(aspect, before, now)], "checks": [(id, before, now)]}``."""
    if not previous:
        return {"since": None, "scores": [], "checks": []}
    before_scores = previous.get("scores") or {}
    before_results = previous.get("results") or {}
    scores = [
        (key, before_scores.get(key), value) for key, value in card.scores.items() if before_scores.get(key) != value
    ]
    checks = [
        (check_id, (before_results.get(check_id) or {}).get("status"), result.status)
        for check_id, result in card.results.items()
        if (before_results.get(check_id) or {}).get("status") != result.status
    ]
    return {"since": f"{previous.get('date')} at {previous.get('commit')}", "scores": scores, "checks": checks}


def latest_previous(root: Path, current: Path) -> Optional[dict[str, Any]]:
    """The newest other scorecard under *root* (``<date>_<sha>/scorecard.json``)."""
    found = sorted(
        (path for path in Path(root).glob("*/scorecard.json") if path.parent.resolve() != Path(current).resolve()),
        key=lambda path: path.parent.name,
    )
    if not found:
        return None
    return json.loads(found[-1].read_text(encoding="utf-8"))


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def scorecard_table(card: Scorecard) -> str:
    """The summary table, in the document's own layout, to paste into
    ``RELEASE_REVIEW.md``: what holds each aspect is its lowest failing level."""
    lines = ["| # | Aspect | Score | Document | What holds it there (the next checks to pass) |", "|---|---|---|---|---|"]
    for aspect in ASPECTS:
        score = card.scores[aspect.key]
        failing = [
            each.id for each in checks_of(aspect.key)
            if card.results[each.id].status != PASS and each.level <= max(2, score - score % 2 + 2)
        ]
        lines.append(
            f"| {aspect.number} | {aspect.title} | **{score}** | {card.document_scores.get(aspect.key, '-')} | "
            f"{', '.join(failing) or '-'} |"
        )
    return "\n".join(lines)


def markdown(
    card: Scorecard,
    *,
    moved: Mapping[str, Any],
    sections: Sequence[tuple[str, str]] = (),
) -> str:
    """The whole report: scores, what moved, disagreements with the
    document, every check's result and source, then the tiers' own detail."""
    measured = sum(result.source != DOCUMENT for result in card.results.values())
    unknown = [check_id for check_id, result in card.results.items() if result.status == UNKNOWN]
    lines = [
        f"# Release review scorecard, {card.date} at `{card.commit}`",
        "",
        f"Tiers run: {', '.join(card.tiers) or 'none'}. {measured} of {len(CHECKS)} checks measured or recorded here; "
        f"the other {len(CHECKS) - measured} carried from `RELEASE_REVIEW.md`'s rows. "
        f"Unknown counts as a fail ({len(unknown)} unknown).",
        "",
        f"**Lowest {card.lowest}, mean {card.mean}.** "
        + ("The release gate is met." if not card.gate_short else f"Gate: {', '.join(card.gate_short)}."),
        "",
        scorecard_table(card),
        "",
        "## What moved",
        "",
    ]
    if moved.get("since") is None:
        lines.append("No earlier scorecard to compare with.")
    else:
        lines.append(f"Since {moved['since']}:")
        lines.append("")
        lines += [f"- {key}: {before} → {now}" for key, before, now in moved["scores"]] or ["- no score moved"]
        lines += [f"- `{check_id}`: {before} → {now}" for check_id, before, now in moved["checks"]] or ["- no check moved"]
    lines += ["", "## Disagreements with RELEASE_REVIEW.md", ""]
    if card.disagreements:
        lines.append("Each is resolved by fixing the collector or the row (`PLAN_release_review.md` R1).")
        lines.append("")
        for check_id, said, now in card.disagreements:
            result = card.results[check_id]
            lines.append(f"- `{check_id}`: the row says {said}, here {now} ({result.source}). {_short(result.note or result.evidence, 400)}")
    else:
        lines.append("None.")
    lines += ["", "## Every check", ""]
    for aspect in ASPECTS:
        lines += [f"### {aspect.number}. {aspect.title} — {card.scores[aspect.key]}", "",
                  "| ID | By | | Source | Evidence or why not |", "|---|---|---|---|---|"]
        for each in checks_of(aspect.key):
            result: CheckResult = card.results[each.id]
            why = result.note + (" — " if result.note and result.evidence else "") + result.evidence
            lines.append(
                f"| {each.id} | {CHECKS_BY_ID[each.id].kind} | {_SYMBOL[result.status]} | {result.source} | {_cell(_short(why))} |"
            )
        lines.append("")
    for title, body in sections:
        lines += [f"## {title}", "", body.rstrip(), ""]
    return "\n".join(lines) + "\n"


def dataclass_rows(items: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(each) for each in items]
