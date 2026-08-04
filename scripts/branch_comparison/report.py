"""Shaping and rendering a two-run comparison.

The report is deliberately blunt about incompleteness: a run that failed, a
setting that could not be applied, or an artefact that only one side produced
is stated at the top rather than left for a reviewer to notice in a table. A
comparison a reviewer trusts wrongly is worse than no comparison at all.
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .metrics import (
    EdgeAttributeComparison,
    MetricRow,
    StageDiff,
    VtkComparison,
    any_differences,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif"}


@dataclass
class SideReport:
    """What one side of the comparison did, and whether it worked."""

    label: str
    ref: str
    commit: str = ""
    checkout: str = ""
    ok: bool = False
    error: str | None = None
    runtime_seconds: float | None = None
    api_style: str = "unknown"
    applied_settings: int = 0
    #: Settings that define the comparison and this branch could not accept.
    #: Any entry here means the two sides ran different experiments.
    unapplied_required: dict[str, str] = field(default_factory=dict)
    #: Settings this branch does not have at all -- a caveat, not a mismatch,
    #: because the branch has no such knob to set either way.
    unapplied_optional: dict[str, str] = field(default_factory=dict)
    final_graph_source: str | None = None
    output_dir: str = ""
    plot_dir: str = ""

    @property
    def unapplied_settings(self) -> dict[str, str]:
        return {**self.unapplied_required, **self.unapplied_optional}


@dataclass
class PlotPair:
    """One plot filename, as produced by each side (``None`` when absent)."""

    name: str
    current: str | None
    reference: str | None


@dataclass
class ComparisonReport:
    """Everything the rendered report shows."""

    current: SideReport
    reference: SideReport
    image_path: str = ""
    metric_rows: list[MetricRow] = field(default_factory=list)
    stage_diffs: list[StageDiff] = field(default_factory=list)
    first_stage_difference: StageDiff | None = None
    edge_attributes: EdgeAttributeComparison = field(
        default_factory=EdgeAttributeComparison
    )
    statistics_rows: list[MetricRow] = field(default_factory=list)
    branch_statistics_rows: list[MetricRow] = field(default_factory=list)
    vtk: list[VtkComparison] = field(default_factory=list)
    runtime_row: MetricRow | None = None
    plots: list[PlotPair] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Both sides ran, and both were configured the way the tool asked."""
        return (
            self.current.ok
            and self.reference.ok
            and not self.current.unapplied_required
            and not self.reference.unapplied_required
        )

    @property
    def caveats(self) -> int:
        """Settings one side simply does not have; reported, not fatal."""
        return len(self.current.unapplied_optional) + len(
            self.reference.unapplied_optional
        )

    @property
    def differs(self) -> bool:
        return (
            any_differences(self.metric_rows)
            or any(diff.differs for diff in self.stage_diffs)
            or self.edge_attributes.differs
            or any_differences(self.statistics_rows)
            or any_differences(self.branch_statistics_rows)
            or any(vtk.differs for vtk in self.vtk)
        )

    @property
    def failures(self) -> list[SideReport]:
        return [side for side in (self.current, self.reference) if not side.ok]

    def status_line(self) -> str:
        """One sentence a reviewer can act on."""
        if self.failures:
            names = ", ".join(f"{s.label} ({s.ref})" for s in self.failures)
            return f"INCOMPLETE - the {names} run failed; no comparison was made."
        if not self.complete:
            blocked = sorted(
                set(self.current.unapplied_required)
                | set(self.reference.unapplied_required)
            )
            return (
                "INCOMPLETE - both runs finished, but these settings that define "
                f"the comparison could not be applied: {', '.join(blocked)}. The "
                "two runs were not the same experiment."
            )
        caveat = (
            f" {self.caveats} setting(s) do not exist on one side; see the run "
            "cards."
            if self.caveats
            else ""
        )
        verdict = (
            "the two runs produced different numbers."
            if self.differs
            else "the two runs produced identical numbers."
        )
        return f"COMPLETE - {verdict}{caveat}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_value(value: Any) -> str:
    """A number a reviewer can scan, without losing a real difference."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value != 0 and abs(value) < 1e-3 or abs(value) >= 1e7:
            return f"{value:.6g}"
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_delta(row: MetricRow) -> tuple[str, str]:
    delta = row.delta
    if delta is None:
        return ("-", "-")
    percent = row.percent_change
    return (
        format_value(delta),
        "-" if percent is None else f"{percent:+.4g}%",
    )


def _rows_with_differences(rows: Sequence[MetricRow]) -> list[MetricRow]:
    return [row for row in rows if row.flagged]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    rows = ["| " + " | ".join(cells) + " |" for cells in body]
    if not rows:
        rows = ["| " + " | ".join(["-"] * len(header)) + " |"]
    return "\n".join(lines + rows)


def _metric_table(rows: Sequence[MetricRow], current: str, reference: str) -> str:
    body = []
    for row in rows:
        delta, percent = _format_delta(row)
        body.append(
            [
                ("**" + row.name + "**") if row.flagged else row.name,
                row.unit or "",
                format_value(row.current),
                format_value(row.reference),
                delta,
                percent,
                "CHANGED" if row.flagged else "",
            ]
        )
    return _md_table(
        ["Metric", "Unit", current, reference, "Delta", "Change", ""], body
    )


def _side_summary_md(side: SideReport) -> list[str]:
    lines = [
        f"### {side.label}: `{side.ref}`",
        "",
        f"- commit: `{side.commit or 'unknown'}`",
        f"- checkout: `{side.checkout}`",
        f"- entry-point API: {side.api_style}",
        f"- settings applied: {side.applied_settings}",
        f"- final graph taken from: {side.final_graph_source or 'not captured'}",
        f"- runtime: {format_value(side.runtime_seconds)} s",
        f"- status: {'ok' if side.ok else 'FAILED'}",
    ]
    if side.error:
        lines += ["", "```", side.error.strip(), "```"]
    if side.unapplied_required:
        lines += [
            "",
            "**Comparison-defining settings this branch could NOT apply:**",
            "",
        ]
        lines += [
            f"  - `{name}`: {reason}"
            for name, reason in sorted(side.unapplied_required.items())
        ]
    if side.unapplied_optional:
        lines += ["", "Settings this branch does not have (caveats):", ""]
        lines += [
            f"  - `{name}`: {reason}"
            for name, reason in sorted(side.unapplied_optional.items())
        ]
    return lines


def render_markdown(report: ComparisonReport) -> str:
    """The full comparison as Markdown."""
    cur, ref = report.current, report.reference
    out: list[str] = [
        "# Branch comparison",
        "",
        f"**{report.status_line()}**",
        "",
        f"- input image: `{report.image_path}`",
        f"- current: `{cur.ref}` @ `{cur.commit or 'unknown'}`",
        f"- reference: `{ref.ref}` @ `{ref.commit or 'unknown'}`",
        "",
    ]

    if report.warnings:
        out += ["## Warnings", ""]
        out += [f"- {warning}" for warning in report.warnings]
        out += [""]

    out += ["## Runs", ""]
    out += _side_summary_md(cur) + [""]
    out += _side_summary_md(ref) + [""]

    if report.failures:
        out += [
            "## No comparison",
            "",
            "At least one run failed, so the sections below are omitted rather "
            "than shown partially filled.",
            "",
        ]
        return "\n".join(out) + "\n"

    if report.runtime_row is not None:
        out += [
            "## Runtime",
            "",
            _metric_table([report.runtime_row], cur.label, ref.label),
            "",
        ]

    out += [
        "## Graph metrics (final graph)",
        "",
        _metric_table(report.metric_rows, cur.label, ref.label),
        "",
    ]

    out += ["## Per-stage divergence", ""]
    if not report.stage_diffs:
        out += [
            "No `*_graph_after_*.pkl` stage snapshots were found on either side.",
            "",
        ]
    else:
        first = report.first_stage_difference
        if first is None:
            headline = "**No stage differs: the graph is identical at every step.**"
        else:
            detail = ", ".join(first.reasons) or first.status
            headline = f"**First differing stage: `{first.label}` ({detail})**"
        out += [
            headline,
            "",
            _md_table(
                ["#", "Stage", "Status", "What differs"],
                [
                    [
                        str(index),
                        ("**" + diff.label + "**") if diff.differs else diff.label,
                        diff.status,
                        ", ".join(diff.reasons) or "-",
                    ]
                    for index, diff in enumerate(report.stage_diffs, start=1)
                ],
            ),
            "",
        ]

    out += ["## Edge attributes", ""]
    attrs = report.edge_attributes
    out += [
        f"- only on {cur.label}: "
        + (", ".join(f"`{n}`" for n in attrs.only_current) or "none"),
        f"- only on {ref.label}: "
        + (", ".join(f"`{n}`" for n in attrs.only_reference) or "none"),
        "",
        _metric_table(attrs.shared_rows, cur.label, ref.label),
        "",
    ]

    out += ["## Statistics CSV", ""]
    if not report.statistics_rows:
        out += ["No statistics CSV was produced on either side.", ""]
    else:
        changed = _rows_with_differences(report.statistics_rows)
        out += [
            f"{len(changed)} of {len(report.statistics_rows)} metrics changed "
            "(the export timestamp row is ignored).",
            "",
            _metric_table(report.statistics_rows, cur.label, ref.label),
            "",
        ]
    if report.branch_statistics_rows:
        out += [
            "### Branch-order statistics CSV",
            "",
            _metric_table(report.branch_statistics_rows, cur.label, ref.label),
            "",
        ]

    out += ["## VTK exports", ""]
    if not report.vtk:
        out += ["No VTK exports were compared.", ""]
    for vtk in report.vtk:
        missing = []
        if vtk.missing[0]:
            missing.append(cur.label)
        if vtk.missing[1]:
            missing.append(ref.label)
        out += [f"### `{vtk.name}`", ""]
        if missing:
            out += [f"- missing on: {', '.join(missing)}", ""]
        out += [
            f"- cell arrays only on {cur.label}: "
            + (", ".join(f"`{n}`" for n in vtk.only_current) or "none"),
            f"- cell arrays only on {ref.label}: "
            + (", ".join(f"`{n}`" for n in vtk.only_reference) or "none"),
            "",
            _metric_table(
                list(vtk.count_rows) + list(vtk.shared_rows), cur.label, ref.label
            ),
            "",
        ]

    out += [
        "## Plots",
        "",
        f"{len(report.plots)} plot filenames were produced across both runs; "
        "open `index.html` to see them side by side.",
        "",
    ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; --fg:#16181d; --bg:#ffffff; --muted:#5b6472;
        --line:#d9dee6; --panel:#f6f8fa; --changed:#8a3d00; --changed-bg:#fff3e6;
        --ok:#1b6b3a; --bad:#a11414; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e7eaf0; --bg:#11141a; --muted:#9aa4b2; --line:#2b313b;
          --panel:#171b22; --changed:#ffb26b; --changed-bg:#2c1e10;
          --ok:#63d391; --bad:#ff8080; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .5rem; }
h2 { font-size: 1.25rem; margin: 2.5rem 0 .75rem; padding-bottom:.3rem;
     border-bottom: 1px solid var(--line); }
h3 { font-size: 1rem; margin: 1.5rem 0 .5rem; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
pre { background: var(--panel); padding: .75rem 1rem; border-radius: 6px;
      overflow-x: auto; border:1px solid var(--line); }
.banner { padding: .85rem 1rem; border-radius: 8px; font-weight: 600;
          border: 1px solid var(--line); background: var(--panel); }
.banner.ok { color: var(--ok); }
.banner.bad { color: var(--bad); }
.meta { color: var(--muted); font-size: .9rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 1rem;
        background: var(--panel); }
.card ul { margin:.4rem 0 0; padding-left: 1.1rem; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th { background: var(--panel); position: sticky; top: 0; }
tr.changed td { background: var(--changed-bg); color: var(--changed); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.plot { border:1px solid var(--line); border-radius:8px; padding:.75rem; margin-bottom:1.25rem; }
.plot h3 { margin-top:0; }
.pair { display:grid; grid-template-columns: 1fr 1fr; gap:.75rem; }
@media (max-width: 720px) { .pair { grid-template-columns: 1fr; } }
.pair figure { margin:0; }
.pair figcaption { font-size:.8rem; color:var(--muted); margin-bottom:.35rem; }
.pair img { width:100%; height:auto; display:block; background:#fff; border-radius:4px; }
.missing { color: var(--muted); font-style: italic; padding: 2rem 0; text-align:center;
           border:1px dashed var(--line); border-radius:4px; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _html_metric_table(
    rows: Sequence[MetricRow], current: str, reference: str
) -> str:
    if not rows:
        return '<p class="meta">Nothing to compare.</p>'
    head = (
        "<tr><th>Metric</th><th>Unit</th>"
        f"<th>{_e(current)}</th><th>{_e(reference)}</th>"
        "<th>Delta</th><th>Change</th></tr>"
    )
    body = []
    for row in rows:
        delta, percent = _format_delta(row)
        cls = ' class="changed"' if row.flagged else ""
        body.append(
            f"<tr{cls}><td>{_e(row.name)}</td><td>{_e(row.unit)}</td>"
            f'<td class="num">{_e(format_value(row.current))}</td>'
            f'<td class="num">{_e(format_value(row.reference))}</td>'
            f'<td class="num">{_e(delta)}</td>'
            f'<td class="num">{_e(percent)}</td></tr>'
        )
    return (
        '<div class="tablewrap"><table><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _html_side_card(side: SideReport) -> str:
    items = [
        f"<li>commit: <code>{_e(side.commit or 'unknown')}</code></li>",
        f"<li>checkout: <code>{_e(side.checkout)}</code></li>",
        f"<li>entry-point API: {_e(side.api_style)}</li>",
        f"<li>settings applied: {_e(side.applied_settings)}</li>",
        f"<li>final graph from: {_e(side.final_graph_source or 'not captured')}</li>",
        f"<li>runtime: {_e(format_value(side.runtime_seconds))} s</li>",
        f"<li>status: {'ok' if side.ok else '<strong>FAILED</strong>'}</li>",
    ]
    extra = ""
    if side.error:
        extra += f"<pre>{_e(side.error.strip())}</pre>"
    for title, entries in (
        (
            "Comparison-defining settings this branch could NOT apply",
            side.unapplied_required,
        ),
        ("Settings this branch does not have (caveats)", side.unapplied_optional),
    ):
        if not entries:
            continue
        rows = "".join(
            f"<li><code>{_e(name)}</code>: {_e(reason)}</li>"
            for name, reason in sorted(entries.items())
        )
        extra += f"<p><strong>{_e(title)}:</strong></p><ul>{rows}</ul>"
    return (
        f'<div class="card"><h3>{_e(side.label)}: <code>{_e(side.ref)}</code></h3>'
        f"<ul>{''.join(items)}</ul>{extra}</div>"
    )


def _html_plot_figure(caption: str, source: str | None) -> str:
    if source is None:
        return (
            f"<figure><figcaption>{_e(caption)}</figcaption>"
            '<div class="missing">not produced</div></figure>'
        )
    suffix = source.rsplit(".", 1)[-1].lower()
    if f".{suffix}" in _IMAGE_SUFFIXES:
        body = f'<img loading="lazy" src="{_e(source)}" alt="{_e(caption)}">'
    else:
        body = f'<p><a href="{_e(source)}">open {_e(source.rsplit("/", 1)[-1])}</a></p>'
    return f"<figure><figcaption>{_e(caption)}</figcaption>{body}</figure>"


def render_html(report: ComparisonReport) -> str:
    """The full comparison as a self-contained HTML page."""
    cur, ref = report.current, report.reference
    ok = report.complete and not report.failures
    parts: list[str] = [
        f"<title>Branch comparison: {_e(cur.ref)} vs {_e(ref.ref)}</title>",
        f"<style>{_CSS}</style>",
        "<main>",
        "<h1>Branch comparison</h1>",
        f'<p class="meta">current <code>{_e(cur.ref)}</code> vs reference '
        f"<code>{_e(ref.ref)}</code> &middot; input <code>{_e(report.image_path)}</code></p>",
        f'<p class="banner {"ok" if ok else "bad"}">{_e(report.status_line())}</p>',
    ]

    if report.warnings:
        warnings = "".join(f"<li>{_e(w)}</li>" for w in report.warnings)
        parts.append(f"<h2>Warnings</h2><ul>{warnings}</ul>")

    parts.append("<h2>Runs</h2>")
    parts.append(
        '<div class="grid">' + _html_side_card(cur) + _html_side_card(ref) + "</div>"
    )

    if report.failures:
        parts.append(
            "<h2>No comparison</h2><p>At least one run failed, so no tables or "
            "plots are shown. Fix the failing side and re-run.</p></main>"
        )
        return "\n".join(parts)

    if report.runtime_row is not None:
        parts.append("<h2>Runtime</h2>")
        parts.append(_html_metric_table([report.runtime_row], cur.label, ref.label))

    parts.append("<h2>Graph metrics (final graph)</h2>")
    parts.append(_html_metric_table(report.metric_rows, cur.label, ref.label))

    parts.append("<h2>Per-stage divergence</h2>")
    if not report.stage_diffs:
        parts.append(
            '<p class="meta">No <code>*_graph_after_*.pkl</code> stage snapshots '
            "were found on either side.</p>"
        )
    else:
        first = report.first_stage_difference
        if first is None:
            parts.append(
                '<p class="banner ok">No stage differs: the graph is identical at '
                "every step.</p>"
            )
        else:
            detail = ", ".join(first.reasons) or first.status
            parts.append(
                f'<p class="banner bad">First differing stage: '
                f"<code>{_e(first.label)}</code> ({_e(detail)})</p>"
            )
        stage_cells = []
        for index, diff in enumerate(report.stage_diffs, start=1):
            cls = ' class="changed"' if diff.differs else ""
            reasons = ", ".join(diff.reasons) or "-"
            stage_cells.append(
                f"<tr{cls}><td>{index}</td><td><code>{_e(diff.label)}</code></td>"
                f"<td>{_e(diff.status)}</td><td>{_e(reasons)}</td></tr>"
            )
        rows = "".join(stage_cells)
        parts.append(
            '<div class="tablewrap"><table><thead><tr><th>#</th><th>Stage</th>'
            "<th>Status</th><th>What differs</th></tr></thead><tbody>"
            + rows
            + "</tbody></table></div>"
        )

    attrs = report.edge_attributes
    parts.append("<h2>Edge attributes</h2>")
    parts.append(
        f'<p class="meta">Only on {_e(cur.label)}: '
        + (", ".join(f"<code>{_e(n)}</code>" for n in attrs.only_current) or "none")
        + f" &middot; only on {_e(ref.label)}: "
        + (", ".join(f"<code>{_e(n)}</code>" for n in attrs.only_reference) or "none")
        + "</p>"
    )
    parts.append(_html_metric_table(attrs.shared_rows, cur.label, ref.label))

    parts.append("<h2>Statistics CSV</h2>")
    if not report.statistics_rows:
        parts.append('<p class="meta">No statistics CSV was produced on either side.</p>')
    else:
        changed = len(_rows_with_differences(report.statistics_rows))
        parts.append(
            f'<p class="meta">{changed} of {len(report.statistics_rows)} metrics '
            "changed (the export timestamp row is ignored).</p>"
        )
        parts.append(_html_metric_table(report.statistics_rows, cur.label, ref.label))
    if report.branch_statistics_rows:
        parts.append("<h3>Branch-order statistics CSV</h3>")
        parts.append(
            _html_metric_table(report.branch_statistics_rows, cur.label, ref.label)
        )

    parts.append("<h2>VTK exports</h2>")
    if not report.vtk:
        parts.append('<p class="meta">No VTK exports were compared.</p>')
    for vtk in report.vtk:
        parts.append(f"<h3><code>{_e(vtk.name)}</code></h3>")
        missing = [
            label
            for label, is_missing in zip((cur.label, ref.label), vtk.missing)
            if is_missing
        ]
        if missing:
            parts.append(f'<p class="meta">missing on: {_e(", ".join(missing))}</p>')
        parts.append(
            f'<p class="meta">Cell arrays only on {_e(cur.label)}: '
            + (", ".join(f"<code>{_e(n)}</code>" for n in vtk.only_current) or "none")
            + f" &middot; only on {_e(ref.label)}: "
            + (", ".join(f"<code>{_e(n)}</code>" for n in vtk.only_reference) or "none")
            + "</p>"
        )
        parts.append(
            _html_metric_table(
                list(vtk.count_rows) + list(vtk.shared_rows), cur.label, ref.label
            )
        )

    parts.append("<h2>Plots</h2>")
    if not report.plots:
        parts.append('<p class="meta">Neither run produced any plots.</p>')
    for pair in report.plots:
        parts.append(
            f'<div class="plot"><h3>{_e(pair.name)}</h3><div class="pair">'
            + _html_plot_figure(f"{cur.label} ({cur.ref})", pair.current)
            + _html_plot_figure(f"{ref.label} ({ref.ref})", pair.reference)
            + "</div></div>"
        )

    parts.append("</main>")
    return "\n".join(parts)
