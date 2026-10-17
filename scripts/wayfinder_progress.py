"""Build a wayfinder map's progress.html, or list its frontier.

    venv/bin/python scripts/wayfinder_progress.py .scratch/<effort>             # write <effort>/progress.html
    venv/bin/python scripts/wayfinder_progress.py .scratch/<effort> --frontier  # print the takeable tickets

Reads map.md, issues/*.md (their Type, Status, Blocked by, Model, Effort and Why lines),
sessions.md (one "## <mode>" entry per session, with Date, Mode, Call, Ticket, Model, Effort
and Why lines) and, when present, assets/inventory.md. Run it after each wayfinder session.
"""
import argparse
import html
import os
import re
from datetime import datetime
from pathlib import Path


def sections(text):
    """Split markdown on '## ' headings -> {heading: body}."""
    out, name, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if name is not None:
                out[name] = "\n".join(buf).strip()
            name, buf = line[3:].strip(), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf).strip()
    return out


def inline(s, base=""):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)

    def link(m):
        href = m.group(2)
        if base and not re.match(r"[a-z]+:", href):
            href = os.path.normpath(os.path.join(base, href))
        return f'<a href="{html.escape(href)}">{m.group(1)}</a>'

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, s)


def md(text, base=""):
    """Tiny markdown: paragraphs, nested - lists, 1. lists, tables, ### headings."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, para, stack, table = [], [], [], []

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para), base) + "</p>")
            para.clear()

    def close_lists(depth=0):
        while len(stack) > depth:
            out.append(f"</li></{stack.pop()}>")

    def flush_table():
        if table:
            rows = [r for r in table if not re.match(r"^\|[\s|:-]+\|$", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c, base)}</th>" for c in cells[0]) + "</tr></thead><tbody>")
            for r in cells[1:]:
                out.append("<tr>" + "".join(f"<td>{inline(c, base)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            table.clear()

    for line in text.splitlines():
        m = re.match(r"^(\s*)(-|\d+\.)\s+(.*)", line)
        if line.strip().startswith("|"):
            flush_para(); close_lists(); table.append(line.strip()); continue
        flush_table()
        if m:
            flush_para()
            depth = len(m.group(1)) // 2 + 1
            tag = "ul" if m.group(2) == "-" else "ol"
            if depth > len(stack):
                while depth > len(stack):
                    out.append(f"<{tag}><li>"); stack.append(tag)
            else:
                close_lists(depth)
                out.append("</li><li>")
            out.append(inline(m.group(3), base))
        elif not line.strip():
            flush_para()
            if stack and not line:
                pass
        elif line.startswith("### "):
            flush_para(); close_lists(); out.append(f"<h4>{inline(line[4:], base)}</h4>")
        elif stack and line.startswith(" "):
            out.append(" " + inline(line.strip(), base))
        else:
            close_lists(); para.append(line.strip())
    flush_para(); flush_table(); close_lists()
    return "\n".join(out)


def fields(text):
    return dict(re.findall(r"^(Type|Status|Blocked by|Date|Mode|Call|Ticket|Model|Effort|Why):\s*(.*)$", text, re.M))


def rec_html(f):
    """Recommended model and effort line, or "" when the entry has none."""
    if not f.get("Model"):
        return ""
    why = f" — {inline(f['Why'])}" if f.get("Why") else ""
    return (f'<p class="rec"><span class="b model">{html.escape(f["Model"])}</span> '
            f'<span class="b effort">effort: {html.escape(f.get("Effort", "?"))}</span>{why}</p>')


def load_tickets(here):
    tickets = []
    for p in sorted((here / "issues").glob("*.md")):
        text = p.read_text()
        title = re.search(r"^# (?:\d+:\s*)?(.*)$", text, re.M).group(1)
        f, sec = fields(text), sections(text)
        num = p.name[:2]
        blockers = re.findall(r"\d+", f.get("Blocked by", "")) if "none" not in f.get("Blocked by", "") else []
        tickets.append(dict(num=num, title=title, path=f"issues/{p.name}", type=f.get("Type", "?"),
                            status=f.get("Status", "open").split()[0], blockers=[b.zfill(2) for b in blockers],
                            question=sec.get("Question", ""), answer=sec.get("Answer", ""), rec=rec_html(f)))
    by = {t["num"]: t for t in tickets}
    for t in tickets:
        t["frontier"] = t["status"] == "open" and all(by.get(b, {}).get("status") == "resolved" for b in t["blockers"])
    return tickets, by


def frontier_lines(tickets):
    """One line per frontier ticket (open, every blocker resolved), then any claimed ones."""
    lines = [f'{t["num"]} {t["title"]} ({t["type"]}) {t["path"]}' for t in tickets if t["frontier"]]
    lines = lines or ["no ticket on the frontier"]
    lines += [f'claimed: {t["num"]} {t["title"]} {t["path"]}' for t in tickets if t["status"] == "claimed"]
    return lines


def write_page(here):
    map_text = (here / "map.md").read_text()
    m = sections(map_text)
    map_title = re.search(r"^# (.*)$", map_text, re.M).group(1)
    tickets, by = load_tickets(here)
    done = sum(t["status"] == "resolved" for t in tickets)
    pct = round(100 * done / len(tickets)) if tickets else 0
    fog = len(re.findall(r"^- ", m.get("Not yet specified", ""), re.M))

    sess_text = (here / "sessions.md").read_text() if (here / "sessions.md").exists() else ""
    # Session headings repeat ("Work through the map"), so split in order rather than by key.
    sessions = [(c.split("\n", 1)[0].strip(), c.split("\n", 1)[1].strip() if "\n" in c else "")
                for c in re.split(r"^## ", sess_text, flags=re.M)[1:]]

    inv = here / "assets" / "inventory.md"
    inv_html = ""
    if inv.exists():
        it = inv.read_text()
        head = it.split("## ", 1)[0].split("\n", 1)[1]
        inv_html = (f'<h2>Inventory headline</h2>{md(head)}{md(sections(it).get("Totals by scope class", ""))}'
                    '<p><a href="assets/inventory.md">Full inventory</a></p>')

    def badge(t):
        if t["status"] == "resolved":
            return '<span class="b done">resolved</span>'
        if t["status"] == "claimed":
            return '<span class="b claimed">claimed</span>'
        if t["frontier"]:
            return '<span class="b front">frontier</span>'
        return '<span class="b blocked">blocked</span>'

    def name(num):
        t = by.get(num)
        return f'<a href="#t{num}">{html.escape(t["title"])}</a>' if t else num

    rows = []
    no_rec = '<span class="muted">see session</span>'  # outside the f-string: Python < 3.12 forbids a backslash in one
    for t in tickets:
        blk = ", ".join(name(b) for b in t["blockers"]) or "none"
        rows.append(f'<tr><td><a href="#t{t["num"]}">{html.escape(t["title"])}</a></td><td>{html.escape(t["type"])}</td>'
                    f'<td>{badge(t)}</td><td>{blk}</td><td>{t["rec"] or no_rec}</td></tr>')

    cards = []
    for t in tickets:
        ans = (f'<details open><summary>Answer</summary>{md(t["answer"], "issues")}</details>' if t["answer"]
               else '<p class="muted">No answer yet.</p>')
        cards.append(f'<article id="t{t["num"]}" class="card {t["status"]}"><h3>{html.escape(t["title"])} {badge(t)}</h3>'
                     f'<p class="muted">{html.escape(t["type"])} · <a href="{t["path"]}">{t["path"]}</a></p>{t["rec"]}'
                     f'<details><summary>Question</summary>{md(t["question"], "issues")}</details>{ans}</article>')

    sess_html = []
    for i, (n, body) in enumerate(sessions, 1):
        f = fields(body)
        rest = "\n".join(l for l in body.splitlines() if not re.match(r"^(Date|Mode|Call|Ticket|Model|Effort|Why):", l))
        tk = f.get("Ticket", "")
        tk_html = name(tk.zfill(2)) if tk.strip().isdigit() else html.escape(tk)
        sess_html.append(f'<li class="sess"><div class="dot">{i}</div><div><h3>{html.escape(n)}</h3>'
                         f'<p class="muted">{html.escape(f.get("Date", ""))} · mode: {html.escape(f.get("Mode", ""))} · ticket: {tk_html}</p>'
                         f'{rec_html(f)}<p>{inline(f.get("Call", ""))}</p>{md(rest)}</div></li>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Progress: {html.escape(map_title)}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{margin-bottom:.2rem}} h2{{border-bottom:2px solid #ddd;padding-bottom:.2rem;margin-top:2.2rem}}
code{{background:#f2f2f2;padding:0 .25em;border-radius:3px;font-size:.9em}}
.muted{{color:#777;font-size:.9em;margin:.2rem 0}}
.bar{{background:#eee;border-radius:8px;height:18px;overflow:hidden}} .bar>div{{background:#2e8b57;height:100%}}
.stats{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}} .stat{{background:#f7f7f7;border-radius:8px;padding:.6rem 1rem}}
.stat b{{display:block;font-size:1.5em}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0}} th,td{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;vertical-align:top}}
th{{background:#f5f5f5}}
.b{{font-size:.75em;padding:.1em .55em;border-radius:10px;color:#fff;vertical-align:middle;font-weight:600}}
.done{{background:#2e8b57}} .front{{background:#1e6fd9}} .blocked{{background:#999}} .claimed{{background:#d98a1e}}
.card{{border:1px solid #ddd;border-left:6px solid #999;border-radius:6px;padding:.4rem 1rem;margin:.8rem 0}}
.card.resolved{{border-left-color:#2e8b57}} .card.claimed{{border-left-color:#d98a1e}}
.card h3{{margin:.4rem 0}} summary{{cursor:pointer;font-weight:600}}
ol.timeline{{list-style:none;padding:0}} .sess{{display:flex;gap:1rem;margin-bottom:1rem}}
.sess h3{{margin:.2rem 0}} .dot{{flex:none;width:2rem;height:2rem;border-radius:50%;background:#1e6fd9;color:#fff;display:grid;place-items:center;font-weight:700}}
.model{{background:#6a4fb3}} .effort{{background:#555}} .rec{{margin:.2rem 0;font-size:.9em}}
.dest{{background:#eef5ff;border-left:4px solid #1e6fd9;padding:.5rem 1rem}}
</style></head><body>
<h1>{html.escape(map_title)}</h1>
<p class="muted">Built {datetime.now():%Y-%m-%d %H:%M} from <a href="map.md">map.md</a>. Rerun <code>venv/bin/python scripts/wayfinder_progress.py .scratch/{html.escape(here.name)}</code> after each session.</p>

<div class="dest"><strong>Destination.</strong> {md(m.get("Destination", ""))}</div>

<div class="stats">
<div class="stat"><b>{done} / {len(tickets)}</b>tickets resolved</div>
<div class="stat"><b>{sum(t["frontier"] for t in tickets)}</b>on the frontier</div>
<div class="stat"><b>{sum(t["status"] == "open" and not t["frontier"] for t in tickets)}</b>blocked</div>
<div class="stat"><b>{fog}</b>patches of fog</div>
<div class="stat"><b>{len(sessions)}</b>wayfinder sessions</div>
</div>
<div class="bar"><div style="width:{pct}%"></div></div>
<p class="muted">{pct}% of the current tickets resolved. Fog graduates into new tickets as the frontier moves, so the total will grow.</p>

<h2>Wayfinder sessions</h2>
<ol class="timeline">{"".join(sess_html)}</ol>

<h2>Ticket board</h2>
<table><thead><tr><th>Ticket</th><th>Type</th><th>State</th><th>Blocked by</th><th>Recommended model · effort</th></tr></thead><tbody>{"".join(rows)}</tbody></table>

<h2>Decisions so far</h2>{md(m.get("Decisions so far", ""))}

{inv_html}

<h2>Tickets</h2>{"".join(cards)}

<h2>Not yet specified (fog)</h2>{md(m.get("Not yet specified", ""))}

<h2>Out of scope</h2>{md(m.get("Out of scope", ""))}
</body></html>
"""
    (here / "progress.html").write_text(page)
    print(f"wrote {here / 'progress.html'}: {done}/{len(tickets)} resolved, {len(sessions)} sessions")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("map_dir", type=Path, help="the map's folder, holding map.md and issues/")
    parser.add_argument("--frontier", action="store_true", help="print the frontier instead of writing the page")
    args = parser.parse_args(argv)
    if not (args.map_dir / "map.md").exists():
        parser.error(f"no map.md in {args.map_dir}")
    if args.frontier:
        print("\n".join(frontier_lines(load_tickets(args.map_dir)[0])))
    else:
        write_page(args.map_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
