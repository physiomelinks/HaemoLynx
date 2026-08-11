"""Render the H1 whitepaper markdown to a self-contained HTML artifact."""
import base64
import html
import re
from pathlib import Path

ROOT = Path("/home/dsas627/PycharmProjects/ImageLynx")
SRC = ROOT / "H1_preliminary_results_whitepaper.md"
FIGDIR = ROOT / "examples" / "outputs" / "cb_h1_batch"
OUT = ROOT / "examples" / "outputs" / "cb_h1_batch" / "h1_whitepaper.html"

GRADES = {
    "Established": "ok",
    "Provisional (weak)": "warn",
    "Provisional": "warn",
    "Not supported": "no",
    "Implemented": "ok",
    "Implemented, not conclusive": "warn",
    "Implemented, confounded": "warn",
    "Not implementable": "no",
}


def embed(name):
    data = base64.b64encode((FIGDIR / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def inline(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", text)
    return text


def grade_chip(cell_html, plain):
    stripped = plain.strip().strip("*").strip()
    for label, kind in GRADES.items():
        if stripped == label:
            return f'<span class="chip chip--{kind}">{html.escape(label)}</span>'
    return cell_html


def render_table(rows):
    head, body = rows[0], rows[2:]
    out = ['<div class="scroll"><table><thead><tr>']
    for cell in head:
        out.append(f"<th>{inline(cell)}</th>")
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        for cell in row:
            plain = cell.strip()
            numeric = bool(re.fullmatch(r"[−+\-0-9.,%×⁴⁵⁶⁻¹²³ ]+", plain)) and any(
                c.isdigit() for c in plain)
            klass = ' class="num"' if numeric else ""
            out.append(f"<td{klass}>{grade_chip(inline(cell), plain)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(markdown_text):
    lines = markdown_text.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre class="scroll"><code>{html.escape(chr(10).join(block))}</code></pre>')
            continue

        if stripped == "---":
            out.append('<hr />')
            i += 1
            continue

        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2:
                out.append(render_table(rows))
            continue

        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            text = " ".join(block)
            figure = re.match(r"\*\*Figure (\d)\*\*\s*—\s*`([^`]+)`\.\s*(.*)", text)
            if figure:
                number, filename, caption = figure.groups()
                out.append(
                    f'<figure><img src="{embed(filename)}" alt="Figure {number}" />'
                    f'<figcaption><span class="figlabel">Figure {number}</span>'
                    f'{inline(caption)}</figcaption></figure>')
            else:
                out.append(f'<blockquote>{inline(text)}</blockquote>')
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        heading = re.match(r"(#{1,4})\s+(.*)", stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2)
            number = re.match(r"((?:Appendix [A-D]|\d+(?:\.\d+)?))\.?\s+—?\s*(.*)", text)
            anchor = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            if level == 2 and number:
                label, rest = number.groups()
                out.append(f'<h2 id="{anchor}"><span class="secnum">{html.escape(label)}</span>'
                           f'{inline(rest)}</h2>')
            elif level == 3 and number:
                label, rest = number.groups()
                out.append(f'<h3 id="{anchor}"><span class="subnum">{html.escape(label)}</span>'
                           f'{inline(rest)}</h3>')
            else:
                out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            i += 1
            continue

        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|\||>|-\s|```|---$)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def build_nav(markdown_text):
    entries = []
    for line in markdown_text.split("\n"):
        heading = re.match(r"##\s+(.*)", line.strip())
        if heading and not line.strip().startswith("###"):
            text = heading.group(1)
            anchor = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            number = re.match(r"((?:Appendix [A-D]|\d+))\.?\s+—?\s*(.*)", text)
            label, rest = number.groups() if number else ("", text)
            # Only drop a trailing clause when the head is already descriptive; the three
            # Results sections are distinguished solely by what follows the dash.
            head = re.split(r"\s*—\s*", rest)[0]
            if head.lower() not in ("results",) and len(head) > 6:
                rest = head
            label = label.replace("Appendix ", "App. ")
            entries.append(f'<a href="#{anchor}"><span>{html.escape(label)}</span>'
                           f'{html.escape(rest)}</a>')
    return "\n".join(entries)


CSS = """
:root{
  --paper:#f6f8fa; --raised:#ffffff; --ink:#131920; --muted:#56626f; --faint:#7c8896;
  --rule:#dce3ea; --rule-soft:#eaeff4; --accent:#1f5fa8; --accent-soft:#e8f0f9;
  --wky:#2a78d6; --shr:#eb6834;
  --ok:#166f48; --ok-bg:#e4f1ea; --warn:#8f5610; --warn-bg:#f8eede;
  --no:#93302f; --no-bg:#f8e6e6;
  --serif:"Charter","Bitstream Charter","Iowan Old Style","Palatino Linotype",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono","Roboto Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#11161c; --raised:#171e26; --ink:#e7edf4; --muted:#a0adbc; --faint:#7d8a99;
    --rule:#26313d; --rule-soft:#1d262f; --accent:#7fb2ec; --accent-soft:#16283d;
    --wky:#5a9ae8; --shr:#f0834f;
    --ok:#6cc79a; --ok-bg:#12291f; --warn:#e0aa63; --warn-bg:#2c2115;
    --no:#e88b8a; --no-bg:#2d1a1a;
  }
}
:root[data-theme="dark"]{
  --paper:#11161c; --raised:#171e26; --ink:#e7edf4; --muted:#a0adbc; --faint:#7d8a99;
  --rule:#26313d; --rule-soft:#1d262f; --accent:#7fb2ec; --accent-soft:#16283d;
  --wky:#5a9ae8; --shr:#f0834f;
  --ok:#6cc79a; --ok-bg:#12291f; --warn:#e0aa63; --warn-bg:#2c2115;
  --no:#e88b8a; --no-bg:#2d1a1a;
}
*{box-sizing:border-box;}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{display:grid; grid-template-columns:1fr; gap:0; max-width:1180px; margin:0 auto; padding:0 24px;}
@media(min-width:1040px){
  .wrap{grid-template-columns:230px minmax(0,1fr); gap:56px; padding:0 40px;}
}
nav{display:none;}
@media(min-width:1040px){
  nav{
    display:block; position:sticky; top:0; align-self:start; max-height:100vh;
    overflow-y:auto; padding:56px 0 40px; font-family:var(--sans); font-size:12.5px;
    line-height:1.45;
  }
  nav a{
    display:grid; grid-template-columns:46px 1fr; gap:8px; padding:5px 0;
    color:var(--muted); text-decoration:none; border-left:2px solid transparent;
    padding-left:12px; margin-left:-14px;
  }
  nav a span{color:var(--faint); font-variant-numeric:tabular-nums; white-space:nowrap;}
  nav a:hover{color:var(--accent); border-left-color:var(--accent);}
  nav a:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
}
main{padding:56px 0 96px; min-width:0;}
.masthead{border-bottom:2px solid var(--ink); padding-bottom:28px; margin-bottom:14px;}
.eyebrow{
  font-family:var(--sans); font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); margin:0 0 14px;
}
h1{font-size:2.45rem; line-height:1.14; margin:0 0 12px; font-weight:600; letter-spacing:-.012em; text-wrap:balance;}
.standfirst{font-size:1.16rem; color:var(--muted); margin:0 0 18px; text-wrap:balance;}
.byline{font-family:var(--sans); font-size:12.5px; color:var(--faint); margin:0;}
.cohorts{display:flex; gap:20px; flex-wrap:wrap; margin:20px 0 0; font-family:var(--sans); font-size:12.5px; color:var(--muted);}
.cohorts b{display:inline-flex; align-items:center; gap:7px; font-weight:500;}
.swatch{width:11px; height:11px; border-radius:2px; display:inline-block;}
main > *{max-width:68ch;}
main > h2:first-of-type{border-top:none; margin-top:34px; padding-top:0;}
main > .scroll, main > figure{max-width:none;}
h2{
  font-size:1.5rem; font-weight:600; letter-spacing:-.008em; margin:56px 0 4px;
  padding-top:26px; border-top:1px solid var(--rule); display:flex; gap:14px; align-items:baseline;
  text-wrap:balance;
}
h3{font-size:1.11rem; font-weight:600; margin:34px 0 2px; display:flex; gap:11px; align-items:baseline;}
h4{font-size:1rem; font-weight:600; margin:26px 0 2px;}
.secnum,.subnum{
  font-family:var(--mono); font-size:.72em; color:var(--accent); font-weight:500;
  font-variant-numeric:tabular-nums; flex:none;
}
p{margin:14px 0;}
ul{margin:14px 0; padding-left:22px;}
li{margin:7px 0;}
strong{font-weight:600;}
code{font-family:var(--mono); font-size:.85em; background:var(--rule-soft); padding:1px 5px; border-radius:3px;}
pre{
  background:var(--raised); border:1px solid var(--rule); border-radius:6px;
  padding:16px 18px; margin:20px 0; font-size:13px; line-height:1.6;
}
pre code{background:none; padding:0; font-size:inherit;}
hr{border:none; height:0; margin:0;}
blockquote{
  margin:22px 0; padding:14px 20px; background:var(--accent-soft);
  border-left:3px solid var(--accent); border-radius:0 4px 4px 0; color:var(--ink);
}
blockquote p{margin:0;}
.scroll{overflow-x:auto; margin:22px 0; border:1px solid var(--rule); border-radius:6px; background:var(--raised);}
table{border-collapse:collapse; width:100%; font-size:14px; font-family:var(--sans);}
th{
  text-align:left; font-weight:600; font-size:11px; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted); padding:11px 14px; border-bottom:1.5px solid var(--rule);
  background:var(--rule-soft); white-space:nowrap;
}
td{padding:10px 14px; border-bottom:1px solid var(--rule-soft); vertical-align:top;}
tr:last-child td{border-bottom:none;}
td.num{font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap;}
.chip{
  display:inline-block; font-size:11px; font-weight:600; letter-spacing:.03em;
  padding:3px 9px; border-radius:11px; white-space:nowrap;
}
.chip--ok{background:var(--ok-bg); color:var(--ok);}
.chip--warn{background:var(--warn-bg); color:var(--warn);}
.chip--no{background:var(--no-bg); color:var(--no);}
figure{margin:30px 0; padding:0;}
figure img{width:100%; height:auto; display:block; border:1px solid var(--rule); border-radius:6px; background:#fcfcfb;}
figcaption{
  font-family:var(--sans); font-size:12.5px; line-height:1.55; color:var(--muted);
  margin-top:11px; max-width:74ch;
}
.figlabel{
  display:inline-block; font-weight:600; color:var(--accent); letter-spacing:.05em;
  text-transform:uppercase; font-size:11px; margin-right:9px;
}
a{color:var(--accent);}
a:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
@media(prefers-reduced-motion:reduce){*{animation:none !important; transition:none !important;}}
"""


def main():
    text = SRC.read_text()
    body = text.split("---", 1)[1] if text.startswith("#") else text
    # Drop the original title block; the masthead replaces it.
    body = re.sub(r"^.*?## 0\. Executive summary", "## 0. Executive summary", text,
                  flags=re.S)
    page = f"""<title>H1 Preliminary Results — Carotid Body Microvascular Morphology</title>
<style>{CSS}</style>
<div class="wrap">
<nav aria-label="Contents">{build_nav(body)}</nav>
<main>
<header class="masthead">
  <p class="eyebrow">Preliminary report · Hypothesis 1</p>
  <h1>Carotid body microvascular morphology, SHR versus WKY</h1>
  <p class="standfirst">A methods-maturation milestone: what the measurement pipeline can
  now support, what it cannot, and the evidence for the difference.</p>
  <p class="byline">ImageLynx / <code>carotid_image_to_model</code> ·
  branch <code>cb_pipeline_improvements_sweep</code> · Dale Sasis</p>
  <div class="cohorts">
    <b><span class="swatch" style="background:var(--wky)"></span>WKY — normotensive control, n = 3</b>
    <b><span class="swatch" style="background:var(--shr)"></span>SHR — spontaneously hypertensive, n = 3</b>
  </div>
</header>
{convert(body)}
</main>
</div>
"""
    OUT.write_text(page)
    print(f"wrote {OUT}  ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
