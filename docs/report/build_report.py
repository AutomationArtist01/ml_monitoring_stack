#!/usr/bin/env python3
"""
Build docs/Team4-Project-Report.pdf from docs/report/report.md.

Directives inside report.md:
  {{code:relative/path.py:START-END}}      → syntax-highlighted listing of those source lines (real code, never pasted)
  {{code:relative/path.py}}                → whole file
  {{file:relative/path}}                   → whole file, plain (yaml, txt)
  ![Caption](screenshots/x.png)            → figure with automatic "Figure N:" caption
Needs: pip install markdown pygments ; Google Chrome for the PDF step.
"""
import base64
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, TextLexer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "report.md"
OUT_PDF = HERE.parent / "Team4-Project-Report.pdf"

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm; @bottom-center { content: counter(page); font: 10pt Georgia, serif; color:#333; } }
html { font-size: 11pt; }
body { font-family: Georgia, "Times New Roman", "DejaVu Serif", serif; color:#111; line-height:1.42; }
h1 { font-family: Georgia, serif; font-size: 18pt; font-weight: 700; margin: 26pt 0 8pt; page-break-after: avoid; }
h1.chapter { page-break-before: always; }
h2 { font-size: 13.5pt; font-weight: 700; margin: 18pt 0 6pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: 700; margin: 12pt 0 4pt; }
p { margin: 5pt 0 8pt; text-align: justify; }
li { margin: 2pt 0; }
code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 8.9pt; }
.codebox { background:#f5f5f5; border-radius:3px; padding:9px 11px; margin:8pt 0 10pt; font-size:8.6pt; line-height:1.38; white-space:pre-wrap; word-break:break-word; page-break-inside:auto; }
.codebox .filename { font-family: Georgia, serif; font-size:8.5pt; color:#555; margin-bottom:4px; }
.codebox pre { margin:0; font-family:"DejaVu Sans Mono", Menlo, Consolas, monospace; }
pre.plain { background:#f5f5f5; padding:9px 11px; border-radius:3px; font-size:8.6pt; white-space:pre-wrap; }
table { border-collapse: collapse; width:100%; margin: 8pt 0 12pt; font-size: 9.5pt; }
th, td { border:1px solid #444; padding:4px 7px; vertical-align:top; text-align:left; }
th { background:#eee; font-weight:700; }
figure { margin: 10pt 0 14pt; page-break-inside: avoid; text-align:center; }
figure img { max-width:100%; border:1px solid #999; }
figcaption { font-size:9.5pt; margin-top:5pt; text-align:justify; }
figcaption b { font-weight:700; }
hr { border:none; border-top:1px solid #444; width:45%; margin:16pt auto; }
.cover { text-align:center; padding-top:120pt; page-break-after: always; }
.cover h1 { font-size:26pt; margin:0 0 14pt; }
.cover .sub { font-size:13pt; margin: 6pt 0; }
.cover .tools { font-size:11pt; color:#333; margin: 4pt 0 60pt; }
.cover table { width:auto; margin: 0 auto; border:none; font-size:11pt; }
.cover td { border:none; padding:3px 14px; text-align:left; }
.cover .by { font-weight:700; margin-bottom:8pt; }
.toc ul { list-style:none; padding-left:0; column-count:2; column-gap:24pt; font-size:9.8pt; }
.toc li { margin:1pt 0; }
.note { font-style: italic; }
blockquote { border-left:3px solid #999; margin:8pt 0; padding:2pt 10pt; color:#333; }
"""

FIG = [0]
formatter = HtmlFormatter(nowrap=True, noclasses=True, style="default")


def code_block(rel: str, start: int | None, end: int | None, plain=False) -> str:
    p = ROOT / rel
    if not p.exists():
        return f'<p style="color:#b00"><b>[missing source: {rel}]</b></p>'
    lines = p.read_text().splitlines()
    s, e = (start or 1), (end or len(lines))
    chunk = "\n".join(lines[s - 1:e])
    if plain:
        body = html.escape(chunk)
    else:
        try:
            lexer = get_lexer_for_filename(p.name)
        except Exception:  # noqa: BLE001
            lexer = TextLexer()
        body = highlight(chunk, lexer, formatter)
    rng = f" (lines {s}–{e})" if (start or end) else ""
    return f'<div class="codebox"><div class="filename">{html.escape(rel)}{rng}</div><pre>{body}</pre></div>'


def expand_directives(md: str) -> str:
    def code(m):
        rel, a, b = m.group(1), m.group(2), m.group(3)
        s = int(a) if a else None
        e = int(b) if b else None
        return "\n" + code_block(rel, s, e) + "\n"
    md = re.sub(r"\{\{code:([^:}]+)(?::(\d+)[-:](\d+))?\}\}", code, md)
    md = re.sub(r"\{\{file:([^}]+)\}\}", lambda m: "\n" + code_block(m.group(1), None, None, plain=True) + "\n", md)
    return md


def figures(md: str) -> str:
    def repl(m):
        cap, rel = m.group(1), m.group(2)
        p = HERE / rel
        if not p.exists():
            return f'<p style="color:#b00"><b>[screenshot missing: {rel}]</b></p>'
        FIG[0] += 1
        data = base64.b64encode(p.read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}"><figcaption><b>Figure {FIG[0]}:</b> {cap}</figcaption></figure>'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, md)


def main():
    md = SRC.read_text()
    md = expand_directives(md)
    md = figures(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
                             extension_configs={"toc": {"toc_depth": "1-2"}})
    # numbered top-level sections start on a new page (LaTeX \\section feel) except the first
    body = re.sub(r'<h1(?![^>]*class=)', '<h1 class="chapter"', body)
    body = body.replace('<h1 class="chapter"', '<h1', 1)  # first h1 after cover: no extra break
    html_doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>Team 4 – Project Report</title>"
                f"<style>{CSS}</style></head><body>{body}</body></html>")
    tmp = HERE / "_report.html"
    tmp.write_text(html_doc)
    chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                               shutil.which("google-chrome"), shutil.which("chromium")] if c and os.path.exists(c)), None)
    if not chrome:
        print("Chrome not found; open", tmp)
        sys.exit(1)
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox",
                    f"--print-to-pdf={OUT_PDF}", "--virtual-time-budget=15000", tmp.as_uri()],
                   check=True, capture_output=True, timeout=240)
    tmp.unlink()
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} kB, {FIG[0]} figures)")


if __name__ == "__main__":
    main()
