#!/usr/bin/env python3
"""Build docs/Team4-Project-Report.pdf from docs/report/report.md (+ screenshots). Needs: pip install markdown ; Chrome."""
import base64, os, re, shutil, subprocess, sys
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "report.md"
OUT_PDF = HERE.parent / "Team4-Project-Report.pdf"
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color:#1a1a1a; font-size:10.5pt; line-height:1.45; }
h1 { font-size:22pt; border-bottom:3px solid #1f5fa8; margin:0 0 6pt; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size:14.5pt; color:#1f5fa8; margin:16pt 0 6pt; page-break-after: avoid; }
h3 { font-size:12pt; margin:12pt 0 4pt; }
code, pre { font-family: Menlo, Consolas, monospace; font-size:8.8pt; }
code { background:#f3f3f3; padding:1px 3px; border-radius:3px; }
pre { background:#f6f8fa; border:1px solid #ddd; border-radius:4px; padding:8px 10px; white-space:pre-wrap; page-break-inside: avoid; }
table { border-collapse:collapse; width:100%; margin:8pt 0 12pt; font-size:9pt; }
th, td { border:1px solid #ccc; padding:4px 6px; vertical-align:top; text-align:left; } th { background:#f0f0f0; }
img { max-width:100%; border:1px solid #ccc; border-radius:4px; }
figure { margin:8pt 0 14pt; page-break-inside: avoid; } figcaption { font-size:8.5pt; color:#555; margin-top:3pt; }
.cover { text-align:center; padding-top:150pt; page-break-after: always; }
.cover h1 { border:none; font-size:28pt; page-break-before: avoid; } .cover p { font-size:13pt; color:#444; }
a { color:#0b5cad; }
"""
md = SRC.read_text()
# ![caption](screenshots/x.png) → embedded base64 figure (missing files → visible note, never a placeholder image)
def repl(m):
    cap, rel = m.group(1), m.group(2)
    p = HERE / rel
    if not p.exists():
        return f'<p style="color:#b00"><b>[screenshot missing: {rel}]</b></p>'
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}"><figcaption>{cap}</figcaption></figure>'
md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Team 4 – Project Report</title><style>{CSS}</style></head><body>{body}</body></html>"
tmp = HERE / "_report.html"; tmp.write_text(html)
chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"), shutil.which("chromium")] if c and os.path.exists(c)), None)
if not chrome:
    print("Chrome not found; open", tmp); sys.exit(1)
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox", f"--print-to-pdf={OUT_PDF}", "--virtual-time-budget=10000", tmp.as_uri()], check=True, capture_output=True, timeout=180)
tmp.unlink()
print("wrote", OUT_PDF, OUT_PDF.stat().st_size // 1024, "kB")
