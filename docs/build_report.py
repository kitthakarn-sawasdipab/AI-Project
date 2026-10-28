"""Build docs/report.pdf from docs/report_final.md (cover page plus body with page numbers).

Needs: pip install markdown playwright pymupdf   (uses the installed Chrome, no browser download)
Usage: python docs/build_report.py
"""
import re
from pathlib import Path

import fitz  # PyMuPDF
import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent
TITLE = "Predicting Daily Building Energy Use with Machine Learning"
MEMBERS = ["Kitthakarn Sawasdipab", "Narawit Midsalae", "Arissara Arree", "Nattawut Kamchoo", "Phakchira Nunnum"]
APP = "https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/"
REPO = "https://github.com/kitthakarn-sawasdipab/AI-Project"

CSS = """
@page { size: A4; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #0F172A; margin: 0; }
h1, h2, h3, h4 { color: #0F766E; line-height: 1.25; }
h2 { font-size: 15pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 2px solid #CFE3E0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10px 0 3px; page-break-after: avoid; }
p { margin: 5px 0; text-align: left; }
ul, ol { margin: 4px 0 4px 18px; padding: 0; }
li { margin: 2px 0; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #EEF2F7; padding: 0 3px; border-radius: 3px; }
blockquote { margin: 8px 0; padding: 6px 12px; background: #EAF4F2; border-left: 4px solid #0F766E; }
blockquote p { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 7px 0; font-size: 8.6pt; page-break-inside: auto; }
th { background: #0F766E; color: white; text-align: left; padding: 4px 6px; }
td { padding: 3px 6px; border-bottom: 1px solid #D5E0E8; vertical-align: top; }
tr { page-break-inside: avoid; }
tr:nth-child(even) td { background: #F4F8FB; }
img { max-width: 100%; display: block; margin: 0 auto; }
figure { margin: 8px 0; page-break-inside: avoid; }
figcaption { font-size: 8.4pt; color: #475569; font-style: italic; text-align: center; margin-top: 3px; }
.grid2 { display: flex; gap: 12px; align-items: flex-start; margin: 8px 0; page-break-inside: avoid; }
.grid2 figure { flex: 1 1 0; margin: 0; }
figure.wide img { width: 88%; }
.crop { overflow: hidden; position: relative; width: 100%; aspect-ratio: 1470 / 1090; border: 1px solid #D5E0E8; border-radius: 6px; }
.crop img { position: absolute; width: 130.6%; max-width: none; left: -30.6%; top: -9.2%; margin: 0; }
figure.narrow img { width: 62%; }
.pagebreak { page-break-after: always; }
a { color: #1D4ED8; text-decoration: none; }
"""

COVER = f"""
<html><head><meta charset="utf-8"><style>
body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; margin: 0; color: #0F172A; }}
.band {{ background: linear-gradient(120deg, #0F766E 0%, #2563EB 100%); color: white; padding: 70px 50px 60px; }}
.band .course {{ font-size: 14pt; opacity: .92; letter-spacing: .5px; }}
.band h1 {{ font-size: 32pt; line-height: 1.15; margin: 26px 0 10px; }}
.band .case {{ font-size: 15pt; opacity: .95; }}
.box {{ padding: 36px 50px; }}
.box h3 {{ color: #0F766E; margin: 22px 0 6px; font-size: 12pt; text-transform: uppercase; letter-spacing: 1px; }}
.box p, .box li {{ font-size: 12pt; line-height: 1.6; margin: 2px 0; }}
.box ul {{ list-style: none; padding: 0; margin: 0; }}
.small {{ font-size: 10pt; color: #475569; }}
</style></head><body>
<div class="band">
  <div class="course">IFJC4002 · MC Practical Artificial Intelligence with Python</div>
  <h1>{TITLE}</h1>
  <div class="case">Case 4: Building energy consumption prediction</div>
</div>
<div class="box">
  <h3>Group</h3>
  <ul>{''.join(f'<li>{m}</li>' for m in MEMBERS)}</ul>
  <h3>Course</h3>
  <p>Faculty of Intelligent Computing, Universiti Malaysia Perlis<br>Course coordinator: Ir. Dr. Marni Azira Markom</p>
  <h3>Links</h3>
  <p>Live application: <a href="{APP}">{APP}</a></p>
  <p>GitHub repository: <a href="{REPO}">{REPO}</a></p>
  <h3>Date</h3>
  <p>28 October 2026</p>
  <p class="small" style="margin-top:40px">The data used are the course's synthetic teaching data. Accuracy figures show how well the models recover the simulation, not performance on real buildings.</p>
</div></body></html>
"""

FOOTER = """
<div style="font-size:8px; width:100%; padding:0 18mm; color:#64748B; display:flex; justify-content:space-between;">
  <span>IFJC4002 · Case 4 · Building energy consumption prediction</span>
  <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
</div>
"""


def main() -> None:
    md = (DOCS / "report_final.md").read_text(encoding="utf-8")
    # Python-Markdown needs a blank line before a list that follows a paragraph line
    md = re.sub(r"(?m)^(?![ \t]*(?:[-*] |\d+\. |\||<|>|#))(.+)\n(?=(?:[-*] |\d+\. ))", r"\1\n\n", md)
    html = markdown.markdown(md, extensions=["extra", "sane_lists"])
    body_html = DOCS / "_report_body.html"
    body_html.write_text(f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html}</body></html>', encoding="utf-8")
    cover_html = DOCS / "_report_cover.html"
    cover_html.write_text(COVER, encoding="utf-8")

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.goto(cover_html.as_uri()); pg.wait_for_timeout(500)
        pg.pdf(path=str(DOCS / "_cover.pdf"), format="A4", print_background=True, margin=dict(top="0", bottom="0", left="0", right="0"))
        pg.goto(body_html.as_uri()); pg.wait_for_timeout(1500)
        pg.pdf(path=str(DOCS / "_body.pdf"), format="A4", print_background=True, display_header_footer=True,
               header_template="<div></div>", footer_template=FOOTER,
               margin=dict(top="17mm", bottom="18mm", left="18mm", right="18mm"))
        b.close()

    out = fitz.open()
    for name in ("_cover.pdf", "_body.pdf"):
        part = fitz.open(DOCS / name); out.insert_pdf(part); part.close()
    out.set_metadata({"title": TITLE, "author": ", ".join(MEMBERS), "subject": "IFJC4002 Case 4"})
    out.save(DOCS / "report.pdf", garbage=4, deflate=True)
    print("pages:", len(out), "| size KB:", round((DOCS / "report.pdf").stat().st_size / 1024))
    for f in ("_report_body.html", "_report_cover.html", "_cover.pdf", "_body.pdf"):
        (DOCS / f).unlink()


if __name__ == "__main__":
    main()
