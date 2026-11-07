"""Merge guide/book.json (+ guide/figures.json if present) into guide/book_template.html -> guide/book.html.

  python guide/build_book.py
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
book = json.loads((HERE / "book.json").read_text())
figs_path = HERE / "figures.json"
figs = json.loads(figs_path.read_text()) if figs_path.exists() else {}


def clean_svg(svg: str) -> str | None:
    svg = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<foreignObject[\s\S]*?</foreignObject>", "", svg, flags=re.I)
    svg = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", svg)
    svg = re.sub(r"(href\s*=\s*[\"'])(?!#)[^\"']*", r"\1#", svg)
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        print("  dropped malformed svg:", e)
        return None
    return svg


bad = 0
for ch in book["chapters"]:
    heads = {s["heading"] for s in ch["sections"]} | {"__intro__"}
    out = []
    for f in figs.get(str(ch["week"]), []):
        if f["after"] not in heads:
            print(f"  ch{ch['week']}: unknown anchor {f['after']!r}, appended to last section")
            f["after"] = ch["sections"][-1]["heading"]
        if f["kind"] == "svg":
            f["svg"] = clean_svg(f.get("svg", ""))
            if not f["svg"]:
                bad += 1
                continue
        out.append(f)
    ch["figures"] = out

blob = json.dumps(book, ensure_ascii=False).replace("</", "<\\/")
html = (HERE / "book_template.html").read_text().replace("/*DATA*/", blob)
(HERE / "book.html").write_text(html)
nf = sum(len(c["figures"]) for c in book["chapters"])
words = sum(len(" ".join([c["why"], c["lab"], *(s["md"] for s in c["sections"])]).split()) for c in book["chapters"])
print(f"{len(book['chapters'])} chapters, {words} words, {nf} figures ({bad} dropped) -> {HERE / 'book.html'}"
      f" ({len(html) / 1e6:.1f} MB)")
