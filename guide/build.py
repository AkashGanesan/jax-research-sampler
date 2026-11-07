"""Inject guide/weeks.json into guide/template.html -> guide/weekbook.html.

  python guide/build.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "weeks.json").read_text())
blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
html = (HERE / "template.html").read_text().replace("/*DATA*/", blob)
(HERE / "weekbook.html").write_text(html)
n = sum(len(w["questions"]) for w in data["weeks"])
print(f"{len(data['weeks'])} weeks, {n} questions -> {HERE / 'weekbook.html'}")
