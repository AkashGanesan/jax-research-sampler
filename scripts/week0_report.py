"""Summarize week-0 runs: dev gate, noise floor, SFT pool. Writes runs/week0_report_<tag>.md
and data/sft_pool_<tag>.jsonl.

  uv run --project envs/jax python scripts/week0_report.py qwen3-1_7b
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.prompts import build_messages

tag = sys.argv[1]
runs = ROOT / "runs"
MAX_PER_ITEM = 4  # cap self-generated solutions per prompt so easy items don't dominate SFT


def load(name):
    return json.loads((runs / name / "summary.json").read_text())


def fmt(m):
    return f"{100 * m['mean']:.1f}% [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}]"


lines = [f"# Week 0 report: {tag}\n"]

# 1. dev gate --------------------------------------------------------------------
g = load(f"week0_gate_dev_{tag}")
p8 = g["basic"]["pass@4"] if "pass@8" not in g["basic"] else g["basic"]["pass@8"]
k = 8 if "pass@8" in g["basic"] else 4
ok = 0.10 <= p8["mean"] <= 0.70
n_dev = json.loads((runs / f"week0_gate_dev_{tag}" / "meta.json").read_text())["n"]
lines += [f"## 1. Spine gate (dev, n={n_dev} per item, basic tests)\n",
          "| metric | value (95% CI) |", "|---|---|",
          *[f"| {m} | {fmt(v)} |" for m, v in g["basic"].items() if m.startswith("pass@")],
          f"\nThe plan's gate is pass@8 in [10%, 70%]: **pass@{k} = {100 * p8['mean']:.1f}%**"
          + ("" if k == 8 else " (a lower bound on pass@8, since pass@k is reported only for k <= n/2)") + ". "
          f"**Gate: {'PASS' if ok else 'FAIL'}**"
          + ("" if k == 8 else ". Run with --n 16 to report pass@8 exactly.") + "\n"]

# 2. noise floor -----------------------------------------------------------------
seeds = sorted(runs.glob(f"week0_noise_test_{tag}_s*/summary.json"))
lines += ["## 2. Noise floor (test, n=4 per item, 3 sampling seeds)\n",
          "| seed | basic pass@1 | plus pass@1 (compatible subset) |", "|---|---|---|"]
vals = {"basic": [], "plus": []}
for p in seeds:
    s = json.loads(p.read_text())
    vals["basic"].append(s["basic"]["pass@1"]["mean"])
    vals["plus"].append(s["plus"]["pass@1"]["mean"])
    lines.append(f"| {p.parent.name[-2:]} | {fmt(s['basic']['pass@1'])} | {fmt(s['plus']['pass@1'])} |")
if len(seeds) >= 2:
    for v in ("basic", "plus"):
        lines.append(f"\n**Noise floor ({v})**: mean {100 * statistics.mean(vals[v]):.2f}%, "
                     f"SD across seeds **{100 * statistics.stdev(vals[v]):.2f} pts**. "
                     f"A headline gain must exceed 2 x SD = {200 * statistics.stdev(vals[v]):.2f} pts (REVIEW 3.2.3).")
lines.append("")

# 3. SFT pool --------------------------------------------------------------------
items = {it["id"]: it for it in map(json.loads, (ROOT / "data/items.jsonl").open())}
scored = [json.loads(line) for line in (runs / f"week0_sftpool_train_{tag}/scored.jsonl").open()]
norm = lambda c: re.sub(r"\s+", " ", c).strip()
pool, seen, solved = [], set(), set()
for r in scored:
    if r.get("basic") != "pass" or r.get("plus", "pass") != "pass" or r.get("finish_reason") != "stop":
        continue
    key = (r["id"], norm(r["code"]))
    if key in seen or sum(1 for p in pool if p["id"] == r["id"]) >= MAX_PER_ITEM:
        continue
    seen.add(key)
    solved.add(r["id"])
    pool.append({"id": r["id"], "source": "self", "messages": build_messages(items[r["id"]]) +
                 [{"role": "assistant", "content": r["completion"].strip()}]})
train_ids = json.loads((ROOT / "data/splits.json").read_text())["train"]
for i in train_ids:  # MBPP reference solutions (verified in build_data)
    pool.append({"id": i, "source": "reference", "messages": build_messages(items[i]) +
                 [{"role": "assistant", "content": f"```python\n{items[i]['reference'].strip()}\n```"}]})
with open(ROOT / f"data/sft_pool_{tag}.jsonl", "w") as f:
    for p in pool:
        f.write(json.dumps(p) + "\n")
n_self = sum(p["source"] == "self" for p in pool)
lines += ["## 3. SFT pool (train, n=16, verified rejection sampling)\n",
          f"- Train prompts solved at least once: {len(solved)}/{len(train_ids)}",
          f"- Self-generated verified examples: {n_self} (distinct, at most {MAX_PER_ITEM} per prompt, "
          f"passing basic and plus tests, finish_reason=stop)",
          f"- MBPP reference solutions: {len(train_ids)}",
          f"- **Total: {len(pool)}** -> `data/sft_pool_{tag}.jsonl` (target 300-1k)",
          f"- Unsolved train prompts: {len(train_ids) - len(solved)}. Only the reference solution covers these.\n"]

out = runs / f"week0_report_{tag}.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
