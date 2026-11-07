"""Score a completions JSONL and log a result row.

  python -m harness.score runs/<name>/completions.jsonl

Input rows: {"id", "sample", "completion", ...}; a sidecar `meta.json` in the same dir
(written by the generator) holds model/decoding config. Outputs, next to the input:
  scored.jsonl   input rows + code + basic/plus statuses
  summary.json   pass@k with 95% bootstrap CIs per verifier + verifier throughput
and appends one row to runs/results.jsonl (REVIEW.md 3.2.8).
"""

import argparse
import json
import subprocess
from pathlib import Path

from .prompts import extract_code
from .stats import summarize
from .verifier import verify_many

ROOT = Path(__file__).resolve().parent.parent


def load_items() -> dict[str, dict]:
    path = ROOT / "data" / "items.jsonl"
    if not path.exists():
        raise SystemExit("data/items.jsonl missing: run `python -m harness.build_data`")
    return {it["id"]: it for it in map(json.loads, path.open())}


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("completions")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    path = Path(args.completions).resolve()
    items = load_items()
    rows = [json.loads(line) for line in path.open()]
    codes = [extract_code(r["completion"], items[r["id"]]) for r in rows]
    results, vstats = verify_many([(items[r["id"]], c) for r, c in zip(rows, codes)], args.workers)

    with open(path.with_name("scored.jsonl"), "w") as f:
        f.writelines(json.dumps({**r, "code": code, **res}) + "\n" for r, code, res in zip(rows, codes, results))
    scored = [{**r, **res} for r, res in zip(rows, results)]

    meta_path = path.with_name("meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    summary = {
        "run": path.parent.name,
        "basic": summarize(scored, "basic"),
        "plus": summarize(scored, "plus"),
        "status_counts": {s: sum(r.get("basic") == s for r in scored)
                          for s in ("pass", "fail", "error", "timeout")},
        "verifier": vstats,
    }
    path.with_name("summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(ROOT / "runs" / "results.jsonl", "a") as f:
        f.write(json.dumps({"scorer_git": git_sha(), "meta": meta, **summary,
                            "per_item": str(path.with_name("scored.jsonl").relative_to(ROOT))}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
