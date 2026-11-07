"""Freeze the spine splits (REVIEW.md 3.1 / 3.2).

Run once:  uv run --project envs/jax python -m harness.build_data

Writes:
  data/splits.json  ID lists per split (committed; NEVER change `test` afterwards)
  data/items.jsonl  normalized items (regenerable from splits.json; gitignored)

Splits:
  train : MBPP train minus the 100 items moved to dev          (274)
  dev   : MBPP validation (90) + prompt (10) + 100 from train   (200)
  test  : MBPP test                                            (500)
  ood   : HumanEval+                                            (164)  # finalized week 3
Every MBPP item also carries MBPP+ tests (the stricter held-out verifier), but ONLY when the
original MBPP reference passes them: MBPP+ changed some specs (signatures, return types, names),
so its tests would mark spec-following MBPP answers wrong. The compatible IDs are frozen in
splits.json["plus_compatible"]. HumanEval+ items whose canonical solution fails are dropped.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from .verifier import verify_many

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEV_FROM_TRAIN = 100
SEED = 0


def mbpp_item(row, split):
    return {
        "id": f"mbpp/{row['task_id']}",
        "split": split,
        "source": "mbpp",
        "text": row["text"],
        "example_test": row["test_list"][0],  # shown in the prompt (standard MBPP practice)
        "setup": row["test_setup_code"] or "",
        "tests": row["test_list"],
        "challenge_tests": row["challenge_test_list"],
        "reference": row["code"].replace("\r\n", "\n"),
        "plus_test": None,
    }


def main():
    mbpp = load_dataset("google-research-datasets/mbpp", "full")
    plus = {f"mbpp/{r['task_id']}": r for r in load_dataset("evalplus/mbppplus")["test"]}
    hep = load_dataset("evalplus/humanevalplus")["test"]

    train = [mbpp_item(r, "train") for r in mbpp["train"]]
    rng = random.Random(SEED)
    moved = set(rng.sample([it["id"] for it in train], DEV_FROM_TRAIN))
    items = []
    for it in train:
        it["split"] = "dev" if it["id"] in moved else "train"
        items.append(it)
    items += [mbpp_item(r, "dev") for r in mbpp["validation"]]
    items += [mbpp_item(r, "dev") for r in mbpp["prompt"]]
    items += [mbpp_item(r, "test") for r in mbpp["test"]]

    for it in items:
        p = plus.get(it["id"])
        if p is not None:
            ti = p["test_imports"] or []
            imports = "\n".join(json.loads(ti) if isinstance(ti, str) else ti)
            it["plus_test"] = (imports + "\n" + p["test"]).strip()

    for r in hep:
        items.append({
            "id": r["task_id"].replace("HumanEval/", "humaneval/"),
            "split": "ood",
            "source": "humaneval",
            "text": r["prompt"],  # signature + docstring; model completes the function
            "entry_point": r["entry_point"],
            "example_test": None,
            "setup": "",
            "tests": [],
            "challenge_tests": [],
            "reference": r["prompt"] + r["canonical_solution"],
            "plus_test": r["test"] + f"\ncheck({r['entry_point']})",
        })

    # Keep plus tests only where the reference solution passes them.
    with_plus = [it for it in items if it["plus_test"]]
    res, vstats = verify_many([(it, it["reference"]) for it in with_plus], timeouts={"basic": 10, "plus": 60})
    bad = {it["id"] for it, r in zip(with_plus, res) if r.get("plus") != "pass"}
    bad_basic = [it["id"] for it, r in zip(with_plus, res) if r.get("basic", "pass") != "pass"]
    assert not bad_basic, f"references failing basic tests: {bad_basic}"
    for it in items:
        if it["id"] in bad:
            it["plus_test"] = None
    items = [it for it in items if not (it["source"] == "humaneval" and it["id"] in bad)]
    print(f"plus tests dropped (reference fails them): {len(bad)}; verifier {vstats}")

    splits = {}
    for it in items:
        splits.setdefault(it["split"], []).append(it["id"])
    for k in splits:
        splits[k].sort(key=lambda s: (s.split("/")[0], int(s.split("/")[1])))

    splits["plus_compatible"] = sorted((it["id"] for it in items if it["plus_test"]),
                                       key=lambda s: (s.split("/")[0], int(s.split("/")[1])))

    DATA.mkdir(exist_ok=True)
    path = DATA / "splits.json"
    if path.exists() and json.loads(path.read_text()) != splits:
        raise SystemExit(f"{path} exists and differs — splits are frozen. Delete it deliberately if you must.")
    path.write_text(json.dumps(splits, indent=1) + "\n")
    with open(DATA / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_plus = {k: sum(1 for it in items if it["split"] == k and it["plus_test"]) for k in splits}
    print({k: len(v) for k, v in splits.items() if k != "plus_compatible"}, "with plus tests:",
          {k: v for k, v in n_plus.items() if k != "plus_compatible"})


if __name__ == "__main__":
    main()
