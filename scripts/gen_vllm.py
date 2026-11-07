"""Generate completions with vLLM (torch env). REVIEW.md allows vLLM generation for weeks 0-2.

  uv run --project envs/torch python scripts/gen_vllm.py --split dev --n 8 --seed 0 --out runs/week0_gate_dev
  uv run --project envs/jax python -m harness.score runs/week0_gate_dev/completions.jsonl

Writes <out>/completions.jsonl (id, split, sample, completion, finish_reason, n_tokens) and
<out>/meta.json (full decoding config + git SHA). Never import jax here.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.prompts import build_messages

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--split", required=True, help="dev|train|test|ood (from data/splits.json)")
ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
ap.add_argument("--n", type=int, default=8, help="samples per item")
ap.add_argument("--temperature", type=float, default=0.7)  # Qwen3 non-thinking recommendation
ap.add_argument("--top-p", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--out", required=True)
args = ap.parse_args()

# FlashInfer's sampler JIT-compiles with nvcc, which rejects the host gcc (>14). Set here, not
# only in the flake shellHook, so the script works from any shell.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

ids = json.loads((ROOT / "data/splits.json").read_text())[args.split][: args.limit]
items = {it["id"]: it for it in map(json.loads, (ROOT / "data/items.jsonl").open())}
convs = [build_messages(items[i]) for i in ids]

llm = LLM(args.model, seed=args.seed, gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
          dtype="bfloat16")
sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_tokens, seed=args.seed)
t0 = time.perf_counter()
outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
dt = time.perf_counter() - t0

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
n_tok = 0
with open(out / "completions.jsonl", "w") as f:
    for item_id, o in zip(ids, outs):
        for s, c in enumerate(o.outputs):
            n_tok += len(c.token_ids)
            f.write(json.dumps({"id": item_id, "split": args.split, "sample": s, "completion": c.text,
                                "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}) + "\n")
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
meta = {**vars(args), "generator": "vllm", "chat_template": "qwen3 enable_thinking=False",
        "git": sha, "items": len(ids), "gen_seconds": round(dt, 1),
        "gen_tokens_per_s": round(n_tok / dt, 1), "accelerator": "RTX 4090 (local)", "cost_usd": 0}
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"{len(ids)} items x {args.n} samples, {n_tok} tokens in {dt:.1f}s ({n_tok / dt:.0f} tok/s) -> {out}")
