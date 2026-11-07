#!/usr/bin/env bash
# Week 0 baseline runs (REVIEW.md "Overnight"). On a 4090 with vLLM this takes minutes, not a night.
#   1. dev gate:     dev (200) x 16 samples        -> base pass@8 must be in ~[10%, 70%]
#   2. noise floor:  test (500) x 4 samples x 3 seeds -> SD of pass@1 across seeds
#   3. SFT pool:     train (274) x 16 samples       -> verified rejection-sampled pool (week 1)
# Usage: scripts/week0_overnight.sh [model]   (run inside `nix develop` / direnv)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen3-1.7B}
TAG=$(basename "$MODEL" | tr 'A-Z.' 'a-z_')
gen()   { uv run --project envs/torch python scripts/gen_vllm.py --model "$MODEL" "$@"; }
score() { uv run --project envs/jax python -m harness.score "$1/completions.jsonl" > /dev/null && echo "scored $1"; }

# Separate statements, not `gen && score`: set -e ignores failures on the left of &&.
run() { local out=$1; shift; gen "$@" --out "$out"; score "$out"; }
run runs/week0_gate_dev_$TAG --split dev --n 16 --seed 0
for s in 0 1 2; do
  run runs/week0_noise_test_${TAG}_s$s --split test --n 4 --seed $s
done
run runs/week0_sftpool_train_$TAG --split train --n 16 --seed 0
uv run --project envs/jax python scripts/week0_report.py "$TAG"
