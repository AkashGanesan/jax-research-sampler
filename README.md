# ai-overview: ML research study environment

- `modern_ml_research_sampler_12_week_plan.pdf`: the original syllabus
- `REVIEW.md`: multi-agent review and the **revised 14-week JAX-first plan** (follow this one)

## Setup (nix for tools, uv for python and packages)

```bash
direnv allow                       # loads the nix devShell (uv, git, nvtop, ruff) and activates envs/jax/.venv
uv sync --project envs/jax         # primary: JAX/Flax NNX (jax[cuda13], flax, optax, orbax, ...)
uv sync --project envs/torch       # secondary: torch/vLLM/TRL/PEFT, for parity checks only
```
Without direnv: `nix develop`, then the same commands.

Nix deliberately does **not** provide Python or CUDA. On Ubuntu they clash with the system NVIDIA driver. uv manages Python (3.13 for jax, 3.12 for torch), and CUDA comes from the pip wheels.

## Smoke tests
```bash
uv run --project envs/jax   python scripts/check_jax.py    # GPU, NNX train step, 8-fake-device sharding
uv run --project envs/torch python scripts/check_torch.py
```

## Notes
- **GPU memory:** the dev shell sets `XLA_PYTHON_CLIENT_MEM_FRACTION=.75`. For benchmarks and long runs, set it explicitly and record it. Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` only for dev and parity runs.
- **Never run JAX and vLLM on the GPU at the same time**, because both preallocate memory. Never import torch and jax in the same process. Exchange parity data through `.npy`/`.safetensors` files.
- The JAX compilation cache is at `.cache/jax`.
- Adding packages: `uv add --project envs/jax <pkg>`.
- Cloud GPU/TPU: see `cloud/README.md`.

## Eval harness (`harness/`, framework-neutral: stdlib + numpy, never jax/torch)
```bash
uv run --project envs/jax python -m harness.build_data            # once: freezes data/splits.json
uv run --project envs/torch python scripts/gen_vllm.py --split dev --n 16 --out runs/<name>
uv run --project envs/jax python -m harness.score runs/<name>/completions.jsonl
uv run --project envs/jax --group dev python -m pytest -q tests
```
Any generator works (JAX sampler, vLLM, API) as long as it writes `completions.jsonl` rows
`{id, sample, completion}` plus a `meta.json`. Scoring appends a row to `runs/results.jsonl`.

## Week 0
- `scripts/week0_jax_smoke.py`: Qwen3 via Tunix NNX: load, generate, LoRA fwd/bwd (selection rule 3.1a)
- `scripts/week0_overnight.sh`: dev gate (n=16), noise floor (test, 3 seeds), SFT pool, then `runs/week0_report_*.md`
