# Running on cloud compute

The same `envs/jax` project runs on cloud machines. The nix flake is optional there; you only need `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <your repo> && cd ai-overview
```

## Cloud GPU (H100/A100, etc.)
```bash
uv sync --project envs/jax          # jax[cuda13], same as local
uv run --project envs/jax python scripts/check_jax.py
```
CUDA 13 wheels need NVIDIA driver >= 580. On older images, edit `envs/jax/pyproject.toml` to `jax[cuda12]` and run `uv lock`.

## Cloud TPU VM (v5e / v6e, TRC, spot)
```bash
uv sync --project envs/jax
uv pip install --python envs/jax/.venv -U "jax[tpu]"   # replaces the CUDA plugin with libtpu
uv run --no-sync --project envs/jax python -c "import jax; print(jax.devices())"
```
`check_jax.py` asserts a GPU backend, so on TPU use the one-liner above. Use `--no-sync` on TPU, or uv will reinstall the CUDA wheels from the lockfile.

## Cost habits
- Develop and debug locally on the 4090. Only send runs to the cloud after a local smoke run of the exact config passes.
- Use spot/preemptible instances, and use Orbax checkpoints with a tested resume path.
- Shut instances down when runs finish. Keep a cost log (see REVIEW.md, section 6).
