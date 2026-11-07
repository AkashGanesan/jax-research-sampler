"""Week 0 model smoke test (JAX): load Qwen3 via Tunix NNX, generate, LoRA fwd/bwd.

Run:  uv run --project envs/jax python scripts/week0_jax_smoke.py [--model Qwen/Qwen3-1.7B]

Selection rule from REVIEW.md 3.1(a): an off-the-shelf NNX implementation loads,
generates, and runs a LoRA forward/backward on the 4090.
"""

import argparse
import dataclasses
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # dev run: see real usage

import jax
import jax.numpy as jnp
import optax
import qwix
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params

CONFIGS = {
    "Qwen/Qwen3-1.7B": qwen3_model.ModelConfig.qwen3_1p7b,
    "Qwen/Qwen3-0.6B": qwen3_model.ModelConfig.qwen3_0p6b,
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B", choices=list(CONFIGS))
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--seq", type=int, default=512)
args = ap.parse_args()


def peak_gb():
    s = jax.devices()[0].memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 2**30


t0 = time.perf_counter()
path = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt"])
cfg = CONFIGS[args.model]()
model = qwen3_params.create_model_from_safe_tensors(path, cfg, mesh=None, dtype=jnp.bfloat16)
tok = AutoTokenizer.from_pretrained(path)
print(f"[load] {args.model} in {time.perf_counter() - t0:.1f}s, peak {peak_gb():.2f} GB")

# --- 1. generation -----------------------------------------------------------
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a Python function `is_prime(n)`. Reply with only a code block."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
sampler = sampler_lib.Sampler(
    transformer=model, tokenizer=tok,
    cache_config=sampler_lib.CacheConfig(
        cache_size=1024, num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim),
)
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
for label in ("compile+run", "warm"):
    t = time.perf_counter()
    out = sampler([prompt], max_generation_steps=256, eos_tokens=eos, temperature=0.0)
    print(f"[generate:{label}] {time.perf_counter() - t:.1f}s")
print("-" * 60 + "\n" + out.text[0].strip() + "\n" + "-" * 60)

# --- 2. LoRA forward/backward --------------------------------------------------
# Tunix's DECODER remat breaks the KV-cache sampler, so reload a remat copy for training.
del sampler, model
cfg = dataclasses.replace(cfg, remat_config=qwen3_model.RematConfig.DECODER)
model = qwen3_params.create_model_from_safe_tensors(path, cfg, mesh=None, dtype=jnp.bfloat16)
provider = qwix.LoraProvider(
    module_path=r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    rank=args.lora_rank, alpha=2.0 * args.lora_rank,
)
lora_model = qwix.apply_lora_to_model(model, provider, rngs=nnx.Rngs(0), **model.get_model_input())
n_lora = sum(x.size for x in jax.tree.leaves(nnx.state(lora_model, nnx.LoRAParam)))
n_all = sum(x.size for x in jax.tree.leaves(nnx.state(lora_model, nnx.Param)))
print(f"[lora] trainable {n_lora / 1e6:.1f}M of {n_all / 1e6:.0f}M params")

B, L = args.batch, args.seq
ids = jax.random.randint(jax.random.key(0), (B, L), 0, cfg.vocab_size)
mask = jnp.ones((B, L), bool)
positions = jnp.broadcast_to(jnp.arange(L), (B, L))
attn = jnp.tril(jnp.ones((L, L), bool))[None].repeat(B, 0)

opt = nnx.Optimizer(lora_model, optax.adamw(1e-4), wrt=nnx.LoRAParam)


@nnx.jit
def train_step(m, opt, ids):
    def loss_fn(m):
        logits, _ = m(ids, positions, None, attn)
        logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), -1)
        return -jnp.take_along_axis(logp, ids[:, 1:, None], -1).mean()

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))(m)
    opt.update(m, grads)
    return loss


for i in range(3):
    t = time.perf_counter()
    loss = float(train_step(lora_model, opt, ids))
    print(f"[train step {i}] loss={loss:.3f} {time.perf_counter() - t:.2f}s")
n_opt = sum(x.size for x in jax.tree.leaves(nnx.state(opt, nnx.OptState)))
print(f"[optimizer] state elements {n_opt / 1e6:.1f}M (expect ~2x LoRA = {2 * n_lora / 1e6:.1f}M)")
print(f"[memory] peak {peak_gb():.2f} GB (B={B}, L={L}, decoder remat, full-vocab fp32 logits)")
print("SMOKE TEST PASSED")
