"""Smoke test for the secondary PyTorch/vLLM env (parity checks and baselines only).

Run:  uv run --project envs/torch python scripts/check_torch.py
Never import jax in this process.
"""

import torch
import transformers
import vllm  # noqa: F401  (import check only; don't start an engine here)

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, vllm {vllm.__version__}")
assert torch.cuda.is_available(), "CUDA not available to torch"
print("device:", torch.cuda.get_device_name(0))

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(a @ a).all()

cfg = transformers.LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
model = transformers.LlamaForCausalLM(cfg).cuda()
ids = torch.randint(0, 1000, (2, 32), device="cuda")
out = model(input_ids=ids, labels=ids)
out.loss.backward()
print(f"tiny Llama fwd/bwd OK, loss={out.loss.item():.3f}")
print("ALL TORCH CHECKS PASSED")
