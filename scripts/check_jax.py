"""Smoke test for the JAX/Flax env: GPU, jit, Flax NNX + optax training, sharding.

Run:  uv run --project envs/jax python scripts/check_jax.py
The sharding check runs in a subprocess with 8 fake CPU devices.
"""

import os
import subprocess
import sys
import time

if os.environ.get("_SHARDING_CHILD"):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    devs = jax.devices()
    assert len(devs) == 8, devs
    mesh = jax.make_mesh((4, 2), ("data", "model"))
    x = jax.random.normal(jax.random.key(0), (64, 128))
    w = jax.random.normal(jax.random.key(1), (128, 256))
    xs = jax.device_put(x, NamedSharding(mesh, P("data", None)))
    ws = jax.device_put(w, NamedSharding(mesh, P(None, "model")))
    out = jax.jit(lambda a, b: jnp.tanh(a @ b).sum())(xs, ws)
    ref = jnp.tanh(x @ w).sum()
    assert jnp.allclose(out, ref, rtol=1e-4), (out, ref)
    print(f"  sharded == unsharded on 4x2 mesh: {float(out):.4f} vs {float(ref):.4f}")
    sys.exit(0)

import jax
import jax.numpy as jnp
import optax
from flax import nnx

print(f"jax {jax.__version__}, backend={jax.default_backend()}, devices={jax.devices()}")
assert jax.default_backend() == "gpu", "expected GPU backend"

# 1. jitted bf16 matmul throughput
n = 8192
a = jnp.ones((n, n), jnp.bfloat16)
mm = jax.jit(lambda x: x @ x)
mm(a).block_until_ready()
t = time.perf_counter()
for _ in range(10):
    r = mm(a)
r.block_until_ready()
dt = (time.perf_counter() - t) / 10
print(f"bf16 {n}x{n} matmul: {2 * n**3 / dt / 1e12:.1f} TFLOP/s")


# 2. Flax NNX MLP + optax on a toy regression
class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(16, 128, rngs=rngs)
        self.l2 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


model = MLP(nnx.Rngs(0))
opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
X = jax.random.normal(jax.random.key(0), (512, 16))
Y = jnp.sin(X.sum(-1, keepdims=True))


@nnx.jit
def step(model, opt, x, y):
    loss, grads = nnx.value_and_grad(lambda m: ((m(x) - y) ** 2).mean())(model)
    opt.update(model, grads)
    return loss


first = float(step(model, opt, X, Y))
for _ in range(300):
    last = float(step(model, opt, X, Y))
print(f"NNX MLP loss: {first:.4f} -> {last:.4f}")
assert last < first * 0.5, "loss did not decrease"

# 3. sharding on 8 fake CPU devices (subprocess so XLA_FLAGS applies cleanly)
env = dict(os.environ, _SHARDING_CHILD="1", JAX_PLATFORMS="cpu",
           XLA_FLAGS="--xla_force_host_platform_device_count=8")
subprocess.run([sys.executable, __file__], env=env, check=True)
print("ALL JAX CHECKS PASSED")
