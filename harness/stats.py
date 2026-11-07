"""Unbiased pass@k (Chen et al. 2021) and bootstrap CIs over items (REVIEW.md 3.2).

The two-level (seed x item) paired bootstrap is a week-3 task, built on these pieces.
"""

from collections import defaultdict

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples passes), unbiased, given c of n passed. Requires n >= k."""
    if n < k:
        raise ValueError(f"n={n} < k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def per_item_counts(rows: list[dict], verifier: str) -> dict[str, tuple[int, int]]:
    """{item_id: (n, c)} from scored rows. Rows lacking this verifier are skipped."""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        if verifier in r:
            acc[r["id"]][0] += 1
            acc[r["id"]][1] += r[verifier] == "pass"
    return {k: tuple(v) for k, v in acc.items()}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(rows: list[dict], verifier: str, ks=(1, 2, 4, 8, 16, 32)) -> dict:
    counts = per_item_counts(rows, verifier)
    if not counts:
        return {}
    n_min = min(n for n, _ in counts.values())
    out = {"items": len(counts), "samples_per_item": n_min}
    for k in ks:
        if k == 1 or 2 * k <= n_min:  # REVIEW 3.2.6: report k only up to n/2 (k=1 always)
            vals = np.array([pass_at_k(n, c, k) for n, c in counts.values()])
            lo, hi = bootstrap_ci(vals)
            out[f"pass@{k}"] = {"mean": round(float(vals.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0):
    """Single-level paired bootstrap of mean(b - a) over shared item IDs (single seed)."""
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    lo, hi = bootstrap_ci(d, n_boot, seed=seed)
    return {"items": len(ids), "diff": float(d.mean()), "ci95": [lo, hi]}
