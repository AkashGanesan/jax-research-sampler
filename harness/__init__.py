"""Framework-neutral eval harness (stdlib + numpy only; never imports jax or torch).

Pipeline: data/items.jsonl --(any generator)--> completions JSONL --score--> scored JSONL + summary.
"""
