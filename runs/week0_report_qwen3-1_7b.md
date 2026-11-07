# Week 0 report: qwen3-1_7b

## 1. Spine gate (dev, n=16 per item, basic tests)

| metric | value (95% CI) |
|---|---|
| pass@1 | 49.8% [43.9, 55.8] |
| pass@2 | 55.8% [49.5, 61.9] |
| pass@4 | 61.0% [54.9, 66.9] |
| pass@8 | 65.6% [59.3, 71.8] |

The plan's gate is pass@8 in [10%, 70%]: **pass@8 = 65.6%**. **Gate: PASS**

## 2. Noise floor (test, n=4 per item, 3 sampling seeds)

| seed | basic pass@1 | plus pass@1 (compatible subset) |
|---|---|---|
| s0 | 45.9% [41.8, 49.9] | 55.9% [48.6, 62.9] |
| s1 | 45.6% [41.5, 49.5] | 55.8% [48.6, 63.2] |
| s2 | 45.9% [41.9, 49.9] | 57.0% [49.9, 64.2] |

**Noise floor (basic)**: mean 45.80%, SD across seeds **0.17 pts**. A headline gain must exceed 2 x SD = 0.35 pts (REVIEW 3.2.3).

**Noise floor (plus)**: mean 56.21%, SD across seeds **0.66 pts**. A headline gain must exceed 2 x SD = 1.32 pts (REVIEW 3.2.3).

## 3. SFT pool (train, n=16, verified rejection sampling)

- Train prompts solved at least once: 177/274
- Self-generated verified examples: 440 (distinct, at most 4 per prompt, passing basic and plus tests, finish_reason=stop)
- MBPP reference solutions: 274
- **Total: 714** -> `data/sft_pool_qwen3-1_7b.jsonl` (target 300-1k)
- Unsolved train prompts: 97. Only the reference solution covers these.

