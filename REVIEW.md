# Revised Review: Modern ML Research Sampler, JAX-first, 14 weeks

> **Environment used in this repo:** `envs/jax` (`jax[cuda13]`, flax NNX, optax, orbax) is the primary env; `envs/torch` (torch/vLLM/TRL/PEFT) is for parity checks only. See `README.md`. Where week 0 below says "the CUDA extra current at install time", that is `jax[cuda13]` here.

*This revision replaces the earlier review. The plan uses JAX + Flax NNX first. It runs on one RTX 4090 with cloud bursts. Weeks 0-2 are light, and weeks 3-13 are full-time. Date: 2026-11-06. Anything I could not check is marked "(verify)". Prices, quotas, dataset sizes and licenses change often, so check every number in section 6 and every dataset in section 3.1 before you spend money or time.*

---

## 1. Verdict on the original plan

The original syllabus has a strong core:
- Experiments are the curriculum.
- One model and one domain run through every week.
- The comparisons are controlled: SFT vs DPO, sparse vs dense reward, outcome-only vs intermediate reward, and data selection at a fixed budget.
- The week 5 RLVR metrics are the right ones: pass@k, OOD and reward hacking.

The earlier review found these structural gaps, and they still apply:
- No setup week, and no eval harness until week 7.
- No statistics protocol, so most A-vs-B claims would sit inside the noise.
- No mechanism for the actual goal, which is choosing a frontier area.
- An implicit PyTorch/TRL/vLLM stack.

The new constraints add a new problem. If you make JAX primary and change nothing else, the plan breaks in week 4. JAX has no off-the-shelf rollout engine like vLLM that you can rely on for a 4090, and HF transformers no longer ships Flax models. A JAX-first plan only works if it **budgets time to build the LLM infrastructure itself**: the model, loader, KV-cache sampler, LoRA, log-probs and RL loop. The full-time weeks make that affordable, and it becomes the plan's main career asset.

A second review pass found four problems, all fixed below:
- **Load.** Weeks 0, 3, 7 and 9 were overloaded, eval compute was not budgeted, and the budget caps contradicted the estimates.
- **RL correctness.** There were errors in the reference policy, KL placement, sampler log-prob distribution, on-policy staleness, LoRA optimizer filtering and the tool-mask test.
- **Statistics.** Seed variance was ignored, and pass@k at k=n is unreliable.
- **Goal.** The capstone was not connected to the area decision.

## 2. What changed in this revision and why

| Change | Why |
|---|---|
| **JAX/Flax NNX is the default for all LLM work.** PyTorch/vLLM is used only for: (a) logit/loss parity references, (b) a vLLM throughput reference, (c) weeks 0-2 generation if no JAX path works in the timebox, (d) the fallback rule when a JAX path would waste more than ~1 day. It lives in a **separate pinned environment** and never shares a process with JAX. | This is your instruction. JAX also gives you single-process RL: the sampler and trainer share the same arrays, so there is no weight sync. It gives you pure-function interventions for interp and a direct path to TPU sharding. Separate envs avoid torch/vLLM/JAX CUDA-wheel conflicts. |
| **Week 0 is narrowed.** It uses an existing dataset with standard splits and tests, so no splits are built by hand. The spine model is chosen by "an NNX LoRA-trainable implementation exists" plus week-3 simplicity, with a second choice picked in advance. The SFT data source is named. Paired-test code, the OOD slice and the general slice move to weeks 1 and 3. | 6 h covers setup, harness core and an overnight gate, and nothing more. If the gate fails, it costs one overnight run, not a redesign. Weeks 1-3 always use the same model. |
| **New week 3: "Build your JAX LLM stack"**, with a **staged sampler**. The MVP is due Wednesday. Week 4's first 1.5 days are formally reserved for week-3 overflow. The full vLLM comparison and the measured memory table move to week 11. | Every later week depends on this. It is the biggest item you build yourself and the strongest portfolio signal for JAX/TPU roles, so it gets time instead of a fallback that throws it away. |
| **Throughput gate at the end of week 4.** Rollout tokens/s at the week-5 config must be within 4× of a vLLM reference, or 300 steps must fit in one night. If not, there is a planned response. | Weeks 5-8 are only feasible on the 4090 if the sampler is fast enough. Without the gate you would find out in week 5. |
| **RL correctness fixes:** reference = merged SFT weights with a fresh RL adapter at traced scale 0. KL placement (reward penalty vs loss) is pre-registered with a gradient test. RL rollouts use untruncated temperature-T log-probs that match the trainer. On-policy vs one-step-async is stated explicitly. Grads and optimizer state are filtered to LoRA params and kept in fp32. Training is microbatched with selective donation. | Each of these silently invalidates results or causes OOM. |
| **The extra time buys depth, not topics.** The topic list is unchanged: SFT, DPO, RL, RLVR, agents, interp, systems, data and capstone. Instead, every full-time week gets 3 seeds on each headline comparison (5 in the capstone headline), a short LR sweep before each RL run, high-k pass@k curves and one infrastructure component you build yourself. | Your goal is an area decision based on evidence. Seeds and depth make the evidence trustworthy. More topics would only make it wider. |
| **Old weeks 6+7 (agent build + agent eval) are merged into one full-time week.** v1 multi-turn re-prefills the full token-ID context each turn. In-cache continuation is a stretch goal. | Re-prefill is cheap at 2-3 turns and ≤2k tokens, and it reuses the tested path. A ragged per-sequence cache is a multi-day job. |
| **Systems week moves to real hardware.** It runs on a TPU v5e-8/v6e-8 slice (spot or TRC) for 1-2 days: DP, FSDP, TP and an MFU scaling curve. Sharding is first tested on 8 fake CPU devices. The cloud dry run allocates the same slice type ahead of time. | Fake devices check correctness, not performance. Testing the slice early proves that you can get the quota. |
| **Cloud policy and cost log, with a GPU dry run in weeks 4-5.** Week-6 bursts run on rented NVIDIA GPUs only. TPU is first used after a TPU smoke test. | The first paid burst should not also be the first time the stack runs remotely. |
| **Length: 14 plan weeks (0-13), ~460 h.** Weeks 0-2 are ~18 h, weeks 3-13 are 11 × 40 h. Holiday weeks are **non-plan weeks**, so the calendar may stretch to 15-16 weeks. Week 9 is a real buffer, and weeks 5 and 8 each get a half-day buffer. | The decision lands around mid-February 2027 if you start 2026-11-09 with no time off, or late February with 2 holiday weeks. That is roughly 2.5× the hours of the old 16-weekend plan. Adding weeks would add topics. Adding seeds and infrastructure adds credibility. |
| **Statistics protocol tightened.** Pre-registration, ≥3 seeds, and a **two-level (seed × item) paired bootstrap**. A headline claim requires the sign to agree across seeds. The noise floor is defined numerically. pass@k uses n ≥ 2k. | Item-only bootstraps are overconfident when seed variance dominates. pass@k at k=n is the least reliable point on the curve. |
| **Scorecard and capstone are linked.** Rows are filled at weeks 2, 6, 8, 10, 11 and 12. The capstone is chosen from the top-2 areas, and the tiebreak pilot runs inside week 12. The week 13 memo cites the capstone as the deciding row. A "JAX/TPU leverage" criterion and a "≤$100/month + 4090" criterion are added. | This ties the plan's largest block of evidence to the decision it exists to make. |
| **Three required public write-ups:** #1 the RLVR note, #2 "JAX KV-cache sampler + single-process RL on a 4090", #3 the capstone plus decision memo. | The infrastructure artifact is your clearest engineering signal, so it should not be optional. |
| **Budget reconciled.** The planned upper bound is ~$400 against a $500 hard cap. Weeks 11 and 13 are pre-approved (week 11 only on spot or TRC). $150 is reserved for week 13. There is an explicit cut order. | The caps and estimates used to contradict each other. |

What is carried over in substance: the eval harness, noise floor, per-item JSONL, the DPO data-matched control arm, the RL logging list and acceptance criteria, the stricter held-out verifier for hacking, interp control tasks, the sharding equivalence test, buffers, public write-ups and the spine gate.

## 3. Working rules

### 3.1 Spine (locked in week 0)

**Domain.** Use an **existing dataset that ships standard splits and tests**. Do not build splits by hand in week 0. Freeze the splits by item-ID lists in the repo.
- **Code (preferred):** MBPP with its standard train/validation/test splits (verify sizes; test is ~500).
  - EvalPlus MBPP+ extended tests serve as the stricter held-out verifier (verify availability, license and the overlap with MBPP test).
  - HumanEval / HumanEval+ is the OOD set (verify license).
  - Dev = MBPP validation (small, ~90, verify) plus a fixed slice carved from train, for ~200 total.
  - If RL needs more training prompts than MBPP train provides, add a format-compatible set that has tests, such as APPS introductory (verify), decontaminated against test.
- **Math:** GSM8K train/test (verify sizes), with a numeric checker plus a stricter answer checker (normalization, units, and rejection of multiple answers) as the held-out verifier. OOD comes from a different math set (verify).
- **General-capability slice (~200):** a small fixed subset of a standard multiple-choice benchmark (verify license). It is built in week 1.

**Verifier.** It must take under 1 s/sample on CPU. Measure its throughput in week 0.
- Create the pool with `multiprocessing.get_context("spawn")` or `"forkserver"`, or start it before importing jax.
- Workers never import jax.
- Each test runs in its own subprocess with `resource` limits and a timeout.

**SFT source (named in week 0).**
- Code: MBPP train reference solutions (~374, verify), plus an overnight, verified, rejection-sampled pool from the base model on train prompts. Target 300-1k examples.
- Math: GSM8K train solutions. Target 1-5k.

**Model.** Use one ~0.5-1.5B instruct model for all weeks. Candidates:
- **Gemma 3 1B-it.** It has official JAX code, and Tunix/Gemma JAX implementations exist as references (verify NNX vs Linen and GPU support). It uses local/global attention. Its vocab is 262k, so chunked log-probs are mandatory.
- **Qwen2.5-0.5B/1.5B-Instruct** or **Qwen3-0.6B/1.7B** (verify). These are standard dense GQA models and simpler to implement yourself. Note the spurious-reward caveat for this family.
- **Selection rule:** (a) an off-the-shelf **NNX** implementation of *that same model* loads and runs a LoRA forward/backward on the 4090 within the week-0 timebox, and (b) it is simple to implement yourself in week 3. Pick the second choice in advance. The same model is used in weeks 1, 2 and 3, so week 3's re-run compares like with like.

**Gate.** Base pass@8 on dev must fall between ~10% and 70%. If it does not, switch to the second-choice model, or to the other domain, the next night.

**Training.**
- Base weights stay in bf16.
- **LoRA A/B and optimizer state are fp32**, cast to bf16 inside the matmul.
- Gradients and the optimizer are **filtered to LoRA params only**.
- Use gradient checkpointing and cap generations at 512-1024 tokens.

**Reference policies.**
- **DPO:** reference = the SFT model. Precompute reference log-probs once, before DPO training. DPO then continues the SFT adapter.
- **RL:** merge the SFT adapter into the base (merge in fp32, cast to bf16, re-run parity), then add a fresh RL adapter. Reference = merged weights with the RL adapter at scale 0.
  - `scale` is a **traced array argument**, not a Python attribute, so toggling it does not cause retracing.
  - Test that reference log-probs equal the SFT model's log-probs.
  - The other option is two adapters: a frozen SFT one and a trainable RL one.

**Larger models** (Gemma 3 4B / Qwen ~3-7B class) appear only as a teacher, an inference benchmark or a capstone scale check.

**SystemVerilog** is a capstone transfer target only, and only if base pass@8 is at least ~10%.

### 3.2 Statistics protocol
1. **Splits are frozen in week 0** by ID list: train / dev (~200) / test (≥300, ideally 500) / OOD (~100-200, finalized in week 3 Mon) / general-capability slice (~200, built in week 1). The test split never changes.
2. **CIs and paired tests.**
   - Every number gets a 95% bootstrap CI over items.
   - A-vs-B comparisons with training seeds use a **two-level paired bootstrap**: resample seeds within each arm, then resample the item IDs shared across arms. Report it together with the per-seed paired differences.
   - McNemar applies **only** to single-sample binary outcomes (for example greedy pass@1). For per-item pass rates estimated from many samples, use the paired bootstrap or a Wilcoxon signed-rank test.
   - There is no fixed "ignore below X pts" rule. Interpret only the paired CI. For scale: one proportion at n=300 and p=0.5 has a CI of about ±5.7 pts, an unpaired difference is about ±8, and a paired difference can be much narrower.
3. **Noise floor** = the SD of the headline metric across 3 sampling seeds on the same checkpoint. Re-measure it whenever the sampler changes (week 3). A headline claim needs all three of these:
   - the gain exceeds 2× the noise floor
   - the two-level paired CI excludes 0
   - the sign of the paired difference is the same on every training seed
4. **Seeds:** ≥3 training seeds for every headline comparison, shown as per-seed curves plus the mean. The capstone headline gets 5 seeds and its ablation arm gets 3. Single-seed results are labeled "exploratory".
5. **Pre-registration:** before each headline run, commit a short note in the repo before launch. It contains the hypothesis, arms, metric, n, the decision rule, the compute plan (runs × measured hours, accelerator, cost), and, for RL, the KL placement and the on-policy/async choice.
6. **Unbiased pass@k** (Chen et al.) with **n ≥ 2k samples per item**, so you report k only up to n/2.
   - High-k curves: n=64 on a fixed 200-item test subset, with k ≤ 32.
   - k=64 needs n=128 on that subset, and only if the budget allows.
   - Report the bootstrap CI at every k.
7. If you test more than ~5 comparisons in one week, apply a Holm correction or label them exploratory.
8. **Every result row logs:**
   - git SHA and config hash
   - chat template and decoding config (T, top-p)
   - sampler version
   - seed(s)
   - allocator settings
   - `accelerator_hours` and `cost_usd`
   - the path to the per-item JSONL

### 3.3 Weekly rhythm
- **Light weeks (0-2, 5-7 h):** two weekday evenings (~1.5 h each) plus one weekend block (~3-4 h). Long runs go overnight. If something isn't working by the start of the weekend block, ship the smaller deliverable.
- **Full-time weeks (3-13, ~40 h):**
  - **Mon:** reading (~2 h), pre-registration, build.
  - **Tue-Wed:** build and test, then launch the main runs by Wednesday evening.
  - **Thu:** seeds and arms run, and analysis starts.
  - **Fri:** analysis, write-up, scorecard if due, and slack (~4 h).
  - Main runs go overnight. Weekends are off by default and serve only as a last-resort overflow valve. Planned slip goes to the explicit buffers: week 4 Mon-Tue AM (for week 3), the Friday afternoons of weeks 5 and 8, and week 9.
- **Wednesday rule:** if the core experiment isn't producing sane curves by Wednesday evening, cut scope (fewer arms, a smaller subset), ship a smaller deliverable and write up the failure.
- **Infrastructure timebox and fallback rule.** Every infrastructure component has an acceptance test and a timebox. If a component is more than ~1 working day over its timebox, use the off-the-shelf path for that week, record why, and add the build to the backlog. Off-the-shelf means Tunix, MaxText, vLLM or TRL, as applicable.
  - **Exception for the sampler:** its fallback is the staged MVP, not vLLM. vLLM cannot serve a single-process JAX RL loop without cross-framework weight sync.
  - If even the MVP sampler isn't working by week 4 Tue, run weeks 4-6 RL on a TRL/vLLM path in the torch env (record why), and keep building the JAX sampler as background work.

### 3.4 Compute and cloud policy
- **Default local.** Use the 4090 for development, single-seed runs, overnight runs and most evaluation.
- **Go to cloud only when one of these holds:**
  - (a) seed or arm fan-out would occupy the 4090 for more than ~2 nights
  - (b) you need real multi-device sharding
  - (c) the model is too large for 24 GB (training above ~2-4B with LoRA, inference above ~4-7B in bf16)
  - (d) a deadline-critical run, such as the capstone
- **Dry runs come before the first burst on each hardware type:**
  - **GPU:** weeks 4-5. Rent one GPU for under 1 h and run `launch.sh` end to end with a 20-step RL run, a kill and a resume.
  - **TPU:** week 9, on the **same slice type** you will use in week 11. If TRC TPUs arrive early and you want them before week 9, do a 1-2 h TPU smoke test of the RL loop in week 5 first, with log-prob parity vs the 4090 at `highest` matmul precision.
  - Week-6 bursts use rented NVIDIA GPUs only.
- **Burst checklist (mandatory):**
  1. The same config ran locally at small scale on the same pinned JAX version.
  2. Orbax checkpoints at least every 20 min to a bucket in the same region, and resuming from a killed run has been tested.
  3. The persistent compilation cache is enabled.
  4. Use spot/preemptible wherever resume works.
  5. Auto-shutdown on idle or at a hard time limit, plus a billing budget alert.
  6. Pull the results and **delete the VM/TPU** the same day.
  7. Fill in `cost_usd` and `accelerator_hours` in the results row.
  8. When renting an H100, specify SXM (~3.35 TB/s), not PCIe (~2 TB/s).
- **Quota and TRC:**
  - Apply for TRC in week 0. When it is approved, ask whether the window can be timed or extended to cover about weeks 9-13 (verify policy; grants are time-limited, often ~30 days, verify).
  - In week 0, also file GCP quota requests for the exact week-11 slice type and region. Default multi-chip quota is often zero (verify).
- **Budget:**
  - Hard cap ~$500 without TRC. The **planned upper bound is ~$400**, which leaves about 20% margin.
  - Per-burst cap ~$150.
  - Weeks 11 and 13 are pre-approved. Week 11 is approved only on spot or TRC. Week 13 has $150 reserved from the start.
  - **Cut order when spending nears the cap:** (1) week-8 burst seeds, which then become 2 local seeds labeled exploratory; (2) week-5 extra seeds; (3) the week-6 n=128 high-k eval; (4) the week-12 12B teacher, which is already off by default.
  - Keep a running `cost_usd` total in the README and check it at every scorecard week.

### 3.5 Public work and retention
- One public monorepo from week 0. The README results table links each plot to the exact command and config.
- Each week keeps: the best LoRA adapter (Orbax plus a merged safetensors export), the config, per-item JSONL, and one note on "a behavior change worth probing in week 10".
- **Required public write-ups:**
  - **#1, the RLVR note.** Drafted at the end of week 6 and published in week 9.
  - **#2, "JAX KV-cache sampler + single-process RL on a 4090".** Drafted at the end of week 5 and published in week 9. It includes:
    - parity results
    - tokens/s vs the vLLM reference from the week-4 gate
    - the memory budget
    - a reproducible README command for each number
    - an appended update in week 11 with the full inference table and the sharding/MFU curve
  - **#3, the capstone plus decision memo.** Week 13.
- Optional extra: an interp post.

## 4. Week-by-week plan

Weeks are indexed, not dated. If you start Mon 2026-11-09, plan weeks 6-7 fall on Dec 21 - Jan 3. **Holiday weeks are non-plan weeks:** pause the index and resume afterwards. Do not spend buffer week 9 on a holiday. The calendar then becomes about 15-16 weeks.

| Wk | Hours | Goal | Hands-on (core) | Deliverable | Compute |
|---|---|---|---|---|---|
| **0** | 6-7 | Setup + measurement | Two pinned uv envs (JAX; torch/vLLM for parity only). TRC application + TPU quota request. Existing dataset with standard splits/tests, frozen by ID. Spine model chosen by the 3.1 rule (1 h NNX smoke test incl. LoRA fwd/bwd), second choice picked in advance. Harness core: spawn-based CPU verifier pool, unbiased pass@k, per-item JSONL, measured verifier throughput. SFT source named. Overnight: dev gate, then noise floor (3 sampling seeds). | Lockfiles, baseline table + noise floor, named SFT source | Local |
| **1** | 5-7 | SFT + LoRA | `nnx.LoRA` SFT on the available verified examples (300-1k code / 1-5k math), LoRA-filtered optimizer, fp32 LoRA state. Masked log-prob function + pytests. Paired bootstrap code. General-capability slice + forgetting. Ablation attn-only vs all-linear (overnight, optional). | Base vs SFT (target + forgetting, paired CIs) | Local |
| **2** | 5-7 | Preference learning | DPO derivation (written). On-policy verifier-labeled pairs. 3 arms: SFT / SFT + chosen-only / SFT + DPO (reference = SFT, precomputed), β=0.1, single seed. **Scorecard row 1.** | Controlled comparison: metric, length, margins, chosen log-prob | Local |
| **3** | 40 | **Build your JAX LLM stack** | Mon: OOD slice, two-level bootstrap, reading. NNX model + safetensors loader + fp32 parity at `highest` precision. **Sampler MVP by Wed** (fixed max_len, one bucket, temperature, per-seq EOS, untruncated log-probs). Your own LoRA (traced scale, merge/export). Chunked log-prob (scan + remat). Orbax + resume test. Thu-Fri: buckets, eval-only top-k/top-p, zero-recompile assertion. Memory budget (arithmetic). Overnight: noise floor re-measure. | Stack with tests, parity report, sampler MVP+, noise floor v2 | Local |
| **4** | 40 | RL foundations → LLM pipeline | **Mon-Tue AM: week-3 overflow (reserved).** Tue PM-Wed AM: toy NNX RL (no-baseline / leave-one-out / LOO + KL), `vmap` over 16-32 seeds. Wed: GRPO/RLOO advantage + clipped loss + pre-registered KL placement, with tests. Thu-Fri: single-process loop (strict on-policy by default), microbatched gradient accumulation, donation split, memory test at week-5 config, 50-step smoke run, **throughput gate**, GPU dry run if time. Nights: weeks 1-2 headline re-run on your stack, 3 seeds. | Toy curves with seed bands. Tested losses. Working pipeline + smoke log. Gate result. Weeks 1-2 with 3 seeds | Local (+ <1 h GPU dry run) |
| **5** | 40 | First real LLM RL run | GPU dry run (if not done in week 4). LR sweep (3 values × ~100 steps). Main GRPO (or RLOO, pre-chosen) run with full logging, early-kill criteria, 3 seeds. Then μ=1 vs μ=2 updates per batch. Draft post #2. **Fri PM buffer.** | Run meeting acceptance criteria, all curves, seed bands, post #2 draft | Local (+ optional GPU seeds; TPU smoke only if TRC) |
| **6** | 40 | RLVR as research | Sparse vs one pre-chosen alternative (partial credit **or** Dr. GRPO-style), plus a format-only control, 3 seeds each. Eval: n=16 on full test + OOD for every final checkpoint; n=64 on a 200-item subset (k ≤ 32) for base/SFT/DPO/best arm (all seeds). Stricter held-out verifier gap. Inspect 30 samples. **Scorecard row 2.** | RLVR note (~3 pages) = draft of post #1 | **Cloud GPU burst** (9 runs). Eval mostly local overnight |
| **7** | 40 | Agents + agent eval | 50-100 spine tasks with checkers. Tools: sandboxed exec and tests (BM25 only if tasks need it). Multi-turn v1: re-prefill the full concatenated **token IDs** each turn. 3 trials/task, paired CIs, failure taxonomy. LLM vs +tools. API model as a stronger subject if needed. LLM judge only if a dimension needs it. Overnight: warm-start trajectory generation. Stretch: in-cache continuation. | Agent eval table + taxonomy + saved trajectories | Local (+ API) |
| **8** | 40 | Agent RL (2-3 turns) | Multi-turn GRPO reusing the week 4-6 code. Tool-token mask tests (labels/advantages, spans). SFT warm start on the trajectories collected (target ~200). Outcome-only vs outcome + valid-call bonus, 3 seeds (2 local + burst; cut-first → 2 seeds, exploratory). Turn and tool-error logging. Optional test-tampering probe. **Fri PM buffer. Scorecard row 3.** | Learning curves + failure analysis (fallback: verified pipeline + "why no learning") | Local + GPU burst (cut #1) |
| **9** | 40 | **Buffer** | Slippage first. Mandatory: publish posts #1 and #2. If time remains: fake-device sharding test (else week 11 Mon), TPU dry run on the week-11 slice type (<1 h), repo refactor. Cut-first: generality check on the second model, 1 seed. | Posts #1 and #2 public | Local (+ <1 h TPU) |
| **10** | 40 | Interpretability | Activation capture/intervention as function arguments in your model. Contrastive set (SFT→RL change, or hacked vs honest). Difference-of-means direction. Per-layer probes with control tasks. Ablate/add direction (behavior + capability). Base/SFT/DPO/RL on identical inputs. LoRA ΔW norm per layer. **Scorecard row 4. Name the top-2 areas.** | One figure linking an internal change to a training-caused behavioral change | Local |
| **11** | 40 | Systems: inference + sharded training | **Mon-Tue (local):** fake-device sharding if not done. Full sampler table (TTFT, inter-token latency, throughput, VRAM over batch {1,8,32,64} × prompt {128,1024}) vs vLLM (separate process, preallocation on) and the bandwidth roofline. Measured memory table. Int8 weight-only (your own) with an HLO fusion check, bytes per step, accuracy change. Optional Pallas kernel. **Wed-Thu (cloud):** 100-300M decoder on v5e-8/v6e-8 (spot/TRC): DP, FSDP, TP via `NamedSharding`/`shard_map`, sharded == unsharded, MFU at 1/2/4/8 chips, profiler trace, collectives in HLO. **Fri:** **scorecard row 5**, capstone pre-registration draft for the top-2 areas with a compute plan. | Inference tables, MFU scaling curve, sharding note, capstone pre-registration draft | Local + **cloud TPU (~10-16 h)** |
| **12** | 40 | Synthetic data + capstone pilot | **Mon-Wed:** verified candidate pool from a 4B teacher (your sampler, text-only loader, local). Arms at a fixed ~1k budget: random vs difficulty-filtered vs failure-targeted, 3 seeds. N-gram decontamination vs test. **Wed: scorecard row 6, then finalize the capstone pre-registration.** **Thu-Fri:** pilot(s) on the cloud config (one per finalist if the top two are within 10%), then **launch the capstone runs on Fri.** | Data ablation table. Pilot(s). Capstone launched | Local + capstone burst starts |
| **13** | 40 | Capstone run + decision | **Mon-Wed:** runs finish (5 seeds headline, 3 seeds ablation), eval, analysis. **Thu-Fri:** report + one-page area decision memo citing the capstone as the deciding row. | **Public post #3** + decision memo | **Cloud burst** (reserved $150) + local |

Cut or optional relative to the original syllabus:
- curriculum reward
- diversity selection
- web search
- QLoRA
- the IPO/ORPO skim
- "DPO on base"
- the "change one hyperparameter" exercise

Speculative decoding and paged attention are week 11 stretch goals.

### Week details and JAX implementation notes

**Week 0 (6-7 h, light).**
- **Two environments (~1.5 h).** Use `uv` with separate lockfiles. Never import torch and jax in the same process. Exchange parity data through saved `.npy`/`.safetensors` files of token IDs and logits.
  - **JAX env:** `jax` with the CUDA extra current at install time (verify the exact extra in the JAX install docs), `flax` (NNX), `optax`, `orbax-checkpoint`, `safetensors` and `tokenizers`. Add `transformers` only for chat templates, and only if it imports cleanly without torch (verify).
  - **Torch env:** torch, transformers, peft, trl and vLLM, used only for parity, baselines and fallbacks.
- **Memory and compile settings.**
  - Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` **only for development and parity runs**. For benchmarks and long runs, turn preallocation back on or set `XLA_PYTHON_CLIENT_MEM_FRACTION`, and record the setting.
  - Never benchmark vLLM and JAX at the same time on the same GPU. vLLM preallocates ~90% of memory by default (verify).
  - Enable the persistent compilation cache.
- **Applications (~20 min).** Apply for TRC (verify turnaround). File the GCP quota request for the week-11 slice type and region.
- **Model smoke test (≤1 h).**
  - Get an off-the-shelf **NNX** implementation of the primary candidate to load, generate and run one LoRA forward/backward on the 4090. Candidates: Tunix models/sampler or the official Gemma JAX library (verify NNX vs Linen and GPU support).
  - If it fails, try the second-choice model.
  - If neither works, run weeks 1-2 in TRL/PEFT in the torch env. The JAX masked log-prob function and its pytests then move to week 3 Mon, and week 4 Mon-Tue absorbs the overflow.
  - Generation for weeks 0-2 may use vLLM in any case. The harness is framework-neutral (input: JSONL of completions; output: scored JSONL).
- **Harness core (~2 h).** Spawn-based verifier pool, unbiased pass@k and per-item JSONL. Measure verifier samples/s, which you need for week-6 sizing.
- **Overnight.** Run the dev gate first (~200 × 8 samples), then the noise floor (3 sampling seeds on test). For code, also run the verified rejection-sampling pool for SFT.
- **Cut from the old week 0 to save time:** PyTorch onboarding, the throughput benchmark and the memory table. These move to weeks 3, 4 and 11.

**Week 1 (5-7 h, light).**
- Wrap the linear layers of the off-the-shelf NNX model with `nnx.LoRA`, or with the LoRA-aware linear (check the current class names, verify).
- **Train step.**
  - Use `nnx.Optimizer(model, optax.adamw(...), wrt=nnx.LoRAParam)` and differentiate only LoRA params, e.g. `nnx.grad(..., argnums=nnx.DiffState(0, nnx.LoRAParam))`. The constructor signature and filter names changed across recent Flax releases (verify).
  - Test that the optimizer-state leaf count and bytes match only the LoRA params. Unfiltered, you would allocate ~2 GB of grads plus ~8 GB of fp32 Adam state for a 1B model and train the base model by accident.
  - Keep LoRA params and Adam moments in fp32.
- **Masked per-token log-prob function.** Write it now and reuse it for DPO, KL and GRPO. Upcast logits to fp32 for the loss. Pytests:
  - prompt positions contribute exactly 0 to the loss
  - masked mean NLL equals a hand-computed value on a 3-token example
  - EOS is included in the target, and pad tokens are excluded
  - parity vs HF PyTorch NLL on 5 identical sequences, using reference values saved from the torch env (tolerance ~1e-2 in bf16)
- **Stats and forgetting.** Write the paired bootstrap code (single-seed version) and build the general-capability slice for the forgetting check.
- Export a merged SFT checkpoint. You need it for the RL reference in week 4.

**Week 2 (5-7 h, light).**
- Write the derivation first (1 h): KL-regularized objective → optimal policy → Bradley-Terry → DPO loss.
- **Pairs:** 4 samples per prompt from the SFT model, then pair one pass with one fail using the verifier. Skip prompts that are all-pass or all-fail, and log what fraction were skipped.
- The DPO loss is ~15 lines on top of the week 1 log-prob function. Precompute reference log-probs once with the SFT model before DPO training starts. The DPO adapter starts from the SFT adapter.
- **Log:** verifier metric, mean length, chosen-rejected margin, and the chosen log-prob trend (it often falls, so report it).
- Single seed this week. The 3-seed version runs on week 4 nights.

**Week 3 (40 h): build your JAX LLM stack.** This is the most important week. The **gates** are parity, the sampler MVP, LoRA, chunked log-probs and Orbax. Everything else is polish, and week 4 Mon-Tue AM absorbs any overflow.

- **Mon AM (~3 h).**
  - Finalize the OOD slice.
  - Implement the two-level (seed × item) paired bootstrap and the per-seed sign check.
  - Reading.
  - If weeks 1-2 ran in TRL, write the JAX log-prob function and its tests now.
- **Model (Mon PM-Tue, timebox 1.5 days).**
  - Write an `nnx.Module` for your spine architecture yourself, using Tunix/MaxText/the Gemma JAX code as references.
  - Loader: safetensors → NNX state, with an explicit name-mapping table. It errors on unmapped or missing keys.
  - **Precision for parity.** Set `jax.config.update("jax_default_matmul_precision", "highest")` in JAX. In the torch env, turn TF32 off for matmuls (verify both frameworks' defaults for your versions). By default, JAX on Ampere/Ada may use TF32 for fp32 matmuls, and TPU uses bf16 passes, either of which breaks fp32 parity.
  - **Acceptance:**
    - fp32 logits within ~1e-4-1e-3 max abs of the HF fp32 reference on 20 prompts (verify a realistic tolerance for your model)
    - **fp32** greedy decode identical to HF for 64 tokens on ≥95% of 50 prompts. A mismatch counts as a bug only if the top-2 logit gap at the first divergence exceeds a threshold, e.g. 1e-3
    - **bf16:** only top-1 agreement ≥99% and a log-prob tolerance. No greedy-identity test, because near-tied logits diverge spuriously.
  - **Architecture parity checklist** (verify each against the config):
    - Gemma 3: embedding scaled by √d, RMSNorm with (1+w), QK-norm, pre/post norms, separate RoPE bases for local vs global layers, sliding-window local layers (masking is enough at ≤1k tokens).
    - Qwen2.5: QKV bias and tied embeddings.
- **KV-cache sampler, staged (timebox 2 days).**
  - **MVP, due Wed evening:**
    - pre-allocated cache `[layers, B, max_len, kv_heads, head_dim]` with a fixed max_len
    - one left-padded prompt bucket, with position IDs and an attention mask
    - prefill, then a decode loop in `jax.lax.while_loop` with a per-sequence `done` mask
    - temperature sampling
    - returns the sampled tokens' log-probs, computed as `log_softmax(logits / T)` over the **untruncated** vocab, matching what the trainer computes
    
    The MVP alone unblocks week 4.
  - **Thu-Fri polish:**
    - length buckets (e.g. 128/256/512)
    - early exit when all sequences are done
    - **eval-only** top-k/top-p, implemented as `lax.top_k` (e.g. k=64) followed by nucleus filtering inside the top-k, never a full 262k-vocab sort
    - turn on `jax_log_compiles` and assert zero recompiles after warmup
  - **RL rollouts use top-p=1 with top-k off.** Otherwise the sampler's log-probs come from a different distribution than the trainer's, and the importance ratio is biased.
  - Parameters are passed as **jit arguments, never closed over**. Closing over them bakes 2 GB of constants into the compile and causes recompiles. For hot loops, use `nnx.split`/`nnx.merge` with a plain `jax.jit` (check the Flax performance guidance, verify).
  - **Acceptance:**
    - sampler log-probs vs teacher-forced full-forward log-probs on the same token IDs, at T=1 and T=0.7, with mean abs diff ≲1e-2 in bf16
    - that remaining gap is your numerical sampler-trainer mismatch baseline
- **Your own LoRA (half a day).**
  - A module with fp32 `A` and `B`, a `scale` passed as a traced array argument (0 means the reference policy), and merge/unmerge.
  - Tests:
    - identical output vs `nnx.LoRA` given the same params
    - merged == unmerged
    - toggling scale does not recompile
    - the merged safetensors export loads in HF with matching logits
    - reference log-probs with scale 0 on the merged-SFT base equal the SFT model's log-probs
- **Chunked log-probs.**
  - Never materialize `[B, T, V]`. With Gemma's 262k vocab, 256 seqs × 768 tokens × 262k × 2 bytes ≈ 100 GB.
  - Use `lax.scan` over sequence chunks, **with the chunk body wrapped in `jax.checkpoint`** (or `nnx.remat`). Without remat, scan still saves each chunk's logits for the backward pass.
  - Upcast each chunk to fp32, then gather the target logit and the logsumexp.
  - Test peak memory under a set bound, e.g. via `jax.jit(f).lower(...).compile().memory_analysis()` (verify).
- **Memory budget (arithmetic, 30 min).**
  - Weights: 1B in bf16 ≈ 2 GB.
  - LoRA + Adam in fp32: about 100-200 MB at r=16 (depends on targets).
  - KV per token:
    - Qwen2.5-1.5B: 28 layers × 2 KV heads × 128 dims × 2 (K, V) × 2 bytes ≈ 28.7 KB/token.
    - Gemma 3 1B: 26 layers × 1 KV head × 256 dims × 2 × 2 bytes ≈ 26.6 KB/token (verify the config).
    - So 256 seqs × 768 tokens is about 5-6 GB.
  - Activations with remat, plus the logits chunk.
  - The measured table comes in week 11.
- Orbax checkpointing for LoRA + optimizer state, with a resume test.
- Overnight: re-measure the noise floor with your sampler.

**Week 4 (40 h).**
- **Mon-Tue AM: week-3 overflow.** It is reserved, so use it if you need it.
- **Toy RL (Tue PM-Wed AM).**
  - A ~1M-parameter NNX transformer on a verifiable string task (e.g. reverse/sort/sum digits).
  - Train 16-32 seeds at once, either with `nnx.vmap` + `nnx.split_rngs` (verify the current idiom) or with a functional `jax.vmap` over split state.
  - Variants: no-baseline, leave-one-out, and LOO + KL to reference. The mean baseline is optional.
  - Plot reward and gradient variance for each variant.
- **Losses (Wed).**
  - GRPO advantage (group mean/std, with a flag for no std normalization) and the RLOO advantage.
  - Clipped surrogate with a per-token ratio.
  - **KL placement, pre-registered.** Pick one:
    - (a) k1 = log(π/π_ref) subtracted from the reward, inside the advantage, with no gradient through it, or
    - (b) k3 as a loss term, as in DeepSeekMath GRPO.
    
    Do not use k1 directly as a loss term on on-policy samples. Its expected KL gradient is zero. k3 as a loss gives a biased KL gradient (verify the exact analysis). Log k1 and k3 as diagnostics either way.
  - **Tests:**
    - hand-computed values on 2 groups × 3 samples, including a zero-variance group, which must give zero advantage and no NaN
    - on a small categorical policy, compare the expected gradient of your KL implementation (over many samples) with the exact KL gradient
- **Pipeline (Thu-Fri).**
  - **Choose and pre-register one loop:**
    - **(a) Strictly on-policy (default).** Sample batch t, then verify it on CPU while the GPU computes reference log-probs for t, then train, then sample t+1. Sampling and training share one GPU, so only CPU verification can overlap usefully.
    - **(b) One-step async.** Sample t+1 while training on t, store the sampler's behavior log-probs as π_old, and log staleness and clip fraction.
  - **Microbatched train step.** Use gradient accumulation (8-16 seqs per microbatch) via `lax.scan` or a Python loop over a jitted microstep. 256 seqs × ~768 tokens on a 1-1.5B model will not fit in one batch.
  - Free the KV cache before the train phase. The sampler returns only tokens, log-probs and lengths.
  - **Donation.** Split the state into frozen base weights, which are never donated and are shared by the sampler, reference and trainer, and trainable LoRA + optimizer state, which is donated via `donate_argnums`. Donating everything invalidates the arrays the sampler still holds.
  - **Tests:**
    - run sample→train→sample twice without "Array has been deleted"
    - the memory test at the full week-5 config passes with headroom
  - No weight sync is needed: same process, same arrays.
  - Run a 50-step smoke run and log the sampler-vs-trainer log-prob mismatch.
- **Throughput gate (Fri).** Measure rollout tokens/s at the week-5 config (256 × 512). For comparison, run one vLLM reference number at the same config in the torch env, in a separate process at a different time, with preallocation on.
  - **Pass if** you are within 4× of vLLM, **or** tokens/s × planned steps fits 300 steps in one night (~10 h).
  - **If it misses:** spend 1-2 days profiling, taken from week 5 Mon-Tue. Check `jax.profiler`, recompiles, cache layout, bucket choice and early exit.
  - **If it still misses:** reduce group size or generation length, or move weeks 5-6 runs to bursts.
  - Either way, write the measured tokens/s into the week 5-6 compute table.
- **Nights:** re-run the weeks 1-2 headline on your stack with 3 seeds.
- **GPU dry run** if time allows (otherwise week 5 Mon): rent 1 GPU for under 1 h and run `launch.sh` → 20-step RL → kill → resume → upload → delete.

**Week 5 (40 h).**
- **Config and sizing.**
  - Starting config: 32 prompts × group 8 = 256 rollouts/step, ≤512 new tokens.
  - Size the runs from the week-4 measured step time. The prior rough estimate was ~30-90 s/step for a ~1B model, so 300 steps would take ~3-6 h. Do not trust it over your measurement.
- **LR sweep** with 3 LoRA LRs × 100 steps, then the main run plus 2 more seeds overnight.
  - Extra seeds may go to a GPU burst if the dry run passed (cut #2).
  - If TRC TPUs are available and you want them for week 6, do the TPU smoke test now, with log-prob parity vs the 4090 at `highest` precision.
- **Always log:**
  - reward mean/std
  - fraction of zero-variance groups
  - KL (k1 and k3)
  - length distribution
  - entropy
  - clip fraction
  - held-out pass@1 every N steps
  - sampler-trainer log-prob mismatch
  - tokens/s and step time
  - staleness, if you chose option (b)
- **Early kill:** NaN, entropy collapsing below a threshold, KL blowing past a set bound, or reward flat for 100 steps with more than 80% zero-variance groups.
- **Acceptance:** held-out pass@1 beats SFT by more than 2× the noise floor, the two-level paired CI excludes 0 with sign agreement across seeds, and KL stays bounded.
  - Fallback: reward clearly rising on a task subset within ~200 steps.
- **μ=1 vs μ=2.** Run μ=1 first.
  - Under loop (a), with π_old recomputed by the trainer, the ratio is exactly 1, so clip fraction is trivially 0.
  - If π_old comes from the sampler, the ratio reflects numerical mismatch. Log it.
  - Then run μ=2 to exercise the ratio and clipping.
- Draft post #2.
- **Fri PM is a buffer.**

**Week 6 (40 h).**
- **Training.** Pre-register the arms (sparse vs one alternative) and the format-only control: 3 arms × 3 seeds = 9 runs. At ~4 h each, that is 4-5 nights locally, so burst on 3× rented H100 SXM instead (section 6). NVIDIA only. The code is unchanged.
- **Eval (budgeted).**
  - Every final checkpoint gets n=16 at the training temperature on full test (≥300) and OOD, for pass@1 CIs.
  - Base, SFT, DPO and the best arm (all 3 seeds) get n=64 on the fixed 200-item subset, with pass@k for k = 1..32.
  - k=64 with n=128 only if the budget allows (cut #3).
  - That is about 50M generated tokens in total: ~5-10 GPU-h on the 4090 at the gate throughput, plus verifier time at the measured rate. Run it locally over 1-2 nights, and burst only if it would take longer.
  - Plot pass@k curves on the same axes with CIs, and compare with Yue et al. 2025's claim.
- **Hacking.**
  - Stricter held-out verifier: EvalPlus-style extended tests on the overlapping items, or a stricter math checker. It is never used for reward.
  - Training items have no extended tests, so measure the training-reward vs held-out gap on a ~100-item subset with added hidden tests.
  - Inspect 30 high-reward samples by hand.
  - Track length and format statistics.
- Write the RLVR note on Friday.

**Week 7 (40 h).**
- **Tasks (Mon).** Write the tasks and checkers first. Tools return strings, and tool output is truncated to a fixed token budget. Add BM25 or an LLM judge only if the tasks need it.
- **Multi-turn v1 (Tue AM).**
  - Re-prefill the full concatenated context each turn through the tested prefill path. At 2-3 turns and ≤2k tokens this is cheap.
  - Keep everything as **token IDs** end to end, with the chat template applied at the token level. Never re-tokenize concatenated text: turn-boundary tokenization can differ, and you would compute log-probs for tokens that were never sampled.
  - Log the token-ID span of each inserted tool output.
- **Stretch: in-cache continuation.**
  - Keep per-sequence write positions (an int array `[B]`) and write with a vmapped `dynamic_update_slice` or a scatter.
  - Pad appended chunks to fixed buckets with a mask, and size max_len for all turns.
  - Test that the continuation matches a full prefill of the *same token IDs*.
- **Eval.** Run 3 trials per task, then paired CIs and a failure taxonomy (~10 categories, hand-coded on 50 failures).
  - If you use an LLM judge, it covers one dimension only, is validated against 30-50 hand labels, averages both orderings, and reports its agreement rate.
- **Overnight: warm-start data.** Generate warm-start trajectories with many samples per task, or with an API or teacher model (check the terms). Keep the successful ones and aim for ~200.

**Week 8 (40 h).**
- **Environment and data.**
  - 2-3 turns with a per-turn token cap.
  - SFT warm start on the trajectories collected. Use whatever count was reached and report it.
- **Mask tests.** Tool-output tokens are conditioning for later tokens, so they correctly change gradients through the context. That is not what the tests check. They check:
  - (a) the per-token loss and advantage at tool positions is exactly 0
  - (b) changing the *target labels or advantages* at tool positions, with the input tokens fixed, leaves loss and gradient bit-identical
  - (c) the mask covers exactly the token-ID spans the env logged when it inserted them
- **Runs.** Outcome-only vs outcome + valid-call bonus at matched steps, 3 seeds.
  - Default: 2 seeds per arm locally and 1 per arm in a GPU burst.
  - Cut-first: 2 seeds, labeled exploratory.
  - Multi-turn steps are slower than week 5, so measure before sizing.
- Log turns per episode and tool-error rate.
- Optional: give the agent write access to the test file and log the tampering rate.
- **Fri PM is a buffer.**

**Week 9 (real buffer).**
- **Mandatory:** absorb slippage first, then publish posts #1 and #2.
- **If time remains, in this order:**
  1. **Fake-device sharding test.** Otherwise it moves to week 11 Mon.
     - Set `XLA_FLAGS=--xla_force_host_platform_device_count=8` in the test process environment *before* `import jax`: in the shell, or at the top of `conftest.py`. Newer JAX also has a `jax_num_cpu_devices` config option (verify).
     - Assert `len(jax.devices()) == 8` at the start, or the test passes trivially on 1 device.
     - Use `Mesh` + `NamedSharding`/`PartitionSpec` and NNX partition annotations via `nnx.with_partitioning` (verify the current helpers for extracting specs).
     - Shard DP and FSDP-style, and test that the sharded loss and gradients match unsharded within tolerance.
  2. **TPU dry run (<1 h)** on the **same slice type as week 11** (spot or TRC). This proves the quota.
     - Run `pip install` of your pinned stack, the sharding test, and the week-11 train step on one chip for ~10 min, at `highest` precision for any parity checks, then delete the slice.
     - Debug on Kaggle TPU first where possible.
  3. Repo refactor.
  4. **Cut-first:** the generality check (week-5 headline on the second candidate model, 1 seed, exploratory).

**Week 10 (40 h).**
- **Capture and intervention.** You own the model, so capture and interventions are function arguments: return a dict of residual-stream activations per layer, and accept an optional `intervene(layer, h) -> h`. This avoids hook frameworks. NNX `sow`/intermediates are an alternative (verify the API).
- **Data.** Build a contrastive set from your own runs: behavior changed by SFT→RL, or hacked vs honest samples from weeks 6 and 8.
- **Analysis.**
  - Difference-of-means direction.
  - Per-layer linear probes **with a Hewitt-Liang control task**.
  - Ablate or add the direction, measuring both the target behavior *and* the general-capability slice.
  - Compare base/SFT/DPO/RL on identical inputs.
  - Cheap extra: ‖BA‖ per layer and module for each adapter, which shows where training changed the model.
- **Friday:** fill scorecard row 4 and name the **top-2 areas** by weighted score. They are the capstone finalists.

**Week 11 (40 h).**
- **Local (Mon-Tue).**
  - Fake-device sharding, if week 9 didn't get to it.
  - **Methodology:** exclude warmup and compile time, report median and p90 over ≥10 runs, run with preallocation on, and run vLLM in a separate process at a different time.
  - **Roofline:** the 4090 has ≈1 TB/s, so a 1B bf16 model reading 2 GB per decode step takes ~2 ms/step at batch 1 in the ideal case.
  - Build the full sampler table vs vLLM and the measured memory table. They go into the post #2 update.
  - **Int8 weight-only quantization**, written yourself: per-channel scales, dequantize in the matmul.
    - Check the optimized HLO and the profiler trace to see whether the dequantize fuses into the dot. If it does not, bf16 weights are materialized every step and the speedup is ~0.
    - If it isn't fused, add a Pallas int8×bf16 kernel or compare against Qwix (verify).
    - Report bytes moved per decode step next to latency, against the roofline, plus the accuracy change on the harness.
  - Optional: one Pallas kernel. On the 4090 (sm_89), Pallas GPU uses the Triton lowering, and Mosaic GPU targets Hopper and newer (verify the current status).
  - Stretch: speculative decoding with a smaller same-family draft.
- **Cloud (Wed-Thu, spot or TRC).**
  - A 100-300M decoder in your NNX code, on synthetic or tokenized-text data.
  - Scale over 1/2/4/8 chips, comparing DP vs FSDP vs FSDP+TP via `PartitionSpec`.
  - **MFU** = (6·N·tokens/s) / (chips × peak bf16 FLOPs). Add the attention term if the sequence is long (PaLM-style accounting).
  - A `jax.profiler` trace viewed in TensorBoard/Perfetto.
  - Find the all-gather/reduce-scatter in the compiled HLO.
  - Write a `shard_map` version of one layer (check the import path: `jax.shard_map` vs `jax.experimental.shard_map`, verify).
- **Friday.**
  - Fill **scorecard row 5** (systems), using evidence from weeks 3, 4 and 11.
  - Draft the capstone pre-registration for the top-2 areas. Each one tests that area's highest-Surprise open question and includes a compute plan: runs × measured hours, accelerator and cost.

**Week 12 (40 h).**
- **Teacher.** Gemma 3 4B, or a Qwen ~3-7B text model, in your loader.
  - Gemma 3 4B/12B are **multimodal checkpoints** whose safetensors include vision-tower weights and different name prefixes. Extend the loader to load only the language-model weights, with explicit prefix handling and errors on unmapped or missing keys.
  - Re-run logit parity against the HF text-only forward before generating the pool.
  - A 12B teacher is off by default: use an API (check the terms) if you need it.
- **Data ablation.** Fixed 1k budget, 3 arms × 3 seeds, LoRA SFT. N-gram decontamination (e.g. 13-gram) against test.
- **Wednesday.** Fill **scorecard row 6** (data). Finalize the capstone pre-registration. If synthetic data is now a finalist, its capstone can extend this week's pipeline directly.
- **Thursday-Friday.**
  - Run the capstone pilot on the cloud config, so the week-13 launch is copy-paste.
  - If the top two areas are within 10% after row 6, run one pilot per finalist and choose the capstone from the pilot results. This replaces the old post-plan 2-week tiebreak.
  - **Launch the capstone runs Friday.**

**Week 13 (40 h).**
- **Capstone** = the pre-registered hypothesis on an existing pipeline. The headline comparison gets 5 seeds, and one ablation arm gets 3 seeds.
- **Report:** question, related work, setup, results with two-level CIs, ablation, failures, next steps.
- **Decision memo (1 page):** the chosen area, the scorecard evidence with the capstone as the deciding row, the area you are dropping and why, and the first 3 follow-up experiments with their compute plan.

## 5. JAX/Flax stack: build vs use

**Confident core:** JAX, Flax NNX (the recommended API; Linen is in maintenance), `nnx.LoRA`, Optax, Orbax, `jax.sharding` (`Mesh`, `NamedSharding`, `PartitionSpec`), `shard_map`, `jax.profiler` and Pallas. HF transformers has dropped its Flax model classes, so do not plan around `FlaxAutoModel`. HF is used for tokenizers, chat templates, safetensors and PyTorch parity references, all in the separate torch env except tokenizers.

| You build it (portfolio-grade) | Week | Off-the-shelf |
|---|---|---|
| Eval harness: frozen splits from an existing dataset, spawn-based verifier pool, unbiased pass@k, paired and two-level bootstrap, JSONL | 0, 1, 3 | numpy/scipy, `multiprocessing` |
| Masked per-token log-prob + tests | 1 (or 3) | none |
| DPO loss + reference precompute | 2 | none |
| NNX model for one architecture + safetensors loader + parity tests | 3 | Tunix / MaxText / Gemma JAX code as *reference reading* (verify APIs); HF PyTorch as parity oracle |
| KV-cache sampler (MVP → buckets, `while_loop`, eval-only top-k/top-p, EOS, untruncated log-probs; in-cache multi-turn as week 7 stretch) | 3, 7 | vLLM only as a throughput reference |
| LoRA module (fp32, traced scale, merge/export), compared against `nnx.LoRA` | 3 | `nnx.LoRA` |
| Chunked vocab log-softmax (scan + remat) | 3 | none |
| SFT / DPO / GRPO / RLOO train loops, microbatching, selective donation, single-process RL orchestration, logging | 1-6 | Optax, Orbax, tracker (W&B or similar) |
| Toy RL with `vmap`ped seeds, KL-gradient test | 4 | none |
| Multi-turn agent env + tool-token masking | 7-8 | subprocess sandbox; BM25 library only if needed (e.g. `rank_bm25`, verify) |
| Activation capture / intervention API | 10 | none (Penzai/Treescope for visualization, optional, verify) |
| Int8 weight-only quant with fusion check, roofline analysis, optional Pallas kernel | 11 | Qwix as comparison (verify) |
| Sharded train step (DP/FSDP/TP), MFU accounting, HLO inspection | 9, 11 | `jax.sharding`, `jax.profiler` |

**Where PyTorch/vLLM is legitimately used (secondary, separate env):**
- Logit and loss parity oracles (weeks 1, 3, 12), exchanged through saved files.
- Week 0-2 generation, or week 1-2 training via TRL/PEFT, **only if** no NNX path works within the week-0 timebox.
- The vLLM throughput reference (week 4 gate, week 11 table).
- Optionally, a large-teacher generation pool in week 12 if your sampler would take more than a day longer.
- The fallback rule (section 3.3) anywhere a JAX path is more than ~1 day over its timebox, including the RL fallback if the sampler MVP fails by week 4 Tue.

**Verify before relying on (status uncertain as of Nov 2026):**
- Tunix maturity on a single NVIDIA GPU, and its model/sampler coverage.
- EasyDeL and Levanter model coverage and GPU status.
- MaxText on a single GPU.
- Qwix APIs.
- Whether the official Gemma JAX library is NNX or Linen.
- vLLM's TPU/JAX backend.
- cuDNN flash attention via `jax.nn.dot_product_attention(implementation="cudnn")` on sm_89.
- Current names of `nnx.Optimizer(..., wrt=...)`, `nnx.LoRAParam` and `nnx.DiffState`.
- `compile().memory_analysis()`.
- `jax_num_cpu_devices`.
- Whether `transformers` imports cleanly without torch.
- EvalPlus / MBPP+ / HumanEval+ availability and licenses.

**Recurring JAX pitfalls to watch for:**
1. Parameters closed over in `jit` instead of passed as arguments.
2. Dynamic shapes causing recompiles. Bucket the shapes and watch `jax_log_compiles`.
3. Full `[B, T, V]` logits, including scan without remat.
4. Missing `donate_argnums`, or donating frozen weights the sampler still uses.
5. Reusing a PRNG key. Use `fold_in(step)` and `split` per sample.
6. `nnx.jit` Python overhead in tight loops. Prefer split/merge + `jax.jit` (verify current guidance).
7. bf16 accumulation in the loss. Upcast logits and log-probs to fp32.
8. Unfiltered `nnx.grad`/`nnx.Optimizer`, which trains and stores state for the whole base model.
9. LoRA params or Adam state kept in bf16.
10. TF32 (GPU) or bf16-pass (TPU) default matmul precision breaking fp32 parity. Use `highest` for parity tests.
11. Forking after JAX initialized CUDA. Use spawn/forkserver pools whose workers never import jax.
12. Sampler log-probs from a truncated (top-p/top-k) or different-temperature distribution than the trainer's.
13. Benchmarking with preallocation off, or with vLLM sharing the GPU.
14. Setting `XLA_FLAGS` after `import jax`.
15. A Python-attribute LoRA `scale`/`enabled` flag toggled under jit, which causes retracing or bakes in the wrong branch.

## 6. Cloud compute guide

**When:** see the triggers in section 3.4. Planned bursts:

| Week | Run | Suggested hardware | Rough size | Rough cost (verify prices) |
|---|---|---|---|---|
| 4-5 | **GPU dry run:** `launch.sh`, 20-step RL, kill + resume | 1× H100 SXM or A100 80GB | <1 h | ~$2-4 |
| 5 (optional, cut #2) | 2 extra seeds of the main run | same | ~2 × 2-3 h | ~$10-20 |
| 5 (only if TRC approved early) | TPU smoke test of the RL loop + log-prob parity vs 4090 | v5e-1 / v6e-1 | 1-2 h | $0 (TRC) |
| **6** | 9 RL runs (3 arms × 3 seeds) | 3× H100 SXM in parallel (**NVIDIA only**) | ~9 × 2-3 h (scale from week-4 tokens/s; measure) | ~$40-80 |
| 6 | Eval generation (n=16 on all checkpoints, n=64 on the subset) | local overnight by default; burst only if it would take more than 2 nights | ~50M tokens, ~5-10 GPU-h on the 4090 (estimate; size from measured tokens/s) | $0-20 |
| 8 (cut #1) | Agent-RL extra seeds (1 per arm) | 2× H100 SXM | ~2-4 × 2-3 h | ~$15-35 |
| 9 | TPU dry run on the **week-11 slice type** | v5e-8 spot (or TRC) | <1 h | ~$5-10 |
| **11** | Sharded training, scaling, profiling | **TPU v5e-8** (16 GB HBM/chip, ~197 TFLOPs bf16/chip) or v6e-8 (32 GB/chip, ~918 TFLOPs bf16/chip), **spot or TRC only** | 10-16 h active | ~$50-80 on spot (v5e-8 on-demand ~$10/h → $100-160 is not planned; verify) |
| 12 | 12B teacher pool | **off by default** (4B local, or API) | — | $0 (API cost if used) |
| **13** | 5-seed capstone headline + 3-seed ablation, and/or 4B scale check | parallel H100s for seeds, v6e-8 for scale | ~10-30 accelerator-h | **reserved $150** |

The planned total is roughly **$150-400 without TRC**, under the $500 hard cap with ~20% margin. It is near zero if TRC and Kaggle quota cover the TPU work.

List prices are roughly:
- v5e ≈ $1.2/chip-h
- v6e ≈ $2.7/chip-h
- spot ≈ 40-60% of on-demand
- rented H100 ≈ $2-3.5/h
- A100 80GB ≈ $1-2/h

All of these vary by region and provider and change often (verify). If spending approaches the cap, follow the cut order in section 3.4. Never cut the week-13 reservation.

**Free and cheap sources (verify all):**
- **TRC (TPU Research Cloud):** free TPU quota for a limited window (often ~30 days from activation, verify) after an application. Apply in week 0 and ask for the window to cover about weeks 9-13.
- **Kaggle:** free TPU notebooks with a weekly hour quota. The chip generation has changed over time (verify). Good for debugging sharded code in weeks 9-11, bad for long unattended runs.
- **Colab:** TPU runtimes with variable availability. Fine for quick checks only.
- **GCP TPU VMs** (on-demand/spot; file the multi-chip quota request in week 0), and GPU marketplaces/clouds for H100/A100.

**TPU vs GPU for your JAX code:**
- **Same-code portability.** Your 4090 code runs unchanged on an H100/A100. On TPU it usually runs too, but several things differ: recompile behavior, default matmul precision (bf16 passes), memory (16 GB/chip on v5e) and GPU-specific kernel paths (cuDNN attention, Pallas-Triton). Never fan out on TPU before a TPU smoke or dry run.
- **Fan-out of independent single-model runs (weeks 6, 8, 13 seeds):** rented NVIDIA GPUs are simplest, with one run per GPU, no sharding and no porting. A v6e-1 (32 GB) also fits a 1B LoRA RL run under TRC, but only after the TPU smoke test.
- **Multi-device sharding and MFU (week 11):** TPU is the natural target. The ICI interconnect, XLA's best-tuned backend and the role-relevant skill all point there. An 8×GPU node is far more expensive per hour.
- **Decode-heavy RL rollouts:** memory bandwidth dominates. The 4090 has ~1 TB/s, v5e ~0.8 TB/s/chip, v6e ~1.6 TB/s/chip, H100 SXM ~3.35 TB/s and H100 PCIe ~2 TB/s. An H100 SXM is the fastest single device for your sampler.
- **Compile time:** large jitted programs compile slowly, especially on TPU. Use the persistent cache and pay compile time once per shape bucket.

**Cost-conscious burst habit (do it every time):**
1. Make it work locally at 1/10 scale with the same pinned versions.
2. Write a single `launch.sh` that does setup → run → upload results → self-delete or shutdown. The week 4-5 dry run tests it end to end.
3. Use spot instances with Orbax resume, tested by killing a run mid-way.
4. Watch the first 10 minutes. If throughput or loss looks wrong, kill it.
5. Delete the instance the same day.
6. Log `accelerator_hours` and `cost_usd` per row, and update the README running total.
7. At each scorecard week, review cost per insight. If a burst produced no decision-relevant number, tighten the trigger.

## 7. Area-decision scorecard

Fill one row per candidate area at weeks **2, 6, 8, 10, 11, 12** (15-20 min each). Score each criterion 1-5 and **link the evidence** (a plot or table in the repo). An unlinked score counts as 1.

| Criterion | Weight | What a 5 looks like |
|---|---|---|
| Surprise / open questions from *your* data | 3 | Your results contradicted an expectation or raised a question you can state as a hypothesis |
| Iteration speed on 4090 + small bursts | 2 | Meaningful experiment cycle under 1 day locally |
| Follow-up feasible on ≤$100/month + 4090 | 2 | The next 3 experiments are sized and fit |
| Enjoyment (honest) | 2 | You worked past the timebox voluntarily |
| Relevance to target roles | 2 | Maps directly to job postings or labs you want |
| JAX/TPU leverage | 1 | Your built infrastructure is a real advantage here |
| Public-signal potential | 1 | A clear post or repo others would use or cite |

Candidate areas, scored where you have touched them:
- post-training / RLVR (weeks 2, 5-6)
- agents and agent RL (7-8)
- evals and methodology (0, 6-7)
- interpretability (10)
- inference and training systems / JAX infra (3-4, 11)
- synthetic data and curation (12)

**Capstone selection rule:**
1. End of week 10: take the top-2 areas by weighted score.
2. Week 11 Fri: after the systems row, draft a pre-registered capstone for each finalist. Each one tests that area's highest-Surprise open question and includes a compute plan.
3. Week 12 Wed: after the data row, finalize. If the top two are within 10%, run one pilot per finalist in week 12 Thu-Fri and choose the capstone from the pilot results. Otherwise, pilot the leader's capstone.

**Decision rule (week 13 memo):**
1. Start from the highest weighted total, with the capstone result entered as the deciding row.
2. Require that the chosen area has at least one row with Surprise ≥4 backed by your own data.
3. Require a written first follow-up experiment that fits the compute rule.
4. If the capstone falsifies the leader's premise, the memo may choose the runner-up, but it must say why, citing the week-12 pilot evidence.
5. Record the area you are dropping and why, so the decision can be checked against later evidence.

## 8. Reading list

**SFT and PEFT:**
- LoRA (Hu et al. 2021)
- QLoRA (Dettmers et al. 2023)
- LIMA (Zhou et al. 2023)
- LoRA Learns Less and Forgets Less (Biderman et al. 2024)

**Preference learning:**
- DPO (Rafailov et al. 2023)
- IPO, "A General Theoretical Paradigm to Understand Learning from Human Preferences" (Azar et al. 2023)
- ORPO (Hong et al. 2024)
- KTO (Ethayarajh et al. 2024)
- Tulu 3 (AI2, 2024)
- Length-controlled AlpacaEval (Dubois et al. 2024)

**RL and RLVR:**
- PPO (Schulman et al. 2017)
- InstructGPT (Ouyang et al. 2022)
- Schulman, "Approximating KL Divergence" (blog note, k1/k2/k3)
- RLOO, "Back to Basics" (Ahmadian et al. 2024)
- DeepSeekMath, which introduced GRPO (Shao et al. 2024). It is the reference for the k3-as-loss KL placement.
- DeepSeek-R1 (2025)
- Dr. GRPO, "Understanding R1-Zero-Like Training: A Critical Perspective" (Liu et al. 2025)
- DAPO (2025)
- Yue et al. 2025, "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
- Codex / unbiased pass@k (Chen et al. 2021)
- "Spurious Rewards: Rethinking Training Signals in RLVR" (2025) (verify exact citation)

**Agents and evals:**
- ReAct (Yao et al. 2022)
- Toolformer (Schick et al. 2023)
- SWE-bench (Jimenez et al. 2023)
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al. 2023)
- *Added for the spine datasets:*
  - "Program Synthesis with Large Language Models" (Austin et al. 2021), the MBPP paper
  - "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation" (Liu et al. 2023), the EvalPlus paper
  - "Training Verifiers to Solve Math Word Problems" (Cobbe et al. 2021), the GSM8K paper

**Interpretability:**
- Hewitt & Liang 2019, "Designing and Interpreting Probes with Control Tasks"
- Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction"
- Representation Engineering (Zou et al. 2023)
- Activation Addition (Turner et al. 2023)
- Toy Models of Superposition (2022)
- Towards Monosemanticity (2023)
- Scaling Monosemanticity (2024)
- Anthropic circuit-tracing / attribution-graph work (2025)
- Gemma Scope (2024). It covers Gemma 2, so check SAE coverage for your chosen model (verify).

**Systems and scaling:**
- FlashAttention (Dao et al. 2022)
- vLLM / PagedAttention (Kwon et al. 2023)
- Speculative decoding (Leviathan et al. 2023; Chen et al. 2023)
- LLM.int8() (Dettmers et al. 2022)
- GPTQ (Frantar et al. 2022)
- AWQ (Lin et al. 2023)
- Scaling Laws for Neural Language Models (Kaplan et al. 2020)
- Chinchilla (Hoffmann et al. 2022)
- "How to Scale Your Model" (Google DeepMind JAX/TPU scaling book, 2025)
- *Added:*
  - "Efficiently Scaling Transformer Inference" (Pope et al. 2022). KV-cache and decode roofline reasoning for weeks 3 and 11.
  - PaLM (Chowdhery et al. 2022). MFU definition.
  - GSPMD (Xu et al. 2021). The sharding model behind `jax.sharding`.
  - ZeRO (Rajbhandari et al. 2019). The FSDP-style sharding concept.
  - "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al. 2022). Remat trade-offs.
  - Gemma 3 Technical Report (2025). Your model's architecture if you choose Gemma.

**Synthetic data:**
- Self-Instruct (Wang et al. 2022)
- STaR (Zelikman et al. 2022)

**Docs to read live, as of the week you use them:**
- JAX: sharding / distributed arrays tutorial, `shard_map`, profiler, Pallas, install guide, default matmul precision, GPU memory allocation, persistent compilation cache
- Flax NNX: LoRA, Optimizer and filters (`wrt`, `DiffState`), transforms (`nnx.vmap`, `nnx.jit`, `nnx.remat`), partitioning, performance guidance
- Optax
- Orbax
- `uv` (two-environment setup)
- Tunix, MaxText and the Gemma JAX code (as reference implementations)
- Cloud TPU docs (v5e/v6e, spot, quotas, pricing) and the TRC FAQ
- TRL and vLLM (only for parity, baseline and fallback use)
- EvalPlus / MBPP / GSM8K dataset cards (splits, licenses)