# Presentation Reference — Tenacious-Bench v0.1
**Week 11 | Tenx MCP Programme | chalie@10academy.org**

---

## 0. Implementation Decisions Log (what changed and why)

Full chronological trace of every issue hit and how it was resolved.

---

### Act II — Dataset Generation Issues

| # | Issue | Error / Symptom | Fix |
|---|---|---|---|
| 1 | Gemini SDK deprecated | `google.generativeai` printed end-of-life warning | Migrated to `google.genai` SDK across all three files |
| 2 | Wrong model ID | `gemini-2.0-flash` → `404 NOT_FOUND` "no longer available to new users" | Changed to `gemini-2.5-flash` everywhere |
| 3 | JSON output truncated | `"Unterminated string starting at line 3 column 14"` — context field cut off mid-sentence | `max_output_tokens=400 → 1500`; added `thinking_budget=0` (thinking tokens were consuming output quota) |
| 4 | Gemini 503 errors | `"This model is currently experiencing high demand"` | Added retry loop: 3 attempts, sleep 10s × attempt number on 503/429 |
| 5 | Contamination checks FAIL after live generation | n-gram FAIL (11 violations) + embedding FAIL (27 pairs) | Root cause: Gemini generates unique context text per variant, so each of the 5 variants of the same seed was treated as an independent group and split across partitions. Fix: added `seed_id` field (S01–S18) to task metadata; updated `_profile_key()` to group by `seed_id` for multi_llm tasks |
| 6 | Task ID collision (30 duplicates in filtered.jsonl) | Duplicate task IDs in train split — 127 lines but only 117 unique IDs | Adversarial tasks still used old range TB-0166–0200 which overlapped with expanded multi_llm TB-0106–0195. Fix: renumbered adversarial to TB-0196–0230; rebuilt filtered.jsonl via `judge_filter.py --mock` |
| 7 | IRA Round 1 failed (κ=0.662) | Below κ=0.70 threshold on `tone_checker_fn` | Mock heuristic gave partial credit to "just checking in" and missed "My name is" opener. Fix: rewrote to two-tier system — tier-1 phrases = immediate FAIL. Round 2 achieved κ=1.000 |

---

### Act III — Training Path and Data Issues

| # | Issue | Error / Symptom | Fix |
|---|---|---|---|
| 8 | Wrong training path documented | `methodology_rationale.md` argued for Path A (SFT) | Rewrote entirely for Path B — Week 10 40–60% trigger rate = inconsistency not capability gap; SFT teaches new behaviors, not preference consistency |
| 9 | preference_pairs.jsonl not in GitHub | `[ERROR] preference_pairs.jsonl not found` in Colab | `.gitignore` had `training/training_data/path_b_dpo/*.jsonl` blocking it. Removed that rule; committed and pushed the file |

---

### Act IV — Model and Training Issues

| # | Issue | Error / Symptom | Fix |
|---|---|---|---|
| 10 | Wrong initial backbone | `Qwen/Qwen2.5-0.5B-Instruct` — too small for nuanced tone tasks | Changed to `unsloth/Qwen3.5-4B-Instruct` |
| 11 | Qwen3.5 model does not exist | `RuntimeError: Unsloth: No config file found` in Colab | `unsloth/Qwen3.5-4B-Instruct` does not exist on HuggingFace. Unsloth docs only showed 27B/35B examples. Fixed to `unsloth/Qwen3-4B-bnb-4bit` (confirmed working, 97K+ downloads/month) |
| 12 | Qwen3 thinking mode | Training outputs would contain `<think>...</think>` tokens | Added `enable_thinking=False` to all `apply_chat_template()` calls; wrapped in try/except for older tokenizer versions |
| 13 | LoRA config not aligned with Unsloth guide | `lora_alpha=32`, `lora_dropout=0.05`, missing `use_gradient_checkpointing`, missing `optim` | Per Unsloth Qwen3 guide: `alpha=16` (= r), `dropout=0`, `use_gradient_checkpointing="unsloth"`, `random_state=3407`, `optim="adamw_8bit"` |
| 14 | `evaluation_strategy` deprecation warning | TRL printed deprecation warning | Renamed to `eval_strategy` |
| 15 | `warmup_ratio` deprecation warning | `warmup_ratio is deprecated and will be removed in v5.2` in Colab log | Changed to `warmup_steps=10` |
| 16 | CUDA OOM during ORPO backward pass | `torch.OutOfMemoryError: Tried to allocate 684.00 MiB. GPU 0 has 151.81 MiB free` at step 0 | Root cause: `max_length=2048` — ORPO holds chosen+rejected simultaneously, attention is O(n²). Fix: `max_length=512` (emails are <200 words), `batch_size=4→2`, `grad_accum=4→8` (effective batch stays 16) |
| 17 | `warmup_ratio` KeyError after training finished | `KeyError: 'warmup_ratio'` in `training_run.log` writer at line 208 — crashed after adapter was already saved | `ORPO_ARGS` was changed to `warmup_steps=10` (issue #15 fix) but the log writer still referenced the old key. Fixed: `lf.write(f"warmup_ratio: ...")` → `lf.write(f"warmup_steps: ...")`. Adapter was unaffected. |
| 18 | Round 1 preference learning did not activate (mock pairs) | `log_odds_ratio=-0.6931` constant, `rewards/margins≈0` for all 45 steps in both ORPO and SimPO | Root cause: mock-generated preference pairs share ~90% tokens — chosen and rejected are identical templates. Five changes for Round 2: (1) `--live` flag added to `generate_preference_pairs.py` (Gemini Flash, ~$0.03); (2) `--n-rejected` default 1→3 (381+ pairs); (3) LoRA r=16→32, alpha=16→32 (33M→66M trainable params); (4) epochs default 3→5; (5) SimPO gamma 1.0→2.0. |
| 19 | Mock pairs used in Round 2 (live pairs not generated before training) | Round 2 ORPO/SimPO still showed zero preference separation for same root cause | Ran `generate_preference_pairs.py --live --n-rejected 3` locally (macOS). Gemini Flash generated 381 genuinely diverse pairs (127 tasks × 3 rejected variants). Token overlap dropped from ~90% (mock) to **avg 16.8%** (range 4.8–26.8%). File committed and pushed. Round 3 training will be the first run with real preference signal. |
| 20 | Round 3 ORPO (live pairs) still showed zero preference separation | `logps/chosen` = `logps/rejected` identically across all 5 epochs and 110 steps — same symptom as all previous runs despite 16.8% token overlap | Root cause: `to_hf_dataset` passed full conversations (prompt + response) in `chosen`/`rejected`. Shared prompt tokens (~70% of sequence) dominated per-token logps average, making chosen ≈ rejected. Fix: changed to TRL canonical format — `prompt` = formatted prompt with `add_generation_prompt=True`; `chosen`/`rejected` = response text only. Trainer now computes logps over response tokens exclusively. Fixed in both `train_orpo.py` and `train_simpo.py`. |
| 21 | Qwen3.5-4B-bnb-4bit still not on HuggingFace | `Unsloth: No config file found` when trying `unsloth/Qwen3.5-4B-bnb-4bit` | Auto-fallback to `unsloth/Qwen3-4B-bnb-4bit` triggered cleanly. Unsloth docs list Qwen3.5 0.8B/2B/4B as T4-compatible but the pre-quantized HF repos don't exist yet. Will retry when Unsloth publishes them. |

---

## 1. The Problem We Solved

The Week 10 evaluation of Tenacious's B2B sales agent revealed:

| Failure Mode | Frequency | Business Impact |
|---|---|---|
| `tone_drift` — banned phrases, corporate jargon | 38% | $10.8K/10 leads |
| `signal_missing` — ignores prospect trigger | 29% | Lost trust, no reply |
| `trajectory` — ignores objection history | 21% | Deal loss |
| `formulaic` — "just checking in", "hope this finds you" | 8% | Spam filters, unsubscribes |
| `constraint_violation` — over word limit, pricing on first touch | 4% | Brand policy breach |

**Key insight:** The agent fails at 40–60% trigger rates on the same task types it also passes. This is a *consistency* problem, not a capability gap. The model already knows how to write correct outputs — it just does not do so reliably.

**τ²-Bench retail baseline (Week 10):** pass@1 = 0.7267, CI [0.6504, 0.7917] — general-purpose benchmark masked all five failure modes because it only measures task completion, not brand voice or signal grounding.

---

## 2. Dataset Construction — Tenacious-Bench v0.1

### Why a custom benchmark?
τ²-Bench retail scored 0.7267 on our agent while the agent produced banned phrases in 40%+ of runs. A benchmark that misses your failure modes cannot guide improvement. We built Tenacious-Bench to measure exactly what was failing.

### Dataset at a glance

| Property | Value |
|---|---|
| Total tasks | 230 (all passed judge filter) |
| Splits | train=127 / dev=71 / held_out=32 |
| Segments | SMB=72, Series B=77, Enterprise=81 |
| Task types | email_outreach, follow_up, discovery_response, objection_handling, closing |
| Rubric dimensions | 7 machine-verifiable checker functions |
| Total cost | $0.021 |

### 4 Authoring Modes

**1. Trace-derived (30 tasks, TB-0001–0030)**
— Started from 5 real Week 10 agent failures. Each trace became 6 task variants (different task types + difficulty levels). Ensures the benchmark tests the exact scenarios the agent already failed on.

**2. Programmatic (75 tasks, TB-0031–0105)**
— 15 prospect profiles × 5 task types = 75 tasks. Every profile (name, role, company, segment, pain point, verified signal) crossed with all five task types. Maximises task-type diversity while controlling for scenario confounds.

**3. Multi-LLM synthesis (90 tasks, TB-0106–0195)**
— 18 seed scenarios × 5 variation configs. Bulk seeds sent to Gemini 2.5-flash; hard seeds (adv_weight=1.0) sent to DeepSeek Chat via OpenRouter. Introduces lexical variety that templates cannot provide.

**4. Adversarial (35 tasks, TB-0196–0230)**
— Hand-authored traps: leverage/synergy word traps (8), pricing mention traps (6), trajectory/voice consistency traps (7), formulaic opener traps (6), constraint precision traps (8). All difficulty=hard, adversarial_weight=1.0.

### Dataset Quality Controls

**Judge filtering** — Every task scored by an LLM judge on 3 dimensions (input_coherence, ground_truth_verifiability, rubric_clarity). Cross-family routing: Gemini-generated tasks judged by DeepSeek, DeepSeek-generated tasks judged by Gemini. Threshold: mean ≥ 3.5/5. Result: 230/230 passed.

**3 contamination checks** — All PASS:
- Check 1 (n-gram): Zero shared 8-grams between held_out and train+dev
- Check 2 (embedding): Zero pairs with cosine similarity ≥ 0.85 (`all-MiniLM-L6-v2`)
- Check 3 (time-shift): All 108 public-signal tasks have non-null `signal_time_window`

**Inter-rater agreement** — 30-task stratified dev sample. Round 1 failed (κ=0.662 on tone_checker_fn). Root cause: mock heuristic gave partial credit to "just checking in". Fixed with two-tier system (tier-1 = immediate FAIL). Round 2 achieved **κ=1.000** on all rater pairs.

---

## 3. Training Data — Path B: Preference Learning

### Why Path B over Path A (SFT) or Path C (PRM)?

| Path | What it does | Why not chosen |
|---|---|---|
| **A — SFT** | Teaches new behavior from demonstrations | Model already knows correct behavior; SFT adds demonstrations but doesn't fix inconsistency |
| **B — ORPO + SimPO** ✓ | Teaches the model to *prefer* its correct outputs over its incorrect ones | Directly addresses the consistency problem diagnosed in Week 10 |
| **C — PRM** | Step-level process rewards for multi-turn | Multi-turn (trajectory) = only 21% of failures; wrong priority |

**Paper evidence:**
- *Tülu 3* (Lambert et al., 2024): SFT accounts for ≥85% of quality on non-verifiable tasks, but the residual inconsistency is not addressable by SFT alone.
- *LIMA* (Zhou et al., 2023): ~1,000 high-quality preference pairs sufficient to shift output style.
- *ORPO* (Hong et al., 2024): Eliminates reference model via log-odds ratio penalty in the SFT loss.
- *SimPO* (Meng et al., 2024): Length-normalized reward + target margin γ prevents gaming via output shortening.

### Preference Pair Format (TRL/Unsloth compatible)

```json
{
  "prompt":   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen":   [{"role": "assistant", "content": "clean signal-led output"}],
  "rejected": [{"role": "assistant", "content": "output with banned phrase / missing signal"}]
}
```

### Preference Pair Quality

| Metric | Value |
|---|---|
| Total pairs | 254 (127 train tasks × 2 rejected variants) |
| Source | Train split only — dev and held_out untouched |
| Chosen: zero banned phrases | 254/254 (100%) |
| Rejected: ≥1 banned phrase | 232/254 (91%) |

| Failure mode | Pairs | Week 10 frequency |
|---|---|---|
| `signal_missing` | 122 | 29% |
| `tone_drift` | 68 | 38% |
| `trajectory` | 32 | 21% |
| `formulaic` | 20 | 8% |
| `constraint_violation` | 12 | 4% |

---

## 4. Model Fine-Tuning

### Backbone: Qwen3 on Google Colab T4

#### Implementation trace — what we tried and why it changed

**Attempt 1 — `Qwen/Qwen2.5-0.5B-Instruct`**
Initial default in all training scripts. Too small (0.5B) for nuanced tone compliance on B2B email tasks. Changed before first live run.

**Attempt 2 — `unsloth/Qwen3.5-4B-Instruct`**
Updated to Qwen3.5 after reading the Unsloth Qwen3.5 fine-tuning guide which listed 0.8B / 2B / 4B as T4-compatible. Hit this error in Colab:
```
RuntimeError: Unsloth: No config file found - are you sure the `model_name` is correct?
```
Root cause: `unsloth/Qwen3.5-4B-Instruct` does not exist on HuggingFace. The Unsloth documentation only showed large models (27B, 35B MoE); the small-model page used the name loosely. Confirmed by checking HuggingFace directly.

**Attempt 3 — `unsloth/Qwen3-4B-bnb-4bit` ✓ (current)**
Confirmed working on HuggingFace (97K+ downloads/month). Qwen3 is the same generation as what the docs described; `bnb-4bit` suffix = pre-quantized by Unsloth, loads faster than runtime quantization. Also discovered Qwen3 has a **thinking mode** that outputs `<think>...</think>` tokens — added `enable_thinking=False` to all `apply_chat_template` calls to suppress this for sales email generation.

| Model | VRAM (4-bit) | Use case |
|---|---|---|
| `unsloth/Qwen3-0.6B-bnb-4bit` | ~2 GB | Fast iteration / dry runs |
| `unsloth/Qwen3-1.7B-bnb-4bit` | ~4 GB | Mid-size sweep |
| `unsloth/Qwen3-4B-bnb-4bit` | ~8 GB | **Default — best quality that fits T4** |

### LoRA Configuration (aligned with Unsloth Qwen3 guide)

#### Implementation trace

**Initial config:**
- `lora_alpha=32` (2× r), `lora_dropout=0.05` — standard defaults from generic LoRA tutorials
- Missing `use_gradient_checkpointing`, `random_state`, `optim`

**After reading Unsloth Qwen3 guide:**
- `lora_alpha=16` (= r) — Unsloth recommends alpha == r for Qwen3
- `lora_dropout=0` — Unsloth explicitly prefers 0
- Added `use_gradient_checkpointing="unsloth"` — Unsloth's custom memory optimization, required for T4
- Added `random_state=3407` — Unsloth's recommended seed for reproducibility
- Added `optim="adamw_8bit"` — 8-bit Adam saves ~2GB VRAM on T4
- `max_seq_length=512 → 2048`, `max_prompt_length=256 → 1024` — 512 was too short for multi-turn sales conversations; Unsloth guide uses 2048

```python
r = 16
lora_alpha = 16            # alpha == r (Unsloth recommendation)
lora_dropout = 0           # 0 preferred by Unsloth
bias = "none"
use_gradient_checkpointing = "unsloth"   # Unsloth memory optimisation
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### ORPO Training Arguments (actual values used in Colab run)

```python
# Preference loss
beta = 0.1                       # odds-ratio weight λ (ORPO paper default)

# Sequence lengths — reduced from 2048 to fit T4 (issue #16)
max_length = 512                  # emails are <200 words; 2048 caused OOM
max_prompt_length = 256

# Batch / accumulation — reduced batch from 4→2 to fit T4 (issue #16)
per_device_train_batch_size = 2
gradient_accumulation_steps = 8   # effective batch = 2 × 8 = 16
num_train_epochs = 3

# Optimiser
learning_rate = 5e-5
optim = "adamw_8bit"              # Unsloth 8-bit Adam saves ~2 GB VRAM
lr_scheduler_type = "cosine"
warmup_steps = 10                 # changed from warmup_ratio (issue #15)

# Hardware
fp16 = True                       # T4 does not support bf16
```

### SimPO Training Arguments (differences from ORPO)

```python
loss_type = "simpo"               # CPOTrainer with SimPO mode
beta = 2.0                        # SimPO temperature β (different scale to ORPO β)
simpo_gamma = 1.0                 # target reward margin γ — key SimPO hyperparameter
# All sequence, batch, optimiser, and hardware args identical to ORPO above
```

**Why both?** ORPO combines SFT loss + preference loss in one pass (conservative, stable). SimPO uses length-normalized reward + margin (more decisive outputs, better for word-count-constrained tasks). Run both on dev split, pick winner for the held-out ablation.

### ORPO Training Results — First Live Run (Colab T4, 2026-05-02)

#### Hardware and environment

| Property | Value |
|---|---|
| GPU | Tesla T4 (1×), 14.563 GB VRAM |
| Platform | Linux (Google Colab) |
| PyTorch | 2.10.0+cu128 |
| CUDA version | 7.5 (Toolkit 12.8) |
| Unsloth | 2026.4.8 |
| Transformers | 5.5.0 |
| TRL trainer | ORPOTrainer (UnslothORPOTrainer patch) |
| Precision | fp16 (T4 has no bf16 support) |
| Flash Attention | FA2 = False; Xformers = 0.0.35 |

#### Model and parameter counts

| Property | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-bnb-4bit` |
| Total parameters | 4,055,498,240 (4.06B) |
| Trainable parameters (LoRA) | 33,030,144 (33M = **0.81%** of total) |
| Frozen parameters | 4,022,468,096 (99.19%) |
| LoRA rank r | 16 |
| LoRA alpha α | 16 (= r, per Unsloth guide) |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 modules × 36 layers) |
| LoRA dropout | 0 |
| Gradient checkpointing | `"unsloth"` (custom Unsloth memory optimisation) |

#### Training data and steps

| Property | Value |
|---|---|
| Total preference pairs | 254 |
| Train split | 228 (90%) |
| Eval split | 26 (10%) |
| Epochs | 3 |
| Total steps | 45 |
| Batch size per device | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 (2 × 8 × 1 GPU) |
| Steps per epoch | 15 |
| Eval strategy | per epoch (3 evals total) |
| Wall time | 1,197s (~20 min) |
| Samples/second | 0.571 |

#### Loss curve

| Epoch | Step | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|---|
| 0.35 | 7 | 3.951 | 3.882 | — | — |
| 0.70 | 14 | 3.634 | 3.565 | — | — |
| **1** | 15 | 3.145 | 3.105 | **2.994** | 2.923 |
| 1.35 | 22 | 2.848 | 2.779 | — | — |
| 1.70 | 28 | 2.608 | 2.539 | — | — |
| **2** | 30 | 2.390 | 2.335 | **2.311** | 2.242 |
| 2.35 | 37 | 2.242 | 2.172 | — | — |
| 2.70 | 42 | 2.154 | 2.085 | — | — |
| **3** | 45 | 2.140 | 2.073 | **2.149** | 2.081 |

#### Preference metrics (all 45 steps)

| Metric | All steps | Expected if working |
|---|---|---|
| `log_odds_ratio` | -0.6931 (constant) | Should rise toward 0 |
| `rewards/margins` | ≈ 0 (2e-10 to 6e-9) | Should grow positive |
| `rewards/accuracies` | 0.10–0.20 (random) | Should approach 0.7+ |
| `logps/chosen` | = `logps/rejected` at every step | Should diverge |

#### Interpretation

- **SFT component works** — NLL loss dropped 3.882 → 2.073 (46% reduction). The model learned to generate sales email structure.
- **Preference component did not activate** — `log_odds_ratio` is stuck at exactly `-log(2) = -0.6931` for all 45 steps. Root cause: mock pairs share ~90% of tokens (same Python template; rejected = chosen + banned phrase prepended). Chosen and rejected have equal log-probabilities so the odds-ratio loss gets no gradient signal.
- **Adapter is a valid SFT baseline.** True ORPO preference separation requires live-generated pairs with genuine output diversity.
- **Next step if repeating:** run `generate_preference_pairs.py --live` (Gemini Flash, ~$0.03, 254 real pairs) before retraining.

**Output:** Adapter saved → `runs/orpo/checkpoint-45/` (also `checkpoint-15`, `checkpoint-30`)

**Post-run crash:** `KeyError: 'warmup_ratio'` in log writer (issue #17) — did not affect adapter. Fixed and pushed.

---

### ORPO Training Results — Round 2 (r=32, 5 epochs, Colab T4, 2026-05-02)

#### Changes from Round 1

| Parameter | Round 1 | Round 2 |
|---|---|---|
| LoRA rank r | 16 | **32** |
| LoRA alpha α | 16 | **32** |
| Trainable parameters | 33,030,144 (0.81%) | **66,060,288 (1.62%)** |
| Total parameters | 4,055,498,240 | 4,088,528,384 |
| Epochs | 3 | **5** |
| Total steps | 45 | **75** |
| Preference pairs | mock (254) | mock (254) — **live not run yet** |

#### Loss curve

| Epoch | Step | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|---|
| 0.35 | ~5 | 3.921 | 3.852 | — | — |
| 0.70 | ~10 | 3.401 | 3.332 | — | — |
| **1** | 15 | 2.874 | 2.842 | **2.641** | 2.572 |
| 1.35 | ~22 | 2.434 | 2.365 | — | — |
| 1.70 | ~28 | 2.056 | 1.986 | — | — |
| **2** | 30 | 1.648 | 1.607 | **1.434** | 1.367 |
| 2.35 | ~37 | 1.268 | 1.199 | — | — |
| 2.70 | ~42 | 0.968 | 0.899 | — | — |
| **3** | 45 | 0.804 | 0.749 | **0.747** | 0.682 |
| 3.35 | ~52 | 0.695 | 0.626 | — | — |
| 3.70 | ~57 | 0.633 | 0.563 | — | — |
| **4** | 60 | 0.603 | 0.533 | **0.611** | 0.545 |
| 4.35 | ~67 | 0.585 | 0.516 | — | — |
| 4.70 | ~72 | 0.578 | 0.509 | — | — |
| **5** | 75 | 0.550 | 0.485 | **0.594** | 0.529 |

#### Preference metrics

| Metric | Epochs 1–2 | Epochs 3–5 |
|---|---|---|
| `log_odds_ratio` | -0.6931 (constant) | -0.6931 (constant) |
| `rewards/margins` | ≈ 0 (tiny floats) | **exactly 0** (zero) |
| `rewards/accuracies` | 0.09–0.18 (random) | **exactly 0** |

#### Round 1 vs Round 2 comparison

| Metric | Round 1 | Round 2 | Change |
|---|---|---|---|
| Trainable params | 33M (0.81%) | 66M (1.62%) | 2× |
| Final train_loss | 2.140 | **0.550** | −74% |
| Final NLL (train) | 2.073 | **0.485** | −77% |
| Final eval_loss | 2.149 | **0.594** | −72% |
| Final NLL (eval) | 2.081 | **0.529** | −75% |
| Wall time | 1,197s | 2,312s | +93% |
| Preference separation | None | None | No change |

#### Interpretation

- **NLL loss dropped dramatically** — 3.852 → 0.485 (87% reduction). The model has thoroughly learned the sales email structure and the Tenacious brand voice pattern. This is strong SFT learning.
- **Preference metrics went to exactly 0 at epoch 3**, then stayed there. This is the model converging so hard on the NLL task that log-probabilities become near-zero for both chosen and rejected. The mock pairs are so similar that the model assigns near-perfect probability to both — the ORPO odds-ratio term gets a numerically zero gradient.
- **Overfitting signal:** eval_loss bottomed at epoch 3 (0.747), then slightly rebounded (0.611 at epoch 4, 0.594 at epoch 5). Epoch 3 checkpoint (`checkpoint-45`) is the best for generalisation; epoch 5 is the most overfit to training templates.
- **Best checkpoint for next steps:** `runs/orpo/checkpoint-45/` (epoch 3 — lowest eval_loss 0.747).
- **Preference learning still not activated.** Root cause is unchanged: mock pairs. Round 3 requires running `generate_preference_pairs.py --live` first.

**Output:** Adapter saved → `runs/orpo/adapter/` (also `checkpoint-15`, `checkpoint-30`, `checkpoint-45`, `checkpoint-60`, `checkpoint-75`)

---

### ORPO Training Results — Round 3 (live pairs, r=32, 5 epochs, Colab T4, 2026-05-02)

#### Changes from Round 2

| Parameter | Round 2 | Round 3 |
|---|---|---|
| Preference pairs | mock 254 (228 train) | **live 381 (342 train / 39 eval)** |
| Token overlap (chosen vs rejected) | ~90% | **avg 16.8% (range 4.8–26.8%)** |
| Data format bug | present | present (not yet fixed — fixed after this run) |

#### Loss curve

| Epoch | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|
| 0.23 | 3.941 | 3.872 | — | — |
| 0.70 | 2.878 | 2.808 | — | — |
| **1** | — | — | **2.040** | 1.971 |
| 1.61 | 1.201 | 1.132 | — | — |
| **2** | — | — | **0.653** | 0.584 |
| **3** | — | — | **0.447** | 0.378 |
| **4** | — | — | **0.394** | 0.326 |
| **5** | 0.381 | 0.307 | **0.387** | 0.318 |
| avg | **1.137** | — | — | — |

*Wall time: 3,120s (~52 min) | Steps: 110 | Samples/sec: 0.548*

#### Preference metrics — all 110 steps

`logps/chosen` = `logps/rejected`, `rewards/margins` = 0, `rewards/accuracies` = 0, `log_odds_ratio` = -0.6931 throughout. Same pattern as all four previous runs.

#### Root cause identified (issue #20)

Live pairs have 16.8% token overlap — data quality is NOT the issue. The `to_hf_dataset` function was passing **full conversations** (prompt + response) in `chosen` and `rejected`. Both include the same prompt (~70% of tokens), so the per-token average log-prob is identical for chosen and rejected.

**Fix:** Changed to TRL canonical format — `prompt` = formatted prompt with `add_generation_prompt=True`; `chosen`/`rejected` = response text only. Trainer now computes logps over response tokens exclusively. Committed as `f9a8cde`.

#### All runs to date

| Run | Pairs | Data format | NLL (final eval) | Preference |
|---|---|---|---|---|
| R1-ORPO | mock 254 | full-conv (buggy) | 2.081 | None |
| R1-SimPO γ=1 | mock 254 | full-conv (buggy) | 1.130 | None |
| R2-ORPO | mock 254 | full-conv (buggy) | 0.529 | None |
| R2-SimPO γ=2 | mock 254 | full-conv (buggy) | 0.00082 | None |
| **R3-ORPO** | **live 381** | **full-conv (buggy)** | **0.318** | **None** |
| **R4-ORPO** | **live 381** | **response-only (fixed)** | **0.816** | **✓ ACTIVE** |

**Output:** Adapter saved → `runs/orpo/adapter/` (checkpoints at 22, 44, 66, 88, 110)

---

---

### ORPO Training Results — Round 4 ✓ BREAKTHROUGH (live pairs + data format fix, Colab T4, 2026-05-02)

#### Changes from Round 3

| Parameter | Round 3 | Round 4 |
|---|---|---|
| Data format | full-conv in chosen/rejected (buggy) | **response-only — TRL canonical format (fixed)** |
| Default model | `unsloth/Qwen3.5-4B-bnb-4bit` | same — **auto-falls back to `unsloth/Qwen3-4B-bnb-4bit`** (Qwen3.5 repos not on HF yet) |
| Preference pairs | live 381 | live 381 (unchanged) |
| LoRA rank r | 32 | 32 (unchanged) |
| Epochs | 5 | 5 (unchanged) |

#### Model and parameter counts

| Property | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-bnb-4bit` (Qwen3.5 fallback triggered — issue #21) |
| Total parameters | 4,088,528,384 |
| Trainable parameters (LoRA) | **66,060,288 (1.62%)** |
| LoRA rank r | 32 |
| LoRA alpha α | 32 (= r) |

#### Training data and steps

| Property | Value |
|---|---|
| Total preference pairs | 381 (live) |
| Train split | 342 (90%) |
| Eval split | 39 (10%) |
| Epochs | 5 |
| Total steps | 110 |
| Effective batch size | 16 (2 × 8) |
| Wall time | **2,670s (~44.5 min)** |

#### Loss curve

| Epoch | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|
| **1** | — | — | **0.973** | 0.947 |
| **2** | — | — | **0.848** | 0.823 |
| **3** | — | — | **0.843** | 0.820 |
| **4** | — | — | **0.841** | 0.819 |
| **5** | — | — | **0.836** | **0.816** |
| avg (train) | **1.602** | — | — | — |

#### Preference metrics progression — the breakthrough

| Epoch | `log_odds_ratio` | `rewards/margins` | `rewards/accuracies` | `log_odds_chosen` |
|---|---|---|---|---|
| 0 (start) | -1.076 | -0.056 | 0.2625 | -0.561 |
| **2** | — | — | **1.000** ← | — |
| **5** (end) | **-0.211** | **+0.139** | **1.000** | **+1.741** |

Key signals:
- **`rewards/accuracies` hit 1.000 by epoch 2** — model correctly ranks chosen above rejected on 100% of eval pairs.
- **`rewards/margins` went from -0.056 → +0.139** — positive margin means chosen probability exceeds rejected by the required threshold.
- **`log_odds_chosen` rose from -0.561 → +1.741** — model is now assigning substantially higher log-probability to chosen responses.
- **`log_odds_ratio` rose from -1.076 → -0.211** — still negative (chosen is shorter than rejected on average so length-normalized odds still favor rejected slightly) but direction is strongly positive.

#### All rounds comparison

| Run | Pairs | Format | NLL (final eval) | Preference | Wall time |
|---|---|---|---|---|---|
| R1-ORPO | mock 254 | full-conv | 2.081 | None | 1,197s |
| R1-SimPO γ=1 | mock 254 | full-conv | 1.130 | None | 1,195s |
| R2-ORPO | mock 254 | full-conv | 0.529 | None | 2,312s |
| R2-SimPO γ=2 | mock 254 | full-conv | 0.00082 | None | 2,319s |
| R3-ORPO | live 381 | full-conv | 0.318 | None | 3,120s |
| **R4-ORPO** ✓ | **live 381** | **response-only** | **0.816** | **✓ ACTIVE** | **2,670s** |

#### Interpretation

- **First run with genuine preference learning in all 5 rounds.** The fix was entirely in `to_hf_dataset` — one-line change from full conversation to response-only text in `chosen`/`rejected`.
- **Eval NLL higher than Round 3** (0.816 vs 0.318) — expected. The ORPO objective now allocates gradient toward preference separation, not just NLL minimization. The model is trading some SFT fidelity for preference discrimination. This is correct behavior.
- **`rewards/accuracies` = 1.000 from epoch 2 onward** — model fully separated chosen from rejected on all 39 eval pairs by epoch 2. Epochs 3–5 continued refining magnitude (margins grew from -0.056 to +0.139) while accuracy stayed at ceiling.
- **eval_loss still decreasing at epoch 5** (0.848 → 0.836) — no overfitting signal. Best checkpoint is `checkpoint-110` (final epoch).
- **Next step:** Run SimPO Round 4 with the same data format fix. Then `compare_methods.py` → `run_ablations.py --winner`.

**Output:** Adapter saved → `runs/orpo/adapter/` (checkpoints at 22, 44, 66, 88, 110)

---

### SimPO Training Results — First Live Run (Colab T4, 2026-05-02)

#### Hardware and environment

Same as ORPO run (same Colab session): Tesla T4 / 14.563 GB / Unsloth 2026.4.8 / Transformers 5.5.0 / TRL CPOTrainer (`loss_type="simpo"`) / fp16.

#### Model and parameter counts

| Property | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-bnb-4bit` |
| Total parameters | 4,055,498,240 (4.06B) |
| Trainable parameters (LoRA) | 33,030,144 (33M = **0.81%** of total) |
| LoRA rank r | 16 |
| LoRA alpha α | 16 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

#### Training data and steps

| Property | Value |
|---|---|
| Total preference pairs | 254 |
| Train split | 228 (90%) |
| Eval split | 26 (10%) |
| Epochs | 3 |
| Total steps | 45 |
| Batch size per device | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Steps per epoch | 15 |
| Wall time | 1,195s (~20 min) |
| Samples/second | 0.572 |

#### SimPO-specific hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `beta` | 2.0 | Temperature scaling the reward signal (higher = more decisive) |
| `simpo_gamma` | 1.0 | Target reward margin γ — minimum gap required between chosen/rejected scores |
| Loss type | `simpo` via CPOConfig | Length-normalised average log-likelihood reward, no reference model |

#### Loss curve

| Epoch | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|
| 0.70 | 4.866 | 3.553 | — | — |
| **1** | — | — | **3.716** | 2.403 |
| 1.35 | 3.761 | 2.443 | — | — |
| 2.00 | 2.990 | 1.697 | — | — |
| **2** | — | — | **2.678** | 1.366 |
| 2.70 | 2.544 | 1.230 | — | — |
| **3** | — | — | **2.442** | 1.130 |
| avg | **3.419** | — | — | — |

*SimPO's CPO loss is a log-sigmoid term — the raw loss scale is higher than ORPO's (which includes a cross-entropy SFT term). Compare `nll_loss` across both: ORPO NLL 3.882→2.073, SimPO NLL 3.553→1.130. SimPO achieved lower final NLL because length normalisation down-weights longer sequences.*

#### Preference metrics (all 45 steps)

| Metric | All steps | Expected if working |
|---|---|---|
| `rewards/margins` | ≈ 0 (1e-9 to 3e-8) | Should grow toward γ=1.0 |
| `rewards/accuracies` | 0.07–0.27 (random) | Should approach 0.7+ |
| `logps/chosen` | = `logps/rejected` at every step | Should diverge |

#### ORPO vs SimPO side-by-side (Round 1 — mock pairs, 3 epochs, r=16)

| Metric | ORPO R1 | SimPO R1 | Winner |
|---|---|---|---|
| Final train_loss | 2.140 | 3.419 (CPO scale) | — (different loss scales) |
| Final NLL (train) | 2.073 | 1.230 | SimPO |
| Final NLL (eval) | 2.081 | 1.130 | SimPO |
| NLL reduction | 46% | 68% | SimPO |
| Preference separation | None | None | Tie |
| Wall time | 1,197s | 1,195s | Tie |

**Output:** Adapter saved → `runs/simpo/adapter/` (also `checkpoint-15`, `checkpoint-30`, `checkpoint-45`)

---

### SimPO Training Results — Round 2 (r=32, 5 epochs, γ=2.0, Colab T4, 2026-05-02)

#### Changes from Round 1

| Parameter | Round 1 | Round 2 |
|---|---|---|
| LoRA rank r | 16 | **32** |
| LoRA alpha α | 16 | **32** |
| Trainable parameters | 33M (0.81%) | **66M (1.62%)** |
| Epochs | 3 | **5** |
| `simpo_gamma` γ | 1.0 | **2.0** |
| Preference pairs | mock (254) | mock (254) — live not run yet |

#### Loss curve

| Epoch | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|
| 0.70 | 5.496 | 3.369 | — | — |
| **1** | — | — | **4.016** | 1.890 |
| 1.35 | 4.080 | 1.947 | — | — |
| 2.00 | 2.905 | 0.815 | — | — |
| **2** | — | — | **2.368** | 0.241 |
| 2.70 | 2.203 | 0.076 | — | — |
| **3** | — | — | **2.129** | 0.002315 |
| 3.35 | 2.130 | 0.003135 | — | — |
| 4.00 | 2.128 | 0.001046 | — | — |
| **4** | — | — | **2.128** | 0.000919 |
| 4.70 | 2.128 | 0.000795 | — | — |
| **5** | — | — | **2.128** | 0.000823 |
| avg | **2.951** | — | — | — |

#### Preference metrics

| Metric | Epochs 1–2 | Epochs 3–5 |
|---|---|---|
| `rewards/margins` | ≈ 0 (tiny floats) | **exactly 0** |
| `rewards/accuracies` | 0.02–0.18 (random) | **exactly 0** |
| `logps/chosen` | = `logps/rejected` | = `logps/rejected` |

#### Key technical finding — the SimPO γ=2.0 floor

SimPO's CPO loss has the form `L = -log σ(r_chosen - r_rejected - γ)` where r is the length-normalised log-probability reward. When chosen and rejected are identical sequences (mock pairs), `r_chosen = r_rejected` and the loss simplifies to:

```
L = -log σ(0 - γ) = -log σ(-γ) = log(1 + e^γ)
```

With γ=2.0: `log(1 + e²) = log(1 + 7.389) = log(8.389) ≈ **2.127**`

The training loss plateaued at exactly **2.128** from epoch 3. This is the mathematical floor — no gradient signal can push it lower with identical chosen/rejected. This is a clean proof that mock pairs cannot train SimPO.

With γ=1.0 (Round 1): floor = `log(1 + e¹) ≈ 1.313` — but Round 1 only ran 3 epochs and NLL hadn't collapsed yet, so the combined loss was higher.

#### Round 1 vs Round 2 — SimPO comparison

| Metric | Round 1 (γ=1.0, r=16, 3ep) | Round 2 (γ=2.0, r=32, 5ep) |
|---|---|---|
| Final NLL (train) | 1.230 | **0.00079** — near-zero (memorized) |
| Final NLL (eval) | 1.130 | **0.00082** — near-zero |
| CPO loss plateau | Not reached | **2.128** (mathematical floor) |
| Wall time | 1,195s | 2,319s |
| Preference separation | None | None |

#### All four runs side-by-side (Rounds 1 & 2)

| Run | Method | r | Epochs | Final NLL (eval) | Preference |
|---|---|---|---|---|---|
| R1-ORPO | ORPO | 16 | 3 | 2.081 | None |
| R1-SimPO | SimPO γ=1 | 16 | 3 | 1.130 | None |
| R2-ORPO | ORPO | 32 | 5 | **0.529** | None |
| R2-SimPO | SimPO γ=2 | 32 | 5 | **0.00082** | None |

**Interpretation:**
- SimPO NLL collapsed to near-zero by epoch 3 — essentially memorized all 228 training templates. The eval NLL is also near-zero because the eval templates are generated by the same mock function.
- **ORPO Round 2 is the better SFT baseline** (eval NLL 0.529 vs 0.00082 for SimPO). SimPO's extreme NLL collapse means it will generate very peaked, repetitive outputs at inference.
- **Best checkpoint overall: ORPO R2 checkpoint-45** (epoch 3, eval_loss=0.747) — good SFT learning without template memorisation.
- None of the four runs activated preference learning. All root causes trace to mock pairs.

**Output:** Adapter saved → `runs/simpo/adapter/` (also `checkpoint-15`, `checkpoint-30`, `checkpoint-45`, `checkpoint-60`, `checkpoint-75`)

---

### SimPO Training Results — Round 4 ✓ BREAKTHROUGH (live pairs + data format fix, Colab T4, 2026-05-02)

#### Changes from Round 2

| Parameter | Round 2 | Round 4 |
|---|---|---|
| Preference pairs | mock 254 | **live 381 (342 train / 39 eval)** |
| Data format | full-conv (buggy) | **response-only — TRL canonical (fixed)** |
| Token overlap | ~90% | **avg 16.8%** |
| LoRA rank r | 32 | 32 (unchanged) |
| Epochs | 5 | 5 (unchanged) |

*(Round 3 was skipped for SimPO — the data format fix was applied directly to Round 4 after the ORPO R3/R4 diagnosis.)*

#### Training data and steps

| Property | Value |
|---|---|
| Total preference pairs | 381 (live) |
| Train split | 342 (90%) |
| Eval split | 39 (10%) |
| Epochs | 5 |
| Total steps | 110 |
| Wall time | **2,598s (~43.3 min)** |
| Samples/second | 0.658 |

#### Loss curve

| Epoch | train_loss | nll_loss | eval_loss | eval_nll_loss |
|---|---|---|---|---|
| 0.47 | 7.500 | 4.498 | — | — |
| 0.94 | 3.970 | 2.327 | — | — |
| **1** | — | — | **2.641** | 1.979 |
| 1.37 | 2.427 | 1.937 | — | — |
| 1.84 | 1.845 | 1.696 | — | — |
| **2** | — | — | **1.585** | 1.500 |
| 2.28 | 1.511 | 1.441 | — | — |
| 2.75 | 1.425 | 1.374 | — | — |
| **3** | — | — | **1.382** | 1.357 |
| 3.19 | 1.276 | 1.257 | — | — |
| 3.66 | 1.229 | 1.211 | — | — |
| **4** | — | — | **1.338** | 1.325 |
| 4.09 | 1.153 | 1.147 | — | — |
| 4.56 | 1.170 | 1.149 | — | — |
| **5** | 1.152 | 1.150 | **1.330** | **1.318** |
| avg | **2.241** | — | — | — |

#### Preference metrics progression — the breakthrough

| Epoch | `rewards/accuracies` | `rewards/margins` | `logps/chosen` | `logps/rejected` |
|---|---|---|---|---|
| 0 (step 11) | 0.3000 | -0.8896 | -4.538 | -4.093 |
| 0.94 (step 22) | **0.7625** | **+0.677** | -2.338 | -2.677 |
| **1 (eval)** | **1.000** ← | **+2.336** | -2.002 | -3.170 |
| **2 (eval)** | 1.000 | **+5.950** | -1.508 | -4.483 |
| **3 (eval)** | 1.000 | **+7.707** | -1.367 | -5.220 |
| **4 (eval)** | 1.000 | **+8.636** | -1.336 | -5.654 |
| **5 (eval)** | 1.000 | **+8.704** | -1.329 | -5.681 |

Key signals:
- **`rewards/accuracies` hit 1.000 by epoch 1 eval** — even faster than ORPO R4 (which reached ceiling at epoch 2).
- **`rewards/margins` grew from -0.889 → +8.704** — SimPO's length-normalized reward with β=2.0 and γ=2.0 produces much larger raw margin numbers than ORPO's odds-ratio formulation (+0.139). These are not directly comparable — different loss scales.
- **`logps/chosen` and `logps/rejected` are diverging strongly** — chosen log-prob rising (−4.538 → −1.329), rejected falling (−4.093 → −5.681). The model is confidently separating the two distributions.
- **eval_loss still decreasing at epoch 5** (2.641 → 1.330, no plateau) — no overfitting. Best checkpoint is `checkpoint-110`.
- **eval_nll_loss: 1.318** — higher than ORPO R4's 0.816, but this is expected: SimPO's CPO loss prioritises the preference term over SFT fidelity more aggressively than ORPO.

#### All rounds final comparison — both methods

| Run | Method | Pairs | Format | NLL (eval) | `rewards/accuracies` | `rewards/margins` (final) |
|---|---|---|---|---|---|---|
| R1-ORPO | ORPO | mock 254 | full-conv | 2.081 | random | ≈0 |
| R1-SimPO | SimPO γ=1 | mock 254 | full-conv | 1.130 | random | ≈0 |
| R2-ORPO | ORPO | mock 254 | full-conv | 0.529 | random | exactly 0 |
| R2-SimPO | SimPO γ=2 | mock 254 | full-conv | 0.00082 | random | exactly 0 |
| R3-ORPO | ORPO | live 381 | full-conv | 0.318 | 0 | 0 |
| **R4-ORPO ✓** | **ORPO** | **live 381** | **response-only** | **0.816** | **1.000** | **+0.139** |
| **R4-SimPO ✓** | **SimPO γ=2** | **live 381** | **response-only** | **1.318** | **1.000** | **+8.704** |

#### Who wins?

Both methods reached `rewards/accuracies = 1.000`. To pick the winner for the held-out ablation, run:

```bash
python training/compare_methods.py \
    --orpo-adapter  runs/orpo/adapter \
    --simpo-adapter runs/simpo/adapter \
    --base-model    unsloth/Qwen3-4B-bnb-4bit
```

This scores both adapters on the **dev split** (71 tasks) and picks the one with higher mean rubric score. Then run `run_ablations.py --winner <orpo|simpo>` on the sealed held-out.

**Output:** Adapter saved → `runs/simpo/adapter/` (checkpoints at 22, 44, 66, 88, 110)

---

## 4c. Live Preference Pair Generation (2026-05-02)

### What changed

Ran `generate_preference_pairs.py --live --n-rejected 3` locally on macOS. Gemini Flash (`gemini-2.5-flash`) called twice per task per variant:
- **Chosen call**: Gemini writes a clean, signal-led email following all Tenacious rules (system prompt enforced).
- **Rejected call**: Gemini writes a deliberately bad email matching the task's `failure_mode_tag` (tone_drift, signal_missing, formulaic, trajectory, constraint_violation).

### Output stats

| Property | Mock pairs (Rounds 1–2) | Live pairs (Round 3) |
|---|---|---|
| Total pairs | 254 (127 × 2) | **381 (127 × 3)** |
| Generation method | Python templates | Gemini Flash API |
| Token overlap (Jaccard) | ~90% | **avg 16.8% (range 4.8–26.8%)** |
| Cost | $0 | ~$0.04 (Gemini Flash) |
| Generation time | <1s | ~12 min (local macOS) |

### Failure mode breakdown

| Failure mode | Pairs | % of total |
|---|---|---|
| `signal_missing` | 183 | 48% |
| `tone_drift` | 102 | 27% |
| `trajectory` | 48 | 13% |
| `formulaic` | 30 | 8% |
| `constraint_violation` | 18 | 5% |

### Expected impact on Round 3 training

With token overlap dropping from 90% → 17%, the ORPO/SimPO preference loss will have real gradient signal. Expected metrics in Round 3:
- `log_odds_ratio` (ORPO): rises from -0.6931 toward 0 over epochs
- `rewards/margins`: grows from 0 toward 0.3–1.0 by epoch 3
- `rewards/accuracies`: rises from random (0.15) toward 0.65–0.80

---

## 4b. Enhancement Plan — Round 2 (Before Next Training Run)

These are the concrete changes to make before re-running training. Listed by impact.

### Change 1 — Generate live preference pairs (HIGHEST IMPACT)

**What:** Replace mock Python-template pairs with Gemini Flash generated outputs.

**Why it matters:** Mock pairs share ~90% of tokens. ORPO/SimPO get zero preference gradient. Live pairs reduce token overlap to ~20-30%, activating the full preference loss.

**Expected change in training metrics:**
- `rewards/margins`: ≈0 → grows toward 0.5–1.5 by epoch 3
- `rewards/accuracies`: 0.10–0.27 (random) → 0.65–0.85
- `log_odds_ratio` (ORPO): stuck at -0.6931 → rises toward -0.3 to 0

**How to implement:**
```python
# Add --live flag to generate_preference_pairs.py
# Cost: ~$0.03 (254 pairs × Gemini Flash)
# Time: ~10-15 min in Colab
python training/generate_preference_pairs.py --live
# Then retrain:
python training/train_orpo.py
python training/train_simpo.py
```

**Code needed:** `--live` flag in `generate_preference_pairs.py` that calls Gemini Flash once per task (chosen = clean signal-led email; rejected = email with the task's specific failure mode).

---

### Change 2 — Increase rejected variants per task (DATA VOLUME)

**What:** `--n-rejected 3` generates 3 rejected variants per task instead of 1.

**Why it matters:** More pairs = more gradient updates per epoch. 254→762 pairs, steps go from 45 to 135 per run.

**How to implement:**
```python
python training/generate_preference_pairs.py --live --n-rejected 3
```

No code change required — flag already exists in `generate_preference_pairs.py`.

---

### Change 3 — Increase LoRA rank to r=32 (MODEL CAPACITY)

**What:** Double LoRA rank from r=16 to r=32. Trainable parameters: 33M → ~66M (still 1.6% of 4B — fits T4 at batch=2).

**Why it matters:** r=16 gives 33M trainable params to learn preference distinctions across 7 failure modes × 3 segments × 5 task types. r=32 doubles capacity for the same GPU cost.

**How to implement:** Change in both `train_orpo.py` and `train_simpo.py`:
```python
LORA_CONFIG = dict(
    r=32,          # was 16
    lora_alpha=32, # keep alpha == r
    ...
)
```

**Risk:** Slightly higher VRAM — monitor for OOM. If OOM, drop batch to 1 and double grad_accum to 16 (effective batch stays 16).

---

### Change 4 — Train more epochs (CONVERGENCE)

**What:** Increase epochs from 3 to 5.

**Why it matters:** With 228 pairs (45 steps at batch=16), 3 epochs is very short. With live pairs showing non-zero preference gradients, more epochs let the model fully converge. Standard preference training runs 3–5 epochs.

**How to implement:**
```python
python training/train_orpo.py --epochs 5
python training/train_simpo.py --epochs 5
```

**Watch for:** eval_loss increasing after epoch 3 = overfitting. If that happens, use `checkpoint-30` or `checkpoint-45` (best epoch, not final epoch).

---

### Change 5 — Tune SimPO gamma (FINE-TUNING SIMPO ONLY)

**What:** Increase `simpo_gamma` from 1.0 to 2.0.

**Why it matters:** γ = target reward margin. With γ=1.0 and mock pairs, the margin requirement was never reached. With live pairs, γ=2.0 forces stronger separation between chosen/rejected, producing more decisive outputs — important for word-count-constrained tasks where the model must consistently stay under limits.

**How to implement:**
```python
python training/train_simpo.py --gamma 2.0
```

---

### Recommended run order for Round 2

```bash
# Step 1 — generate live pairs (do once, covers both ORPO and SimPO)
python training/generate_preference_pairs.py --live --n-rejected 3

# Step 2 — train ORPO with higher rank
python training/train_orpo.py --epochs 5

# Step 3 — train SimPO with higher rank and tuned gamma
python training/train_simpo.py --epochs 5 --gamma 2.0

# Step 4 — pick winner on dev
python training/compare_methods.py \
    --orpo-adapter  runs/orpo/adapter \
    --simpo-adapter runs/simpo/adapter

# Step 5 — final held-out ablation
python training/run_ablations.py --winner <orpo|simpo> --adapter runs/<winner>/adapter
```

**Expected Round 2 outcome:** Delta A (trained vs Week 10 baseline) should be significant (p < 0.05) because the model will have genuine preference separation on the 5 failure modes, not just SFT learning.

---

### Day 5–6 Run Sequence

```bash
# Day 5 morning — train both
python training/train_orpo.py            # → runs/orpo/adapter + training_run.log
python training/train_simpo.py           # → runs/simpo/adapter (appends to log)

# Day 5 afternoon — pick winner on dev (71 tasks)
python training/compare_methods.py \
    --orpo-adapter  runs/orpo/adapter \
    --simpo-adapter runs/simpo/adapter   # → results/orpo_vs_simpo.json

# Day 6 — single held-out pass with winner
python training/run_ablations.py \
    --winner orpo \
    --adapter runs/orpo/adapter          # → ablation_results.json + held_out_traces.jsonl
```

---

## 5. Ablations — Act IV

### Four Measurements

**Delta A (PRIMARY — must pass):** Trained model vs Week 10 baseline on held_out.
- Requirement: Δ > 0, p < 0.05 on paired bootstrap (2,000 resamples).
- What it proves: training improved the agent on our benchmark.

**Delta B (honest test):** Trained model vs prompt-engineered baseline on same backbone.
- No LoRA — just a carefully crafted system prompt with all Tenacious style rules.
- If Δ < 0: prompt engineering alone was sufficient. **This is a legitimate publishable finding** — report it honestly.
- If Δ > 0: training added value beyond what prompting can achieve.

**Delta C (informational):** Tenacious-Bench trained score vs τ²-Bench Week 10 score.
- τ²-Bench Week 10: pass@1 = 0.7267
- Tests whether improvement is Tenacious-specific or general.
- **No τ²-Bench re-run** — reusing on-file numbers only.

**Cost-Pareto:** Per-task cost + latency with vs without the trained adapter.
- Week 10 baseline: $0.0199/simulation, p50 latency = 105.9s.
- LoRA inference on T4 is faster; a 3pp lift that triples cost scores worse than a 2pp lift at flat cost.

### Deliverables
- `ablation_results.json` — all four deltas with bootstrap CIs
- `held_out_traces.jsonl` — per-task output + dimension scores × 3 arms
- `training_run.log` — hyperparameters + loss curve

---

## 6. Key Numbers to Remember

| Metric | Value |
|---|---|
| Week 10 τ²-Bench pass@1 | 0.7267 [0.6504, 0.7917] |
| Week 10 agent cost/sim | $0.0199 |
| Dataset tasks | 230 (train=127, dev=71, held_out=32) |
| Dataset cost | $0.021 total |
| Preference pairs | 254 (127 tasks × 2) |
| IRA κ (Round 2) | 1.000 on all rater pairs |
| Contamination checks | 3/3 PASS |
| Training backbone | Qwen3-4B-bnb-4bit (T4, pre-quantized) |
| LoRA rank | r=16, α=16 |
| Training method | ORPO (β=0.1) vs SimPO (β=2.0, γ=1.0) |
| ORPO Round 1 (r=16, 3ep) | train_loss 2.140, eval_loss 2.149, wall 1197s |
| ORPO Round 2 (r=32, 5ep) | train_loss 0.550, eval_loss 0.594, wall 2312s |
| ORPO best checkpoint | checkpoint-45 (epoch 3, eval_loss 0.747) |
| ORPO preference separation | None (both rounds) — mock pairs required |
| SimPO R1 (r=16, 3ep, γ=1.0) | NLL eval 1.130, wall 1195s |
| SimPO R2 (r=32, 5ep, γ=2.0) | NLL eval 0.00082 (memorized), CPO floor=2.128 |
| SimPO γ=2 floor formula | log(1+e²)=2.127 — proved mock pairs give zero gradient |
| Best checkpoint overall | ORPO R2 checkpoint-45 (epoch 3, eval_loss=0.747) |
| ORPO R3 (live pairs, buggy format) | NLL eval 0.318, wall 3120s — still zero preference |
| **ORPO R4 BREAKTHROUGH (live + fixed format)** | **eval_loss 0.836, NLL 0.816, wall 2670s** |
| R4 `rewards/accuracies` | 0.2625 → **1.000** (reached epoch 2) |
| R4 `rewards/margins` | -0.056 → **+0.139** |
| R4 `log_odds_chosen` | -0.561 → **+1.741** |
| R4 best checkpoint | checkpoint-110 (epoch 5 — eval still decreasing) |
| Root cause of R1–R3 failure | `to_hf_dataset` passed full conversations; fix = response-only text in chosen/rejected |
| **SimPO R4 BREAKTHROUGH** | **eval_loss 1.330, NLL 1.318, wall 2598s** |
| SimPO R4 `rewards/accuracies` | 0.300 → **1.000** (reached epoch 1) |
| SimPO R4 `rewards/margins` | -0.889 → **+8.704** (vs ORPO +0.139 — different scales) |
| SimPO R4 best checkpoint | checkpoint-110 (eval still decreasing) |
| Next step | `compare_methods.py` on dev → `run_ablations.py --winner <X>` |
| Delta A requirement | p < 0.05, Δ > 0 on held_out (n=32) |

---

## 7. If Asked About Design Choices

**"Why not just prompt engineer?"**
Delta B tests exactly this. If the trained model doesn't beat the prompt-engineered baseline, we report it honestly. The honest answer is more valuable than a misleading result.

**"Why 254 preference pairs — isn't that small?"**
LIMA showed 1,000 pairs sufficient; we have 254. The compensating factor: every pair is targeted at a specific Week 10 failure mode rather than broad-coverage curation. Failure-mode density over task-type diversity.

**"Why Qwen3.5 and not a larger model?"**
Free T4 (16GB) is the constraint. Qwen3.5-4B at 4-bit quantization fills ~8GB, leaving room for training with batch size 4 + gradient checkpointing. A 7B model would require aggressive batch reduction and likely OOM on SimPO's longer sequences.

**"Why ORPO over DPO?"**
DPO requires a frozen reference model copy — doubles GPU memory. ORPO folds preference loss into the SFT cross-entropy step via log-odds ratio. No reference model = fits T4.

**"What if Delta A fails?"**
Report it. 32 held-out tasks is a small n — the CI will be wide. Document the direction, the CI, and the failure mode breakdown. "Training improved signal_missing but not tone_drift" is a specific, honest finding.
