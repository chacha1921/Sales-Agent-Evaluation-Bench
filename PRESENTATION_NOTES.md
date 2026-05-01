# Presentation Reference — Tenacious-Bench v0.1
**Week 11 | Tenx MCP Programme | chalie@10academy.org**

---

## 0. Implementation Decisions Log (what changed and why)

| Area | Original | Changed To | Why |
|---|---|---|---|
| Generation model | `google.generativeai` (deprecated) + `gemini-2.0-flash` | `google.genai` SDK + `gemini-2.5-flash` | Old SDK end-of-life; `gemini-2.0-flash` returned 404 NOT_FOUND |
| Output truncation | `max_output_tokens=400` | `max_output_tokens=1500` + `thinking_budget=0` | JSON contexts were cut off mid-string; thinking tokens consumed output budget |
| Multi-LLM contamination | No seed grouping | `seed_id` (S01–S18) in `_profile_key()` | Gemini generates unique text per variant; without grouping, variants split across partitions → n-gram FAIL + embedding FAIL |
| Task ID collision | adversarial TB-0166–0200 | adversarial TB-0196–0230 | Expanded multi_llm to 90 tasks (TB-0106–0195) caused 30 duplicate IDs in filtered.jsonl |
| IRA Round 1 | κ=0.662 on `tone_checker_fn` | Two-tier system (tier-1 = immediate FAIL) | Mock heuristic gave partial credit to "just checking in"; below 0.70 threshold |
| Training path | Path A (SFT) in methodology_rationale.md | Path B (ORPO + SimPO) | 40–60% trigger rate = inconsistency, not capability gap; SFT teaches new behavior, not preference consistency |
| Backbone model | `Qwen/Qwen2.5-0.5B-Instruct` | `unsloth/Qwen3-4B-bnb-4bit` | 0.5B too small; Qwen3.5 IDs don't exist on HuggingFace (404 error in Colab) |
| LoRA alpha | 32 (2× r) | 16 (= r) | Unsloth Qwen3 guide: alpha == r |
| LoRA dropout | 0.05 | 0 | Unsloth recommendation |
| Qwen3 thinking | Not handled | `enable_thinking=False` | Qwen3 outputs `<think>` tokens by default; suppressed for sales emails |
| seq length | 512 | 2048 | Unsloth guide + multi-constraint prompts exceed 512 tokens |

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

### ORPO Training Arguments

```python
beta = 0.1                 # odds-ratio weight λ
learning_rate = 5e-5
optim = "adamw_8bit"       # Unsloth 8-bit Adam
batch_size = 4
gradient_accumulation = 4  # effective batch = 16
epochs = 3
lr_scheduler = "cosine"
warmup_ratio = 0.1
fp16 = True                # T4 does not support bf16
max_length = 2048
max_prompt_length = 1024
```

### SimPO Training Arguments (differences from ORPO)

```python
beta = 2.0                 # SimPO temperature (not the same scale as ORPO β)
simpo_gamma = 1.0          # target reward margin — minimum gap between chosen/rejected
# All other args same as ORPO
```

**Why both?** ORPO combines SFT loss + preference loss in one pass (conservative, stable). SimPO uses length-normalized reward + margin (more decisive outputs, better for word-count-constrained tasks). Run both on dev split, pick winner for the held-out ablation.

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
