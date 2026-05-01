# Methodology Rationale — Path A: Supervised Fine-Tuning

**Version:** 1.0 | **Date:** 2026-04-30 | **Author:** Tenacious-Bench team

---

## Why Path A over Paths B and C

### Failure mode distribution as the decision criterion

The Week 10 evaluation produced a failure taxonomy across 200 tasks:

| Failure mode | Count | Share | Trainable? | Path |
|---|---|---|---|---|
| `signal_missing` | 36 | 36% | Yes — model outputs exist; wrong content | **A** |
| `tone_drift` | 33 | 33% | Yes — model outputs exist; wrong style | **A** |
| `formulaic` | 13 | 13% | Yes — model uses banned openers | **A** |
| `trajectory` | 13 | 13% | Partially — multi-turn sequence errors | B or C |
| `constraint_violation` | 4 | 4% | Yes — word count / no-pricing rule missed | A |

**Signal_missing + tone_drift + formulaic = 82% of failures**. All three are generation-quality problems on *single-turn* tasks: the model outputs a complete response but with wrong content, wrong style, or a banned phrase. These are exactly the problems that SFT on targeted examples addresses.

**Path B (DPO)** would require preference pairs: (chosen, rejected) for the same input. Our benchmark does not naturally produce rejection samples — the evaluation pipeline scores outputs, but generating plausible-but-wrong completions to pair against gold outputs adds authoring complexity without benefit. DPO's theoretical advantage over SFT is most pronounced when the model already produces acceptable outputs and needs fine-grained preference tuning. Week 10 shows categorical failures (banned phrases, no signal reference) — not subtle preference gaps.

**Path C (PRM)** requires dense step-level reward modeling across multi-turn trajectories. The majority of Tenacious tasks are single-turn (email_outreach, follow_up, objection_handling, closing, discovery_response). Trajectory failures are 13% of the total. Building a process reward model for 13% of failures while ignoring 82% single-turn failures is not efficient.

**Path A is the right choice** for this failure distribution.

---

## Training Data Design

### Dataset statistics

| Split | Tasks | SFT pairs | Source |
|---|---|---|---|
| train | 99 | ~990 | 10 gold outputs × 99 tasks |
| dev | 63 | — | Evaluation only |
| held_out | 38 | — | Sealed; unlocked once for ablation |

### Failure-mode coverage in training split

| Failure mode | Train tasks | Share | SFT pairs |
|---|---|---|---|
| `signal_missing` | 36 | 36% | ~360 |
| `tone_drift` | 33 | 33% | ~330 |
| `formulaic` | 13 | 13% | ~130 |
| `trajectory` | 13 | 13% | ~130 |
| `constraint_violation` | 4 | 4% | ~40 |

Training distribution mirrors Week 10 failure frequencies. The model sees proportionally more signal_missing and tone_drift examples — exactly the failure modes it needs to improve on.

### Gold output quality gates

All SFT pairs are generated using `training/generate_sft_data.py`. Each gold output must satisfy:
- `signal_grounding_fn` ≥ 0.5 (entity overlap with prospect context)
- `banned_phrase_fn` = 1.0 (zero prohibited phrases)
- `cta_checker_fn` = 1.0 ([CALENDLY_LINK] present)
- `word_count_fn` = 1.0 (under specified limit)
- `pricing_mention_fn` = 1.0 (no pricing language)

Outputs failing any gate are discarded and regenerated. In live mode (`--live`), Claude Haiku generates 10 diverse variations per task; variations failing the rubric gate are dropped. In mock mode (`--mock`), template-based generation is used for testing only.

### SFT format

Training pairs are stored in OpenAI ChatML format (compatible with Unsloth/TRL):
```json
{
  "messages": [
    {"role": "system",    "content": "TENACIOUS_SYSTEM_PROMPT"},
    {"role": "user",      "content": "TASK_INSTRUCTION"},
    {"role": "assistant", "content": "GOLD_OUTPUT"}
  ]
}
```

The system prompt encodes the Tenacious brand voice rules: signal-grounding requirement, banned phrase list, single-CTA constraint, no pricing.

---

## Paper Evidence

### Why SFT is sufficient (Tülu 3)

Lambert et al. (2024) show that RLVR adds significant gains only on tasks with binary verifiable rewards (math, code). For open-ended generation — our domain — the SFT baseline accounts for 85–95% of final quality. None of Tenacious's evaluation dimensions produce a binary verifiable reward suitable for RLVR or DPO. SFT on targeted data is the appropriate method.

See `papers/path_specific/path_a/tulu3_memo.md` for full analysis.

### Why ~990 examples are sufficient (LIMA)

Zhou et al. (2023) demonstrate that 1,000 carefully curated instruction-output pairs substantially shift output style without degrading general capability. Our ~990 SFT pairs match this threshold. We diverge from LIMA's diversity-first curation by instead concentrating training density in the signal_missing and tone_drift failure modes. Density in the failure-mode space outperforms diversity across task types when the training goal is to suppress a strong pretraining prior (banned phrases are common in the base model's training data).

See `papers/path_specific/path_a/lima_memo.md` for full analysis.

### Why synthetic generation is valid (Magpie)

Xu et al. (2024) validate model-self-generated instruction-output pairs as high-quality training data for alignment. Our multi-LLM synthesis authoring mode (60/200 tasks) is a domain-specific implementation of Magpie: a generation model creates both the scenario and a reference output; a distinct judge model filters for quality. The propose/judge model separation prevents self-reinforcement.

See `papers/path_specific/path_a/magpie_memo.md` for full analysis.

---

## Expected Outcomes (Act IV)

The fine-tuned model should improve primarily on the two dimensions most represented in training data:

| Metric | Week 10 baseline (expected) | Target (post-SFT) |
|---|---|---|
| `signal_grounding_fn` mean (dev) | ~0.50 | ≥ 0.70 |
| `tone_checker_fn` mean (dev) | ~0.60 | ≥ 0.80 |
| `banned_phrase_fn` mean (dev) | ~0.75 | ≥ 0.90 |
| Aggregate score mean (dev) | ~2.5 / 5.0 | ≥ 3.5 / 5.0 |

**Success criterion (Act IV):** Δ aggregate score > 0 with bootstrap CI p < 0.05 on dev split. Improvement must be attributable to signal_missing and/or tone_drift dimensions, consistent with training data distribution.

**Null hypothesis:** Fine-tuning on ~990 pairs on a single-task domain with targeted failure-mode coverage does not improve aggregate score vs. the Week 10 baseline. Rejected if p < 0.05.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Catastrophic forgetting (model loses general capability) | Medium | Evaluate on held-out task types not in training split; LoRA rank ≤ 16 to limit capacity change |
| Template memorization (model outputs near-literal training templates) | Medium (mock mode only) | Use live mode for final training; verify output diversity on dev set |
| Insufficient examples for 7B model | Medium | Increase n_per_task from 10 to 20 if dev score < 3.0 after first run |
| T4 OOM during training | Low | Use fp16 + gradient checkpointing; batch size 4, sequence length 512 |
