# Tenacious-Bench v0.1 — Sales Agent Evaluation Bench

Domain-specific evaluation benchmark for Tenacious's B2B sales agent. Built because τ²-Bench retail (pass@1=0.7267) completely missed five categories of production failure — the agent was using banned phrases in 38% of runs while scoring 0.82 on τ²-Bench.

**Headline result:** ORPO fine-tuning on 381 preference pairs lifted held-out mean from **4.008 → 4.462 (+11.3%)**, Δ=+0.454, p=0.001, beating a prompt-engineered baseline by Δ=+0.290, p=0.021.

**Published:** [Blog post](https://chalielijalem.substack.com/p/building-the-sales-evaluation-bench) | [τ²-Bench gap report (issue #287)](https://github.com/sierra-research/tau2-bench/issues/287)

---

## Status — Complete (2026-05-02)

| Act | Deliverable | Status |
|---|---|---|
| Act I | Audit memo (5 gaps, 8 probes, 5 traces) | ✅ |
| Act I | Task schema + scoring evaluator (7 checkers) | ✅ |
| Act II | 230-task dataset (4 authoring modes, 3 contamination checks) | ✅ |
| Act II | Datasheet (Gebru + Pushkarna), IRA κ=1.000 | ✅ |
| Act III | 381 live preference pairs, methodology rationale | ✅ |
| Act IV | ORPO R4 fine-tune (44 min, T4), Delta A/B/C ablations | ✅ |
| Act V | Model card, memo, blog post, evidence graph | ✅ |

---

## Quick Start — reproduce in 10 minutes

**Requirements:** Python 3.10+, no GPU, no API key

```bash
git clone https://github.com/chacha1921/Sales-Agent-Evaluation-Bench
cd Sales-Agent-Evaluation-Bench

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Score 3 hand-authored tasks (zero setup):**

```bash
python evaluation/scoring_evaluator.py --demo
```

Expected output:
```
Task: DEMO-001  Aggregate: 4.33/5.0  PASS ✓
Task: DEMO-002  Aggregate: 3.75/5.0  PASS ✓
Task: DEMO-003  Aggregate: 4.17/5.0  PASS ✓
```

**Score the full dev split (71 tasks, heuristic mode, no API key):**

```bash
python evaluation/scoring_evaluator.py --split dev --mock-llm
```

**Inspect the ablation results from the live Colab run:**

```python
import json
d = json.load(open("ablation_results.json"))
print(f"Delta A: Δ={d['delta_a']['observed_delta']}  p={d['delta_a']['p_value']}")
# → Delta A: Δ=0.4538  p=0.001
```

---

## Dataset

| Property | Value |
|---|---|
| Tasks | 230 (train=127, dev=71, held_out=32 sealed) |
| Segments | SMB=72, Series B=77, Enterprise=81 |
| Task types | email_outreach, follow_up, discovery_response, objection_handling, closing |
| Authoring modes | trace_derived=30, programmatic=75, multi_llm=90, adversarial=35 |
| Rubric dimensions | 7 machine-verifiable checkers |
| Contamination | n-gram PASS, embedding PASS, time-shift PASS |
| IRA κ | 1.000 (Round 2, after two-tier tone rule fix) |
| License | CC BY 4.0 |
| HuggingFace | [Chalie-lijalem/tenacious-bench-v0.1](https://huggingface.co/datasets/Chalie-lijalem/tenacious-bench-v0.1) |

---

## Model

| Property | Value |
|---|---|
| Base model | Qwen3-4B (4-bit, via Unsloth) |
| Training method | ORPO — Odds Ratio Preference Optimization |
| LoRA rank | r=32, α=32 (66M trainable params, 1.62%) |
| Training data | 381 live preference pairs |
| Delta A | +0.454 [0.153, 0.787] p=0.001 vs Week 10 |
| Delta B | +0.290 [0.012, 0.583] p=0.021 vs prompt-eng |
| HuggingFace | [Chalie-lijalem/tenacious-orpo-qwen3-4b](https://huggingface.co/Chalie-lijalem/tenacious-orpo-qwen3-4b) |

---

## Key Artifacts

| Artifact | Path |
|---|---|
| Audit memo (5 gaps, τ²-Bench vs Tenacious) | [audit/audit_memo.md](audit/audit_memo.md) |
| Task schema + 3 annotated examples | [dataset/schema.json](dataset/schema.json) |
| Datasheet (Gebru + Pushkarna) | [dataset/datasheet.md](dataset/datasheet.md) |
| Scoring evaluator (7 checkers) | [evaluation/scoring_evaluator.py](evaluation/scoring_evaluator.py) |
| Preference pairs (381, live) | [training/training_data/path_b_dpo/preference_pairs.jsonl](training/training_data/path_b_dpo/preference_pairs.jsonl) |
| ORPO training script | [training/train_orpo.py](training/train_orpo.py) |
| Ablation runner | [training/run_ablations.py](training/run_ablations.py) |
| **Ablation results (real, mock_mode=False)** | [ablation_results.json](ablation_results.json) |
| **Held-out traces (32 tasks × 3 arms)** | [held_out_traces.jsonl](held_out_traces.jsonl) |
| Training run log | [training_run.log](training_run.log) |
| Model card | [model_card.md](model_card.md) |
| CEO/CFO memo | [memo.md](memo.md) |
| Evidence graph | [evidence_graph.json](evidence_graph.json) |
| Technical blog post | [blog_post.md](blog_post.md) — [Published on Substack](https://chalielijalem.substack.com/p/building-the-sales-evaluation-bench) |
| Inter-rater agreement | [dataset/inter_rater_agreement.md](dataset/inter_rater_agreement.md) |
| Contamination report | [generation/contamination_check.json](generation/contamination_check.json) |
| τ²-Bench Week 10 baseline | [week10_artifacts/score_log.json](week10_artifacts/score_log.json) |

---

## Reproducing Training (Google Colab T4)

```python
# Install
!pip install unsloth trl datasets peft bitsandbytes

# Clone repo and pull preference pairs
!git clone https://github.com/chacha1921/Sales-Agent-Evaluation-Bench
%cd Sales-Agent-Evaluation-Bench

# Train ORPO (~44 min on T4)
!python training/train_orpo.py

# Compare ORPO vs SimPO on dev split
!python training/compare_methods.py \
    --orpo-adapter  runs/orpo/adapter \
    --simpo-adapter runs/simpo/adapter

# Run held-out ablations (upload held_out/tasks.jsonl first — it is sealed locally)
!python training/run_ablations.py \
    --winner orpo \
    --adapter runs/orpo/adapter \
    --mock-llm
```

---

## Repository Layout

```
audit/               Act I  — gap memo (5 τ²-Bench gaps with trace evidence)
dataset/             Act II — schema, banned phrases, datasheet, IRA protocol
  tenacious_bench_v0.1/  train / dev splits (held_out sealed, not pushed)
evaluation/          Act I  — scoring evaluator, 7 machine-verifiable checkers
generation/          Act II — 4 authoring scripts, judge filter, contamination check
training/            Acts III–IV — methodology, preference pairs, training scripts
week10_artifacts/    Read-only seeds — probe library, failure taxonomy, trace log
ablation_results.json     Act IV — Delta A/B/C with bootstrap CIs (mock_mode=False)
held_out_traces.jsonl     Act IV — per-task traces, 32 tasks × 3 arms
training_run.log          Act IV — hyperparameters + ORPO loss curve
model_card.md             Act V  — HuggingFace model card
memo.md                   Act V  — CEO/CFO decision memo (2 pages)
blog_post.md              Act V  — Technical blog post draft
evidence_graph.json       Act V  — machine-readable claim → evidence index
```

---

## Cost

| Item | Cost |
|---|---|
| Dataset generation (Gemini Flash, multi-LLM tasks) | $0.021 |
| Preference pair generation (381 live pairs) | ~$0.04 |
| Training (Google Colab free T4) | $0.00 |
| **Total** | **~$0.06** |

---

## License

Dataset: **CC BY 4.0**.  
Code: **MIT**.  
Model adapter: **Apache 2.0** (inherits from Qwen3 base model).

---

## Credits

- **Author & Dataset Design:** Chalie Lijalem — chalie@10academy.org (Tenx MCP Week 11)
- **Base model:** Qwen3-4B (Alibaba Cloud) via [Unsloth](https://github.com/unslothai/unsloth)
- **Training framework:** [TRL](https://github.com/huggingface/trl) (ORPO trainer), [PEFT](https://github.com/huggingface/peft)
- **Dataset generation:** Gemini 2.5 Flash (Google), DeepSeek Chat (via OpenRouter)
- **Evaluation infrastructure:** HuggingFace Hub, Google Colab (T4 GPU)
