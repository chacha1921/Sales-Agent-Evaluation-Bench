# Tenacious-Bench v0.1 — Sales Agent Evaluation Bench

Domain-specific evaluation benchmark for Tenacious's B2B sales agent. Builds on Week 10's Conversion Engine baseline to answer: **does the agent actually work for Tenacious's segments, voice, and failure modes — not just generic web-agent tasks?**

The Week 10 audit found five categories of failure that τ²-Bench retail scores completely, because τ²-Bench was built for e-commerce slot-filling, not B2B outbound sales. Tenacious-Bench v0.1 closes all five gaps with machine-verifiable rubric dimensions.

---

## Current Status (Interim — 2026-04-29)

| Act | Deliverable | Status |
|---|---|---|
| Act I | Audit memo (5 gaps, 8 probes, 5 traces) | ✅ Complete |
| Act I | Task schema (JSON, 3 annotated examples) | ✅ Complete |
| Act I | Scoring evaluator (7 checkers, bootstrap CI) | ✅ Complete |
| Act I | Methodology draft (Path A declaration) | ✅ Complete |
| Act II | Generation pipeline (200 tasks, 4 modes) | ✅ Complete |
| Act II | Judge filter (200/200 passed ≥3.5/5) | ✅ Complete |
| Act II | Contamination checks (n-gram PASS, timeshift PASS) | ✅ Complete |
| Act II | Dataset splits (train=99, dev=63, held_out=38) | ✅ Complete |
| Act II | Datasheet (7 Gebru sections, Pushkarna layers) | ✅ Complete |
| Act II | Synthesis memos (τ²-Bench, LIMA) | ✅ Complete |
| Act III | Path-specific paper memos, SFT training data format | 🔄 In progress |
| Act IV | LoRA training run, ablations, model card | ⏳ Pending |
| Act V | HuggingFace publish, blog, executive memo | ⏳ Pending |

---

## What's Next

- **Act III (Day 4):** Format training data as instruction-output pairs. Write `papers/path_specific/path_a/` memos for Tülu 3 and LIMA. Write `methodology_rationale.md`.
- **Act IV (Days 5–6):** Run `training/train.py` LoRA fine-tune on T4 (fp16). Evaluate trained model vs. Week 10 baseline on dev split. Run held-out ablation.
- **Act V (Day 7):** Publish HuggingFace dataset (train + dev only; held-out sealed). Publish LoRA adapter. Write blog post and executive memo with Delta A result.

---

## Quick Start (reproduce in < 1 hour)

**Requirements:** Python 3.10+, pip

```bash
git clone <repo-url>
cd Sales-Agent-Evaluation-Bench

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Score a sample task (no API key needed)

```bash
# Run the built-in demo against 3 hand-authored tasks
python evaluation/scoring_evaluator.py --demo

# Score the dev split in mock mode (no LLM calls)
python evaluation/scoring_evaluator.py --split dev --mock-llm
```

### Regenerate the dataset from scratch

```bash
python generation/scripts/trace_derived.py --mock
python generation/scripts/programmatic.py --mock
python generation/scripts/multi_llm_synthesis.py --mock
python generation/scripts/adversarial.py
python generation/judge_filter.py --mock
python generation/contamination_check.py --skip-embedding
```

---

## Repository Layout

```
audit/                   Act I  — gap memo comparing τ²-Bench to Tenacious requirements
dataset/                 Act II — task schema, banned phrases list, datasheet, IRA protocol
  tenacious_bench_v0.1/  200 task JSONL files split into train / dev / held_out
evaluation/              Act I  — scoring evaluator with 7 machine-verifiable checkers
generation/              Act II — dataset authoring scripts, judge filter, contamination check
  scripts/               Four generation scripts (trace_derived, programmatic, multi_llm, adversarial)
  raw_tasks/             Per-script JSONL outputs + filtered.jsonl
papers/                  Required reading memos (common + path-specific)
synthesis_memos/         Critical memos on τ²-Bench and LIMA design choices
training/                Acts III–IV — methodology, cost log, training pipeline
  training_data/         Formatted SFT pairs (to be populated in Act III)
week10_artifacts/        Read-only seeds — probe library, failure taxonomy, trace log
```

---

## Key Artifacts

| Artifact | Path |
|---|---|
| Audit memo (5 gaps, τ²-Bench vs Tenacious) | [audit/audit_memo.md](audit/audit_memo.md) |
| Task schema + 3 annotated examples | [dataset/schema.json](dataset/schema.json) |
| Datasheet (Gebru + Pushkarna) | [dataset/datasheet.md](dataset/datasheet.md) |
| Scoring evaluator | [evaluation/scoring_evaluator.py](evaluation/scoring_evaluator.py) |
| Methodology (Path A, contamination results) | [training/methodology.md](training/methodology.md) |
| Contamination report | [generation/contamination_check.json](generation/contamination_check.json) |
| Synthesis memo: τ²-Bench | [synthesis_memos/tau2bench_memo.md](synthesis_memos/tau2bench_memo.md) |
| Synthesis memo: LIMA | [synthesis_memos/lima_memo.md](synthesis_memos/lima_memo.md) |
| Inter-rater agreement protocol | [dataset/inter_rater_agreement.md](dataset/inter_rater_agreement.md) |
| Cost log | [training/cost_log.md](training/cost_log.md) |
| HuggingFace Dataset | _to be added after Act V_ |
| HuggingFace Model adapter | _to be added after Act V_ |

---

## Reproducibility

All scripts set `random.seed(42)` and `np.random.seed(42)`. LLM judge calls use `temperature=0.0`.
Re-running `scoring_evaluator.py --demo` on a fresh clone produces scores within ±2 percentage points.

Dataset generation is fully deterministic in `--mock` mode (no API calls, template expansion only).

---

## Cost Budget

Total cap: **$10**. Actual spend to date: **$0.00** (all mock mode).
See [training/cost_log.md](training/cost_log.md) for itemized log and remaining estimates (~$0.06–$0.46).

---

## License

Dataset: CC-BY-4.0. Model adapter (to be published): inherits base model license (Apache 2.0).
