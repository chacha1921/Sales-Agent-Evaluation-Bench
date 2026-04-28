# Tenacious-Bench v0.1 — Sales Agent Evaluation Bench

Domain-specific evaluation benchmark for Tenacious's B2B sales agent. Builds on Week 10's
Conversion Engine to answer: **does the agent actually work for Tenacious's segments, voice,
and failure modes?**

## Quick Start (reproduce in < 1 hour)

```bash
git clone <repo-url>
cd Sales-Agent-Evaluation-Bench
pip install -r requirements.txt

# Score baseline agent on the public dev split
python evaluation/scoring_evaluator.py --split dev --agent week10

# Run contamination checks before sealing held-out
python generation/contamination_check.py

# Run full ablation on held-out (Day 6 only, after sealing)
python evaluation/scoring_evaluator.py --split held_out --agent trained --compare week10
```

## Repository Layout

```
audit/                   Act I  — gap memo + τ²-Bench analysis
dataset/                 Act II — Tenacious-Bench v0.1 tasks
generation/              Act II — dataset authoring + filtering scripts
training/                Acts III–IV — LoRA training pipeline
evaluation/              Act IV — scoring evaluator + ablation results
papers/                  Required reading memos (common + path-specific)
publishing/              Act V  — blog post, executive memo, evidence graph
week10_artifacts/        Read-only seeds from Week 10 (trace_log, probes, taxonomy)
```

## Artifact Links

| Artifact | URL |
|---|---|
| HuggingFace Dataset | _to be added after Act V_ |
| HuggingFace Model   | _to be added after Act V_ |
| Blog Post           | _to be added after Act V_ |
| Community Engagement| _to be added after Act V_ |

## Reproducibility

All scripts set `random.seed(42)` and `transformers.set_seed(42)`.
Re-running `scoring_evaluator.py` on a fresh clone produces scores within ±2 percentage points.

## Cost Budget

Total cap: **$10**. See [training/cost_log.md](training/cost_log.md) for itemized log.

## License

Dataset: CC-BY-4.0. Model adapter: inherits Qwen 3.5 license (Apache 2.0).
