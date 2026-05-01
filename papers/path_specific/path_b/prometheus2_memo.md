# Path B Memo: Prometheus 2 (Kim et al., 2024)

**Paper:** Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models (Kim et al., 2024)
**Role:** Reference for open judge model design — validates using a small fine-tuned model as an evaluator

---

## Key Contribution

Prometheus 2 is a 7B open-source model fine-tuned specifically to evaluate other language models. It is trained on a large set of (instruction, response, rubric, score, feedback) tuples. The key results: Prometheus 2 achieves judge agreement comparable to GPT-4 on fine-grained evaluation tasks, at a fraction of the cost, when the rubric is explicit.

The paper makes two points directly relevant to this project:

1. **Rubric specificity is the dominant factor.** A small model with a precise rubric outperforms a large model with a vague rubric. The rubric must specify what counts as a 1, 3, and 5 — not just define the dimension name.

2. **Feedback generation improves calibration.** Models that are trained to produce a natural-language justification before the score are better calibrated than models that output a score directly. The reasoning step forces the model to engage with the rubric.

---

## Application to Tenacious-Bench

### Why Prometheus 2 validates the judge design

The scoring evaluator uses Claude Haiku as the LLM judge for `tone_checker_fn` and `objection_ack_fn`. Haiku is not a specialist judge model — it is a general-purpose model asked to score against an explicit rubric. Prometheus 2's results validate this approach: general-capability models with explicit rubrics perform comparably to specialist judge models, provided the rubric is calibrated.

The rubric in `scoring_evaluator.py` specifies concrete anchors (what a 0.0, 0.5, and 1.0 output looks like for each dimension). This matches Prometheus 2's finding that rubric specificity drives calibration quality.

### What Prometheus 2 suggests we are missing

Prometheus 2 generates a natural-language feedback string before scoring. Our judge prompt requests only a score (`SCORES: tone=N, objection_ack=N`). Adding a required rationale field before the score would:
1. Improve judge calibration (Prometheus 2's finding)
2. Produce interpretable error logs for debugging

This is a low-cost improvement for the live evaluation mode. The mock mode is deterministic and does not need it.

### Relevance to Path B training data

Prometheus 2 is trained from preference pairs: (better response, worse response, rubric) → judge score. The `training/generate_preference_pairs.py` output is structurally identical to Prometheus 2's training format. If future work involves training a specialist Tenacious judge (rather than using a general-purpose model), the preference pairs generated here could serve as training data for that judge — following the Prometheus 2 recipe directly.

---

*~380 words | Key finding: rubric specificity > model size for LLM judges. Application: validates Haiku-as-judge with explicit rubric. Gap: no reasoning chain before score. Future: preference pairs usable as Prometheus-style judge training data.*
