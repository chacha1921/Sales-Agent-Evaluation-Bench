# Inter-Rater Agreement Protocol — Tenacious-Bench v0.1

## Purpose

This document defines the inter-rater agreement (IRA) protocol for `evaluation/scoring_evaluator.py`. The protocol establishes when human annotation is required, how disagreements are resolved, and what the minimum acceptable agreement thresholds are before results can be reported.

All seven checker functions in the scoring evaluator are designed to return deterministic scores without human judgment. However, two functions (`tone_checker_fn` and `objection_ack_fn`) delegate to an LLM judge when `--mock-llm` is not set. These LLM-backed scores require IRA verification before being used in published results.

---

## Trigger Condition

**IRA is required if and only if:**

> The mean pairwise Cohen's κ between any two raters across a random 30-task sample falls below **κ < 0.70** on the `tone_checker_fn` or `objection_ack_fn` dimension.

This is a non-negotiable condition. If triggered, results must not be finalized until the disagreement is resolved per §4.

*Rationale: κ ≥ 0.70 is the "substantial agreement" threshold per Landis & Koch (1977). Below this, the LLM judge scores are unreliable enough that the weighted composite score cannot be trusted.*

---

## Dimensions Subject to IRA Review

| Dimension | Checker | IRA Scope | Trigger κ |
|---|---|---|---|
| `tone_score` | `tone_checker_fn` | LLM-backed (--live mode only) | < 0.70 |
| `objection_ack` | `objection_ack_fn` | LLM-backed (--live mode only) | < 0.70 |
| `signal_grounding` | `signal_grounding_fn` | Deterministic regex + entity overlap | N/A |
| `banned_phrase` | `banned_phrase_fn` | Deterministic keyword list | N/A |
| `cta_present` | `cta_checker_fn` | Deterministic regex | N/A |
| `word_count_ok` | `word_count_fn` | Deterministic count | N/A |
| `no_pricing` | `pricing_mention_fn` | Deterministic regex | N/A |

Deterministic dimensions (rows 3–7) are by definition perfectly reproducible (κ = 1.0 between any two runs on the same model output) and do not require IRA.

---

## Sampling Protocol

When IRA verification is run:

1. Draw a **stratified random sample of 30 tasks** from the dev split (10 smb, 10 series_b, 10 enterprise).
2. Run `evaluation/scoring_evaluator.py` with `--split dev` on the 30 tasks to collect LLM judge scores.
3. Have **two independent annotators** each review the same 30 model outputs and assign binary scores (0 or 1) for `tone_checker_fn` and `objection_ack_fn` using the rubric criteria below.
4. Compute pairwise Cohen's κ between:
   - Annotator A vs. LLM judge
   - Annotator B vs. LLM judge
   - Annotator A vs. Annotator B
5. Report all three κ values in the matrix below.

---

## Scoring Rubric for Human Annotators

### `tone_checker_fn` — binary PASS criterion
Score 1 (pass) if and only if the output:
- Contains no pushy urgency phrases ("last chance", "don't miss out", "act now", "limited time")
- Contains no formulaic openers ("I hope this finds you well", "My name is X and I work at Y")
- Does not exceed one explicit ask in the email body

Score 0 (fail) otherwise.

### `objection_ack_fn` — binary PASS criterion
Score 1 (pass) if and only if the output:
- Explicitly acknowledges the prospect's stated objection (mirrors back the concern in different words)
- Does NOT immediately pivot to a counter-claim without acknowledgment
- Uses a softening phrase before pivoting ("That makes sense given...", "I hear you...", "Understood...")

Score 0 (fail) otherwise.

---

## Agreement Matrix Template

Run the IRA sampling protocol and fill in this matrix. Commit the completed matrix to this file before reporting final results.

| Rater Pair | Dimension | κ | Status |
|---|---|---|---|
| LLM judge vs. Annotator A | `tone_checker_fn` | _TBD_ | _TBD_ |
| LLM judge vs. Annotator B | `tone_checker_fn` | _TBD_ | _TBD_ |
| Annotator A vs. Annotator B | `tone_checker_fn` | _TBD_ | _TBD_ |
| LLM judge vs. Annotator A | `objection_ack_fn` | _TBD_ | _TBD_ |
| LLM judge vs. Annotator B | `objection_ack_fn` | _TBD_ | _TBD_ |
| Annotator A vs. Annotator B | `objection_ack_fn` | _TBD_ | _TBD_ |

**Mean κ (tone):** _TBD_ — **Mean κ (objection_ack):** _TBD_

---

## Disagreement Resolution

If κ < 0.70 on any dimension:

1. **Adjudication round**: A third annotator reviews all disagreed instances. Their score is final.
2. **Rubric update**: If adjudication reveals rubric ambiguity, update the criteria in §Scoring Rubric above and re-annotate the sample.
3. **LLM prompt update**: If the LLM judge systematically disagrees with human annotators, update `TONE_JUDGE_PROMPT` or `OBJECTION_JUDGE_PROMPT` in `evaluation/scoring_evaluator.py` and re-run.
4. **Flag in results**: If κ cannot be brought above 0.70 after one adjudication round, flag the affected dimension in the evaluation results as "low-agreement" and report results with and without that dimension's contribution.

---

## Relationship to Methodology

This protocol satisfies the IRA requirement stated in `training/methodology.md` §Inter-Rater Agreement. The trigger condition (κ < 0.70 on a 30-task sample) is non-negotiable per the methodology commitment. Results reported in `training/methodology.md` must reference the κ values from the completed agreement matrix above.

---

*Protocol version: 1.0 | Created: 2026-04-29 | Status: Matrix pending first evaluation run*
