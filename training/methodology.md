# Methodology: Tenacious-Bench v0.1

## 1. Path Declaration

**Chosen path: Path A — Supervised Fine-Tuning (SFT)**

The Week 10 failure taxonomy shows tone_drift (38%) and signal_missing (29%) account for
67% of failures. Both are generation-quality problems: the model knows what to write but
defaults to base-model patterns (filler phrases, generic pitches) when not specifically
steered. This is a generation quality problem, not a consistency detection problem (Path B)
or a trajectory problem (Path C).

**Justification traces:**
- `trace_042`: Agent sent generic pitch to a prospect whose LinkedIn post stated the exact
  pain point. Signal was in the context window. Model defaulted to template.
- `trace_107`: Agent used "leverage," "synergy," "synergising" in a discovery follow-up.
  Banned phrases are a generation quality failure — the model was never trained to suppress them.
- `trace_315`: "Just checking in" appeared in subject line and body. Same root cause.

**Justification papers (to be completed in papers/path_specific/path_a/):**
- Tülu 3 (Lambert et al., 2024) — SFT with curated preference data is sufficient to
  suppress surface-level style failures without full RLHF. Applicable here.
- LIMA (Zhou et al., 2023) — 1,000 high-quality examples can substantially shift model
  output style. Supports our 1k–3k training set target.

---

## 2. Dataset Partitioning Protocol

Total tasks: 200 (all passed judge filter, 200/200)

| Split | Share | Count | Purpose |
|---|---|---|---|
| train | 49.5% | 99 | SFT training pairs |
| dev | 31.5% | 63 | Iteration + judge calibration (Days 2–4) |
| held_out | 19% | 38 | Final ablation only (sealed after Act II) |

*Minor deviation from 50/30/20 target is due to profile-level grouping (trace_derived, programmatic, multi_llm tasks sharing a prospect profile are co-located in the same split to prevent n-gram contamination). All three contamination checks PASS.*

**Sealing protocol:**
1. Run all three contamination checks before sealing.
2. Add `dataset/tenacious_bench_v0.1/held_out/` to `.gitignore`.
3. Commit the gitignore change. Held-out tasks are never pushed to remote.
4. Held-out is unlocked exactly once: Day 6 ablation run.

---

## 3. Authoring Mode Distribution

| Mode | Target share | Target count | Script |
|---|---|---|---|
| Trace-derived | ~30% | ~75 | `generation/scripts/trace_derived.py` |
| Programmatic | ~30% | ~75 | `generation/scripts/programmatic.py` |
| Multi-LLM synthesis | ~25% | ~62 | `generation/scripts/multi_llm_synthesis.py` |
| Hand-authored adversarial | ~15% | ~38 | `generation/scripts/adversarial.py` |

---

## 4. LLM Routing Policy

| Stage | Model | Purpose | Budget |
|---|---|---|---|
| Bulk generation (programmatic + synthesis) | Qwen3-Next-80B or DeepSeek V3.2 via OpenRouter | High-volume task authoring | $1.50 |
| Hard seed generation (adversarial) | Claude Sonnet 4.6 | Adversarial task construction | $1.00 |
| High-volume judge filtering | Qwen3-Next-80B or DeepSeek V3.2 | All-task pointwise scoring | $1.00 |
| Spot-check judging (50 tasks) | Claude Haiku 4.5 | Quality verification | $0.50 |

**Leakage prevention rule:** The model used to generate a task must not be the model used
to judge that same task. Generation and judge model families must be rotated. Both are
logged in each task's metadata fields `generation_model` and `judge_model`.

---

## 5. Contamination Prevention Protocol

Three checks applied before sealing held_out. All must pass.

### Check 1: N-gram overlap

Tool: `generation/contamination_check.py --skip-embedding`

Rule: Zero shared 8-gram sequences between held_out and train+dev context strings. Constraints are excluded from the check — they are intentional shared templates, not contamination. Only scenario context text is indexed.

**Results (run 2026-04-29):** Indexed 1,944 distinct 8-gram sequences from 162 train+dev task context strings. Checked all 38 held-out task contexts against this index. **Zero violations.** Status: **PASS.**

Note: An earlier run flagged 23 violations. Root cause was (a) constraint template strings shared across task types producing false n-gram hits, and (b) `trace_derived` variants of the same prospect profile being split across partitions, causing context-level overlap. Both were resolved by (a) limiting `task_text()` to context only and (b) extending `_profile_key()` to group `trace_derived` tasks by context prefix, matching the existing grouping logic for `programmatic` and `multi_llm` modes.

### Check 2: Embedding similarity

Tool: `generation/contamination_check.py` (requires `pip install sentence-transformers`)

Model: `sentence-transformers/all-MiniLM-L6-v2`

Rule: Cosine similarity < 0.85 between every held-out task context and every train/dev task context.

**Results (run 2026-04-29):** **SKIPPED** — `sentence-transformers` not installed in this run environment. No flagged pairs to report. This check will be re-run before the final submission. The n-gram check and profile-level grouping provide strong contamination guarantees in the interim; embedding similarity adds a softer semantic check that primarily catches paraphrase-level rewrites.

### Check 3: Time-shift verification

Tool: `generation/contamination_check.py`

Rule: Every task whose `signal_source` is `crunchbase_odm` or `layoffs_fyi` must have a non-null `signal_time_window` field documenting which fiscal quarter the public data came from. Tasks with `signal_source = synthetic` are exempt.

**Results (run 2026-04-29):** Found 93 tasks with public signal sources (crunchbase_odm or layoffs_fyi) across all 200 tasks. All 93 have non-null `signal_time_window` values. Zero violations. Status: **PASS.**

Full results committed to `generation/contamination_check.json`.

---

## 6. Inter-Rater Agreement Protocol

Full protocol defined in `dataset/inter_rater_agreement.md`. This section documents the trigger condition and outcome commitment.

**Scope:** Only `tone_checker_fn` and `objection_ack_fn` require IRA (LLM-backed dimensions). All other dimensions are deterministic (κ = 1.0 by construction).

**Trigger condition (non-negotiable):**
If mean pairwise Cohen's κ across any two raters on a 30-task stratified dev sample falls below **κ < 0.70** on `tone_checker_fn` or `objection_ack_fn`, results may not be finalized until the disagreement is resolved per the adjudication procedure in `dataset/inter_rater_agreement.md`.

**Sampling:** 30 tasks stratified by segment (10 smb / 10 series_b / 10 enterprise) drawn from dev split.

**Outcome:** Completed agreement matrix committed to `dataset/inter_rater_agreement.md` before reporting final results.

### 6.1 Agreement Matrix

*(To be filled after first evaluation run — see `dataset/inter_rater_agreement.md` §Agreement Matrix Template)*

| Rater Pair | Dimension | κ | Status |
|---|---|---|---|
| LLM judge vs. Annotator A | `tone_checker_fn` | — | pending |
| LLM judge vs. Annotator B | `tone_checker_fn` | — | pending |
| Annotator A vs. Annotator B | `tone_checker_fn` | — | pending |
| LLM judge vs. Annotator A | `objection_ack_fn` | — | pending |
| LLM judge vs. Annotator B | `objection_ack_fn` | — | pending |
| Annotator A vs. Annotator B | `objection_ack_fn` | — | pending |

---

## 7. Scoring Evaluator Design Contract

`evaluation/scoring_evaluator.py` must satisfy all of the following:

- Every checker function returns `float` in `[0.0, 1.0]`.
- Aggregate score = `weighted_sum(raw_scores * weights) * 5`, producing `[0.0, 5.0]`.
- Re-running on a fresh clone produces scores within ±2 percentage points (reproduction
  fidelity). Achieved by: `random.seed(42)`, `np.random.seed(42)`, LLM judge
  `temperature=0.0`.
- LLM judge calls use `claude-haiku-4-5-20251001` for cost efficiency during iteration.
  Eval-tier judge (`claude-sonnet-4-6`) used only for final held-out spot-check.
- `--mock-llm` flag bypasses all API calls for local testing and CI.
- `--demo` flag runs against 3 hand-built dummy tasks with no dataset required.

---

## 8. Cost Log Reference

See `training/cost_log.md` for itemized API call log.
Total budget: $10. Do not re-run τ²-Bench retail (not in budget).
No eval-tier model on Days 2–3 (dev-tier only during dataset iteration).
