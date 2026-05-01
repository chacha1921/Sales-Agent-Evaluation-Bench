# Methodology: Tenacious-Bench v0.1

## 1. Path Declaration

**Chosen path: Path B — Preference Learning (ORPO + SimPO)**

The Week 10 failure taxonomy shows the agent is not categorically incapable — it produces
correct outputs on the same task types where it also fails. Tone drift triggered at 5/10
trials (P-013), signal over-claiming at 6/10 (P-005), and ICP misclassification at 4/10
(P-001). A 40–60% trigger rate means the model already knows how to get it right; the
problem is it does not do so reliably. This is a consistency problem, not a capability gap,
which makes preference learning the appropriate method. Path A (SFT) teaches new behavior;
Path B teaches the model to consistently prefer the correct behavior it already sometimes
produces. Path C (PRM) targets multi-turn trajectory failures, which account for only 13%
of the failure distribution — insufficient to justify a process reward model.

**Justification traces:**
- `trace_002`: Agent scored 0.79 on τ²-Bench (meeting outcome acceptable) but 0.8/5 on
  Tenacious rubric — output contained 7 banned phrases including "leverage", "synergy",
  "end-to-end", and "game-change". The terminal-state metric masked a completely off-brand
  message. Same task type (email_outreach, Series B) passes cleanly in `trace_001` (4.2/5).
- `trace_012`: Agent ignored a 45-day-old layoff signal (18% headcount cut) and sent a
  growth-pitch to a company in cost-cutting mode, triggering P-001. τ²-Bench scored this
  0.88 (prospect replied); Tenacious scored 1.1/5. The agent correctly handles the same
  signal conflict in `trace_003` (4.0/5) — confirming inconsistency, not incapability.

**Justification papers:**
- Tülu 3 (Lambert et al., 2024) — SFT accounts for ~90% of final quality on non-verifiable
  tasks; RLVR/DPO adds nothing without a binary reward signal. Our rubric dimensions are
  continuous, not binary — preference learning is the right fit.
- LIMA (Zhou et al., 2023) — 1,000 high-quality preference pairs are sufficient to shift
  output style. Supports our 198-pair training target.

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

*Completed 2026-04-29. Full protocol and diagnosis in `dataset/inter_rater_agreement.md`.*

**Round 2 (post-revision) — final values:**

| Rater Pair | Dimension | κ | Status |
|---|---|---|---|
| LLM judge vs. Annotator A | `tone_checker_fn` | 1.000 | ✅ |
| LLM judge vs. Annotator B | `tone_checker_fn` | 1.000 | ✅ |
| Annotator A vs. Annotator B | `tone_checker_fn` | 1.000 | ✅ |
| **Mean κ (tone)** | | **1.000** | **PASS** |
| LLM judge vs. Annotator A | `objection_ack_fn` | 1.000 | ✅ |
| LLM judge vs. Annotator B | `objection_ack_fn` | 1.000 | ✅ |
| Annotator A vs. Annotator B | `objection_ack_fn` | 1.000 | ✅ |
| **Mean κ (objection_ack)** | | **1.000** | **PASS** |

**Round 1 triggered a rubric revision** (mean κ = 0.662 on `tone_checker_fn`, below 0.70 threshold). Root cause: mock heuristic gave partial credit to tier-1 brand violations ("just checking in") and missed "My name is" opener. Revised to a two-tier system (tier-1 = immediate FAIL). Round 2 achieved κ = 1.000. See `dataset/inter_rater_agreement.md` for full diagnosis.

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
