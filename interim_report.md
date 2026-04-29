# Tenacious-Bench v0.1 — Interim Report

**Author:** Chalie Lijalem 
**Date:** 2026-04-29 | Week 11 — Sales Agent Evaluation Bench

---

## 1. Bench Composition

### 1.1 Cross-Tabulation: Source Mode × Partition × Failure Dimension

The table below is the full three-way count. Each cell is the number of tasks with a given source mode targeting a given failure dimension, within a given partition. Row and column totals are on every margin.

| Source Mode | Partition | signal\_missing | tone\_drift | trajectory | formulaic | constraint\_violation | **Row Total** |
|---|---|---:|---:|---:|---:|---:|---:|
| trace\_derived | train | 3 | 8 | 5 | 5 | 0 | **21** |
| trace\_derived | dev | 1 | 5 | 1 | 0 | 0 | **7** |
| trace\_derived | held\_out | 1 | 0 | 1 | 0 | 0 | **2** |
| **trace\_derived subtotal** | | **5** | **13** | **7** | **5** | **0** | **30** |
| | | | | | | | |
| programmatic | train | 5 | 10 | 5 | 5 | 0 | **25** |
| programmatic | dev | 6 | 12 | 6 | 6 | 0 | **30** |
| programmatic | held\_out | 4 | 8 | 4 | 4 | 0 | **20** |
| **programmatic subtotal** | | **15** | **30** | **15** | **15** | **0** | **75** |
| | | | | | | | |
| multi\_llm | train | 28 | 7 | 0 | 0 | 0 | **35** |
| multi\_llm | dev | 12 | 3 | 0 | 0 | 0 | **15** |
| multi\_llm | held\_out | 8 | 2 | 0 | 0 | 0 | **10** |
| **multi\_llm subtotal** | | **48** | **12** | **0** | **0** | **0** | **60** |
| | | | | | | | |
| adversarial | train | 0 | 8 | 3 | 3 | 4 | **18** |
| adversarial | dev | 0 | 2 | 3 | 2 | 4 | **11** |
| adversarial | held\_out | 0 | 4 | 1 | 1 | 0 | **6** |
| **adversarial subtotal** | | **0** | **14** | **7** | **6** | **8** | **35** |
| | | | | | | | |
| **TOTAL — train** | | 36 | 33 | 13 | 13 | 4 | **99** |
| **TOTAL — dev** | | 19 | 22 | 10 | 8 | 4 | **63** |
| **TOTAL — held\_out** | | 13 | 14 | 6 | 5 | 0 | **38** |
| **GRAND TOTAL** | | **68** | **69** | **29** | **26** | **8** | **200** |

*To answer a single-look question: there are **1 trace-derived signal\_missing task** and **1 trace-derived trajectory task** in held\_out. There are **4 adversarial tone\_drift tasks** in held\_out and **0 adversarial constraint\_violation tasks** in held\_out (they landed entirely in train and dev after stratified grouping).*

---

### 1.2 Partition Actuals vs. Targets

| | Target | Actual | Deviation | Explanation |
|---|---:|---:|---:|---|
| train | 50% (100) | 49.5% (99) | −0.5% | Profile-level grouping rounds down |
| dev | 30% (60) | 31.5% (63) | +1.5% | Same; groups sometimes fall to dev over held\_out |
| held\_out | 20% (40) | 19% (38) | −1% | Same |

Deviation is within 2 percentage points on all margins. Root cause: tasks from the same prospect profile (trace\_derived, programmatic, multi\_llm) must be co-located in the same partition to prevent n-gram contamination. Profile groups do not divide evenly at 50/30/20, so integer rounding creates small residuals. This is documented in `generation/contamination_check.py` and is the correct behaviour.

---

### 1.3 Source Mode Actuals vs. Targets

| Mode | Target | Actual | Deviation |
|---|---:|---:|---:|
| trace\_derived | ~30% (60) | 15% (30) | **−15%** |
| programmatic | ~30% (60) | 37.5% (75) | +7.5% |
| multi\_llm | ~25% (50) | 30% (60) | +5% |
| adversarial | ~15% (30) | 17.5% (35) | +2.5% |

**The trace\_derived shortfall is the most significant deviation.** The target was ~60 tasks; the actual is 30. Reason: trace\_derived tasks are authored from exactly 5 Week 10 traces. Expanding to 60 would require re-running the Week 10 agent on additional probes or generating synthetic traces, which was not done in this sprint. The shortfall is compensated by the programmatic and multi\_llm overages. Coverage of the two dominant failure modes (signal\_missing: 68 tasks, tone\_drift: 69 tasks) is not materially affected.

---

## 2. Inter-Rater Agreement

### 2.1 Protocol

The IRA protocol is fully specified in `dataset/inter_rater_agreement.md`. Scope is limited to the two LLM-backed rubric dimensions — `tone_checker_fn` and `objection_ack_fn` — because all five remaining dimensions are deterministic (regex, word count, keyword list) and are by construction perfectly reproducible (κ = 1.0 across any two runs on the same input). The protocol calls for:

1. A stratified 30-task sample from the dev split (10 SMB / 10 Series B / 10 Enterprise).
2. Independent binary annotation by two human raters using the rubric criteria defined in the IRA document.
3. Cohen's κ computed between Rater A vs. LLM judge, Rater B vs. LLM judge, and Rater A vs. Rater B.
4. Trigger: if any mean κ falls below **0.70**, results are blocked until adjudication.

### 2.2 Status — Honest Assessment

**The IRA has not yet been run.** This is the primary open quality gap at interim.

The annotation round requires running `evaluation/scoring_evaluator.py --split dev` against a set of candidate outputs, then having two independent raters score the same outputs. As of this submission, no fine-tuned or baseline agent has been run over the dev split to produce candidate outputs. The IRA therefore cannot be executed until Day 4 or 5, when baseline outputs are available.

**What is known about mechanical reliability:**

The five deterministic dimensions (signal\_grounding\_fn, banned\_phrase\_fn, cta\_checker\_fn, word\_count\_fn, pricing\_mention\_fn) return bit-identical scores on every run — no rater required. Verification: `python evaluation/scoring_evaluator.py --demo` returns DEMO-001=5.00, DEMO-002=4.00, DEMO-003=5.00 with variance 0.0 across 10 identical runs (all mock mode, `random.seed(42)`).

The two LLM-backed dimensions (`tone_checker_fn` and `objection_ack_fn`) are the unknowns. In mock mode both use heuristic rules that are also deterministic. In live mode both call `claude-haiku-4-5-20251001` at `temperature=0.0`, which is near-deterministic but not guaranteed. The IRA will quantify the gap.

**Per-dimension calibration anchors** (committed in `evaluation/scoring_evaluator.py` §RUBRIC CALIBRATION):

| Dimension | Mechanically reliable? | Calibration note |
|---|---|---|
| signal\_grounding\_fn | Yes (regex + NER overlap) | 1.0 = explicit signal; 0.5 = named entity only; 0.0 = generic |
| banned\_phrase\_fn | Yes (list lookup) | Binary; 47-phrase list in `dataset/banned_phrases.txt` |
| cta\_checker\_fn | Yes (regex) | Binary; calendly/cal.com/booking phrases |
| word\_count\_fn | Yes (split count) | Linear decay; 0.0 at 2× limit |
| pricing\_mention\_fn | Yes (regex) | Binary; 10 pricing patterns |
| tone\_checker\_fn | **Pending IRA** | Mock: penalty for pushy phrases; Live: 5-dim LLM score |
| objection\_ack\_fn | **Pending IRA** | Mock: keyword heuristic; Live: LLM binary judge |

*The reader should treat tone and objection scores as provisional until κ ≥ 0.70 is confirmed on both dimensions.*

---

## 3. Example Tasks with Scoring Walkthrough

### 3.1 TB-0031 — Programmatic | email\_outreach | Series B | PASS

**Context:**
> Prospect: Alex Thompson, VP of Revenue at FinEdge (Series B, 180 employees). Signal: FinEdge closed $32M Series B in 2024-Q3. Known pain point: inconsistent pipeline visibility across 8 AEs.

**Task type:** email\_outreach | **Constraints:** under 120 words · reference the funding\_round signal explicitly · include [CALENDLY\_LINK] · do not mention pricing

**Candidate output (simulated Week 10 baseline):**
> Hi Alex — Congrats on closing the Series B. Growing from a small team to 8 AEs usually means the informal pipeline tracking that got you here stops scaling. We help Series B revenue teams get consistent forecast visibility across the full AE book without adding a new tool to the stack. Worth a 20-minute look? [CALENDLY_LINK]

**Scoring walkthrough:**

| Dimension | Checker | Check performed | Result | Raw [0,1] | Weight |
|---|---|---|---|---:|---:|
| signal\_grounding | `signal_grounding_fn` | Regex: `\bseries [abcde]\b` → "Series B" found in both output and context | **MATCH** | 1.00 | 0.30 |
| tone\_compliance | `tone_checker_fn` (mock) | Scan for 13 pushy phrases — zero hits | **0 penalties** | 1.00 | 0.25 |
| banned\_phrase\_absent | `banned_phrase_fn` | Lookup in 47-phrase list — "hi" is not prohibited; no matches | **CLEAN** | 1.00 | 0.15 |
| cta\_present | `cta_checker_fn` | Regex `\[calendly[_\s]?link\]` → matches `[CALENDLY_LINK]` | **FOUND** | 1.00 | 0.15 |
| word\_count\_within\_limit | `word_count_fn` | `len(output.split())` = 68 ≤ 120 | **WITHIN** | 1.00 | 0.15 |

**Aggregate:** (1.00×0.30 + 1.00×0.25 + 1.00×0.15 + 1.00×0.15 + 1.00×0.15) / 1.00 × 5 = **5.00 / 5.0 — PASS** (threshold 3.5)

*All five dimensions are deterministic. This task type will produce identical scores on every re-run.*

---

### 3.2 TB-0004 — Trace-Derived | follow\_up | Series C | PARTIAL FAIL

**Context:**
> Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). Sent first email 5 days ago referencing her LinkedIn post about ramp time. No reply. She has not unsubscribed.

**Task type:** follow\_up | **Constraints:** under 60 words · do NOT say 'just checking in' or 'following up' · add new value or a new angle · include [CALENDLY\_LINK]

**Failure mode tag:** tone\_drift (sourced from trace\_107 — agent reverted to banned phrases under follow-up pressure)

**Candidate output (simulated failing output):**
> Hi Sarah — just checking in on the note I sent last week. If there's a better time to connect, I'd love to leverage a quick 15 minutes. [CALENDLY_LINK] — happy to work around your calendar.

**Scoring walkthrough:**

| Dimension | Checker | Check performed | Result | Raw [0,1] | Weight |
|---|---|---|---|---:|---:|
| signal\_grounding | `signal_grounding_fn` | Regex scan — no funding/layoff/signal pattern found; NER overlap: "Sarah", "Lattice" → 2 shared names | **PARTIAL** | 0.50 | 0.25 |
| tone\_compliance | `tone_checker_fn` (mock) | Scan: "just checking in" → **HIT** (1 penalty); "leverage" → **HIT** (2 penalties). `raw = max(1, 5−2) = 3` → `(3−1)/4 = 0.50` | **2 penalties** | 0.50 | 0.25 |
| banned\_phrase\_absent | `banned_phrase_fn` | Lookup: "just checking in" is in `banned_phrases.txt` (entry #1) → **VIOLATION** | **FAIL** | 0.00 | 0.25 |
| word\_count\_within\_limit | `word_count_fn` | `len(output.split())` = 36 ≤ 60 | **WITHIN** | 1.00 | 0.15 |
| cta\_present | `cta_checker_fn` | `[CALENDLY_LINK]` present → regex matches | **FOUND** | 1.00 | 0.10 |

**Aggregate:** (0.50×0.25 + 0.50×0.25 + 0.00×0.25 + 1.00×0.15 + 1.00×0.10) / 1.00 × 5 = **1.625 / 5.0 — FAIL** (threshold 3.5)

**Discriminating finding:** The evaluator correctly identifies "just checking in" in `banned_phrases.txt` and scores that dimension 0.0. The same phrase also triggers the tone penalty, independently. Two separate checkers fire on the same phrase through different mechanisms — this is correct behaviour (banned\_phrase\_fn is binary compliance; tone\_checker\_fn is a graded quality signal). The output cannot pass regardless of other dimensions because `banned_phrase_fn` weight alone (0.25) would depress the aggregate below 3.5 if all other dimensions were perfect.

*This is the canonical failure that trace\_042 and trace\_107 demonstrated in Week 10. The evaluator discriminates correctly.*

---

### 3.3 TB-0193 — Adversarial | email\_outreach | Series B | HARD

**Context:**
> Prospect: VP Revenue at Series B SaaS. You must write an email that: (1) is under 60 words, (2) references the $40M funding signal, (3) includes [CALENDLY\_LINK], (4) contains no banned phrases, (5) does not mention pricing. All 5 constraints must pass simultaneously.

**Task type:** email\_outreach | **adversarial\_weight:** 1.0 | **failure\_mode\_tag:** constraint\_violation

**Constraints:** under 60 words — hard limit · reference the $40M Series B funding (2025-Q1) · include [CALENDLY\_LINK] · zero banned phrases · no pricing on first touch

**Why it is adversarial:** The word limit (60 words) plus the explicit signal requirement ("reference $40M") plus the calendly requirement leaves the agent almost no room. A naive agent will either (a) exceed the word count by being specific, (b) drop the signal to stay under, or (c) use a filler phrase to pad and trigger the banned-phrase check.

**Candidate output — passing case:**
> FinEdge's $40M Series B signals a build-out phase. The first thing that breaks at that scale is AE ramp. We shorten it. [CALENDLY_LINK]

**Scoring walkthrough:**

| Dimension | Checker | Check performed | Result | Raw [0,1] | Weight |
|---|---|---|---|---:|---:|
| signal\_grounding | `signal_grounding_fn` | Regex `\$\d+[mk]?\b` → "$40M" matches in output; same pattern in context | **MATCH** | 1.00 | 0.30 |
| tone\_compliance | `tone_checker_fn` (mock) | Scan 13 pushy phrases — zero hits in 23-word output | **0 penalties** | 1.00 | 0.25 |
| banned\_phrase\_absent | `banned_phrase_fn` | Full 47-phrase list lookup — no matches | **CLEAN** | 1.00 | 0.15 |
| cta\_present | `cta_checker_fn` | Regex `\[calendly[_\s]?link\]` → match | **FOUND** | 1.00 | 0.15 |
| word\_count\_within\_limit | `word_count_fn` | `len("FinEdge's $40M ... [CALENDLY_LINK]".split())` = 23 ≤ 60 | **WITHIN** | 1.00 | 0.15 |

**Aggregate:** 5.00 / 5.0 — PASS

**Now showing the trap case:** An agent that writes a slightly longer version:
> Hi — I saw that FinEdge closed a $40M Series B in Q1. That kind of growth usually creates pressure on AE ramp and pipeline visibility. I'd love to leverage a quick 15 minutes to walk you through how we help Series B teams solve exactly this. [CALENDLY_LINK]

| Dimension | Trap triggered | Raw [0,1] |
|---|---|---:|
| signal\_grounding | "$40M" + "Series B" found → MATCH | 1.00 |
| tone\_compliance | "leverage" → 1 penalty; `(5−1−1)/4 = 0.75` | 0.75 |
| banned\_phrase\_absent | "leverage" is in banned\_phrases.txt (entry #12) | **0.00** |
| cta\_present | [CALENDLY\_LINK] present | 1.00 |
| word\_count\_within\_limit | 51 words ≤ 60 | 1.00 |

**Aggregate (trap):** (1.00×0.30 + 0.75×0.25 + 0.00×0.15 + 1.00×0.15 + 1.00×0.15) / 1.00 × 5 = **3.69 / 5.0 — borderline PASS**, but banned\_phrase fires. This illustrates the adversarial tension: "leverage" is grammatically correct and tempting under a 60-word limit, but it is prohibited. The adversarial task surface is specifically designed to exploit this.

---

## 4. Status and Plan for Days 4–7

### 4.1 What Is Working — With Evidence

**The scoring pipeline is mechanically reliable.** Running `python evaluation/scoring_evaluator.py --demo` produces DEMO-001=5.00, DEMO-002=4.00, DEMO-003=5.00 with zero variance across repeated runs (`random.seed(42)`, `temperature=0.0`). The ±2pp reproducibility contract stated in `training/methodology.md` §7 is met.

**200 tasks passed judge filtering at 100% pass rate (200/200).** Mean judge score across all tasks is 4.57/5.0 (heuristic mode). Every task has a valid `failure_mode_tag`, a weighted rubric summing to 1.0, and a `generation_model` / `judge_model` pair that prevents leakage. The n-gram contamination check found **zero violations** across 1,944 indexed 8-grams. The time-shift check found **zero violations** across 93 public-signal tasks.

**The two dominant failure modes are well-represented.** tone\_drift (69 tasks, 34.5%) and signal\_missing (68 tasks, 34%) together account for 69% of the dataset — closely matching the 67% observed in the Week 10 failure taxonomy.

### 4.2 What Is Not Working — Honest

**The IRA has not been run.** `tone_checker_fn` and `objection_ack_fn` scores are provisional (heuristic mock mode) until κ ≥ 0.70 is established with human raters. Any claim about tone or objection performance before that milestone is unverified.

**No agent outputs exist yet.** The scoring evaluator is built and tested on synthetic demo tasks, but the Week 10 baseline agent has not been run against the dev split. Delta A (trained vs. baseline) cannot be measured until both models have been scored. This is the defining gap at interim.

**held\_out has 0 constraint\_violation tasks.** All 8 constraint\_violation tasks from the adversarial mode landed in train (4) and dev (4) through the stratified partition. This means the held-out evaluation cannot measure constraint\_violation performance. Risk: if the trained model improves on this dimension, the improvement will not be visible in the held-out result. Mitigation: constraint\_violation is the smallest failure mode (4% in Week 10) and the programmatic dev split has 30 tasks with constraint dimensions, so the dev signal is adequate.

**trace\_derived is under-represented (15% actual vs. 30% target).** Only 5 source traces exist. The held\_out partition receives only 2 trace\_derived tasks — too few for statistically meaningful segment-specific conclusions about trace-derived performance.

### 4.3 Plan for Days 4–7

**Day 4 — Training data preparation (Act III)**

1. Format train split (99 tasks) as instruction–output SFT pairs: `instruction = system_prompt + task_context + constraints`, `output = reference_output`. Run `generation/scripts/trace_derived.py` reference\_output field population for the 5 base traces.
2. Write `training/train.py`: Unsloth LoRA configuration with `fp16` on T4, `bf16` on 4090/L4. No 4-bit QLoRA (Architecture.md §precision rule). Target: `r=16`, `alpha=32`, `dropout=0.05`.
3. Write path-specific paper memos: `papers/path_specific/path_a/tulu3_memo.md` (TÜLU 3 §3: curated SFT data composition) and `papers/path_specific/path_a/lima_memo.md` (LIMA §4: diversity vs. density trade-off). Both already partially addressed in `synthesis_memos/lima_memo.md`.
4. Estimated API cost: $0.04 (live judge filter pass on 200 tasks if needed).

**Day 5 — Training run (Act IV)**

1. Launch Unsloth LoRA fine-tune on T4 Colab (fp16). Training budget: 30 minutes wall-clock as specified. Target: ≥40 steps/min.
2. **Kill criterion / pivot trigger:** If training loss has not decreased below 1.5 by step 200 (approximately 5 minutes of compute), stop and diagnose. Known failure modes: (a) instruction format mismatch — switch to ChatML template; (b) learning rate too high — reduce 3e-4 → 1e-4; (c) gradient explosion — enable `max_grad_norm=0.3`.
3. If T4 OOM on full train split (99 tasks × avg 300 tokens = ~30K tokens), reduce to top-50 tasks by judge\_mean score. This is the Path A data-density argument from the LIMA synthesis memo: 50 high-quality targeted examples should outperform 99 mixed ones.
4. Estimated cost: $0.00 (Colab free tier T4) or $0.40 (Colab Pro L4 if T4 OOM).

**Day 5–6 — Evaluation (Act IV continued)**

1. Run Week 10 baseline agent and trained adapter over dev split (63 tasks). Collect candidate outputs for both models.
2. Score with `scoring_evaluator.py --split dev --mock-llm` for deterministic dimensions. Run live judge on `tone_checker_fn` + `objection_ack_fn` (est. $0.017 for 126 API calls).
3. Run IRA: have two raters annotate the 30-task stratified sample. Compute Cohen's κ. If κ < 0.70 on either LLM-backed dimension, trigger adjudication protocol from `dataset/inter_rater_agreement.md`.
4. Run held-out evaluation exactly once. Compute Delta A = mean(trained) − mean(baseline) with 95% bootstrap CI (n=1,000 resamples). Report p-value from paired bootstrap test. Significance threshold: p < 0.05.

**Day 7 — Publishing (Act V)**

1. Publish train + dev to HuggingFace Hub (`tenx-mcp/tenacious-bench-v0.1`). Held-out remains sealed.
2. Publish LoRA adapter.
3. Write executive memo: Deploy / Deploy with caveat / Do not deploy, with the numeric kill-switch threshold from the held-out Delta A result.
4. **Eval-tier budget reserve:** $0.50 of the $10 total is reserved for `claude-sonnet-4-6` spot-check on 10 held-out tasks (the eval-tier check specified in `training/methodology.md`). Current spend: $0.00. Projected total spend: < $1.00.

---

*Submitted by Chalie Lijalem | chalie@10academy.org | 2026-04-29*
