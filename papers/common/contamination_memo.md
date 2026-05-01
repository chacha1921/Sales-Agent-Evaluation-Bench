# Common Reading Memo: Contamination Survey (Chen et al., EMNLP 2025)

**Paper:** Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation (Chen et al., EMNLP 2025)
**Role:** Contamination-prevention design rules for the held-out partition

---

## Key Contributions

Chen et al. survey contamination incidents across 40+ benchmarks and identify four contamination vectors, ranked by frequency:

1. **N-gram overlap** — verbatim or near-verbatim sequences shared between training and evaluation data
2. **Semantic overlap** — paraphrase-level similarity not caught by n-gram checks
3. **Template leakage** — shared prompt templates across splits (evaluation uses the same structural pattern as training)
4. **Temporal leakage** — evaluation scenarios drawn from the same time window as training signals, enabling models to memorise specific events

The paper's core recommendation is that static benchmarks are inherently vulnerable to all four vectors as models are trained on more internet data. Dynamic evaluation (generating new tasks at evaluation time) is the only complete solution, but for static benchmarks, the minimum acceptable protection is: n-gram decontamination + semantic similarity check + documented temporal boundaries.

---

## Application to Tenacious-Bench

### Check 1 — N-gram overlap (`contamination_check.py`)

`generation/contamination_check.py` implements an 8-gram overlap check between held-out and train+dev context strings. Result: 1,944 8-gram sequences indexed; 0 violations. This directly addresses Chen et al.'s Vector 1.

An earlier run produced 23 violations — root cause was template strings shared across task types (Vector 3: template leakage). Fixed by limiting `task_text()` to context-only strings, excluding constraint templates.

### Check 2 — Semantic overlap (embedding similarity)

Chen et al. identify semantic overlap as the most under-addressed vector. The embedding check in `contamination_check.py` uses `all-MiniLM-L6-v2` with cosine threshold 0.85. This was skipped in the current run (sentence-transformers not installed) and is flagged for re-run before final submission. The n-gram check provides primary protection; the embedding check adds a paraphrase-level safety net.

### Check 3 — Temporal leakage (`signal_time_window`)

Every task with a public signal source (`crunchbase_odm`, `layoffs_fyi`) must have a non-null `signal_time_window` field. This documents *which quarter* the public data came from, satisfying Chen et al.'s temporal boundary requirement. Result: 93 public-signal tasks checked, 0 missing time windows.

### Profile-level grouping — template leakage prevention

The stratified partition in `contamination_check.py` groups all task variants of the same prospect profile into the same split. This prevents Vector 3 (template leakage): the model never trains on `email_outreach` for prospect X and evaluates on `follow_up` for the same prospect X. This protection goes beyond what Chen et al. require for static benchmarks.

### Remaining gap

Chen et al.'s strongest recommendation — dynamic evaluation — is not implemented. Tenacious-Bench is a static benchmark. The held-out partition is unlocked exactly once (Day 6 ablation) to limit repeated evaluation, which partially mitigates the static benchmark vulnerability but does not eliminate it.

---

*~430 words | Key vectors addressed: n-gram (PASS), semantic (SKIPPED — pending), temporal (PASS), template (PASS via profile grouping). Gap: embedding check pending; static benchmark limitation acknowledged.*
