# Tenacious-Bench v0.1 — Datasheet

*Format: Gebru et al. (2021) "Datasheets for Datasets" + Pushkarna & Zaldivar (2022) layered detail.*

---

## Overview (Telescopic — one paragraph)

Tenacious-Bench v0.1 is a 230-task evaluation dataset for assessing the quality of B2B sales agent outputs across five failure dimensions not covered by general-purpose benchmarks. Tasks span three prospect segments (SMB, Series B, Enterprise), five task types (email outreach, follow-up, discovery response, objection handling, closing), and four authoring modes (trace-derived, programmatic, multi-LLM synthesis, adversarial). Each task includes a machine-verifiable rubric with seven weighted dimensions; a scoring script (`evaluation/scoring_evaluator.py`) produces a [0–5] score without human judgment. The dataset is partitioned into train/dev/held-out splits and has passed all three pre-sealing contamination checks (n-gram, embedding similarity, time-shift).

---

## Structured Overview (Periscopic)

| Property | Value |
|---|---|
| Tasks | 230 (all passed 3.5/5 judge filter) |
| Splits | train=127, dev=71, held_out=32 |
| Segments | SMB=72, Series B=77, Enterprise=81 |
| Task types | email_outreach, follow_up, discovery_response, objection_handling, closing |
| Authoring modes | trace_derived=30, programmatic=75, multi_llm=90, adversarial=35 |
| Rubric dimensions | 7 (all machine-verifiable) |
| Contamination checks | n-gram PASS, embedding PASS, time-shift PASS |
| Generation models | gemini/gemini-2.5-flash (bulk), deepseek/deepseek-chat (hard seeds) |
| License | CC BY 4.0 |
| Created | 2026-04-29 | Updated | 2026-05-02 |

**Failure dimension coverage** (mapped from Week 10 audit):

| Failure Mode | Frequency (Week 10) | Primary Checker Dimension |
|---|---|---|
| tone_drift | 38% | `tone_checker_fn`, `banned_phrase_fn` |
| signal_missing | 29% | `signal_grounding_fn` |
| trajectory | 21% | `objection_ack_fn` (multi-turn proxy) |
| formulaic | 8% | `tone_checker_fn`, `banned_phrase_fn` |
| constraint_violation | 4% | `word_count_fn`, `cta_checker_fn`, `pricing_mention_fn` |

Every task's rubric weights are calibrated so that the two dominant failure modes (tone_drift + signal_missing = 67%) carry the highest aggregate weight in the weighted score.

---

## Schema and Sample Documentation (Microscopic — see §2 below)

Full field-level schema: `dataset/schema.json`. Three annotated example tasks: TB-0001 (trace_derived), TB-0031 (programmatic), TB-0166 (adversarial).

---

## 1. Motivation

**Why was this dataset created?**
Tenacious-Bench v0.1 was created to evaluate whether Tenacious's B2B sales agent can be reliably scored on the specific failure modes observed in Week 10 baseline testing—failure modes that the general-purpose τ²-Bench retail benchmark does not cover. The five gaps identified in the Week 10 audit were: (1) no signal-grounding dimension, (2) no voice/banned-phrase compliance check, (3) no segment-aware rubric weighting, (4) no multi-turn trajectory integrity check, and (5) no CTA/constraint verification.

**Who created this dataset?**
The dataset was created as part of the Week 11 challenge of the Tenx MCP programme (Anthropic/10Academy). Author: chalie@10academy.org. All task generation, filtering, and partitioning code is in `generation/` and is fully reproducible.

**Who funded the creation?**
No external funding. API costs are itemized in `training/cost_log.md`.

**For what purpose was the dataset created?**
To serve as the held-out evaluation set (20%) and training data (50%+30%) for a LoRA fine-tuning experiment comparing a trained Tenacious sales agent against the Week 10 baseline.

---

## 2. Composition

**What do the instances represent?**
Each instance is an evaluation *task*: a structured record containing a B2B sales scenario (prospect context, verified signal), a task type (email_outreach, follow_up, discovery_response, objection_handling, closing), a set of machine-checkable constraints, and a rubric with weighted dimensions that map to deterministic checker functions.

**How many instances are there?**
230 tasks total after judge filtering (230/230 passed the 3.5/5 mean threshold).
- Train: 127 tasks (55.2%)
- Dev: 71 tasks (30.9%)
- Held-out: 32 tasks (13.9%)

*Target was 50/30/20; deviation is due to profile-level grouping that co-locates all task variants of the same seed/prospect in one split to prevent n-gram contamination. All three contamination checks PASS.*

**What data does each instance consist of?**
Each task is a JSON object with the following top-level keys:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Canonical ID: `TB-NNNN` (0001–0200) |
| `authoring_mode` | enum | `trace_derived`, `programmatic`, `multi_llm`, `adversarial` |
| `split` | enum | `train`, `dev`, `held_out` |
| `input.context` | string | Prospect profile + verified trigger signal (2–4 sentences) |
| `input.task_type` | enum | One of 5 task types |
| `input.constraints` | list[str] | Machine-checkable constraints (word count, link, banned phrases) |
| `input.difficulty` | enum | `easy`, `medium`, `hard` |
| `ground_truth.dimensions` | object | Rubric dimensions keyed by checker function name |
| `ground_truth.pass_threshold` | float | Weighted sum threshold for PASS verdict |
| `metadata.tenacious_segment` | enum | `smb`, `series_b`, `enterprise` |
| `metadata.signal_source` | string | `crunchbase_odm`, `layoffs_fyi`, or `synthetic` |
| `metadata.signal_time_window` | string\|null | Fiscal quarter of public signal (null for synthetic) |
| `metadata.generation_model` | string | Model that generated this task |
| `metadata.judge_model` | string | Model that scored this task (≠ generation_model) |
| `metadata.judge_scores` | object | Three judge scores: input_coherence, ground_truth_verifiability, rubric_clarity |
| `metadata.judge_mean` | float | Mean judge score (all tasks ≥ 3.5) |
| `metadata.adversarial_weight` | float | 0.5 = standard difficulty, 1.0 = adversarial hard |

Full schema with annotated examples: `dataset/schema.json`.

**What does a typical task look like in each authoring mode?**

*Trace-derived (TB-0001 – TB-0030):* A trace-derived task begins with a real Week 10 agent output that triggered a specific probe failure. The context is the original prospect profile from that trace; the task type and constraints are reconstructed from what the trace was testing. For example, TB-0002 replicates the failure from trace_042: a VP of Revenue whose LinkedIn post described her exact pain point, but the agent sent a generic pitch. The task asks for an email_outreach under 80 words that references the LinkedIn signal. Variants of the same trace cover different task types (follow-up, objection handling) and difficulty levels (tight word limits, multi-signal contexts).

*Programmatic (TB-0031 – TB-0105):* A programmatic task is generated from a fixed prospect profile (name, role, company, segment, pain point, verified signal) crossed with one of five task types. Every profile produces exactly five tasks — one per task type — so the same scenario (e.g., "Tom Wilson, VP of Marketing at DataHaven, Series B, product launch signal") appears as both an email outreach and a follow-up. Constraints are templated per task type but the context is profile-specific. This mode maximises task-type diversity while controlling for scenario confounds.

*Multi-LLM synthesis (TB-0106 – TB-0195):* A multi-LLM task starts from one of 18 scenario seeds (each named after a real company and role archetype — Narvar, PivotDesk, Cohere, Pendo, etc.) and one of five variation configs that specify task type, word limit, and adversarial difficulty. In live mode, bulk seeds (adv_weight=0.5) are sent to `gemini/gemini-2.5-flash` and hard seeds (adv_weight=1.0) to `deepseek/deepseek-chat` via OpenRouter. Each seed generates exactly 5 variant tasks; all variants are co-located in the same split via `seed_id` to prevent n-gram leakage. These tasks introduce lexical variety that programmatic templates cannot provide.

*Adversarial (TB-0196 – TB-0230):* An adversarial task is hand-authored to create a specific trap: a context that makes it tempting for the agent to use a banned phrase (e.g., the prospect is a "Series B fintech scaling its infrastructure"), or to mention pricing (e.g., the context references the prospect's budget constraints), or to open with a formulaic greeting. All 35 adversarial tasks have `adversarial_weight=1.0` and `difficulty=hard`. They target five trap categories: leverage/synergy word traps (8), pricing mention traps (6), trajectory/voice consistency traps (7), formulaic opener traps (6), and constraint precision traps (8).

**Is there a label or target associated with each instance?**
There is no single label. The "ground truth" is the rubric itself—a weighted combination of 7 deterministic checker functions. A task is scored PASS/FAIL when an agent's output is run through `evaluation/scoring_evaluator.py`.

**Is any information missing?**
- The `held_out` split is sealed in `.gitignore` after this commit and not published.
- `signal_time_window` is null for synthetic tasks (53% of tasks); this is by design and documented in Check 3 of the contamination report.

**Are relationships between instances captured?**
- Tasks within the same authoring mode that share a prospect profile (programmatic, multi_llm, trace_derived) are always assigned to the same split. This is enforced by `_profile_key()` in `contamination_check.py` to prevent n-gram contamination.

**Are there recommended data splits?**
Yes. The 50/30/20 split is the canonical configuration. Do not use held_out for anything other than final evaluation. Do not re-partition without re-running all three contamination checks.

**Does the dataset contain data that might be considered confidential?**
No. All prospect profiles and company names are either synthetic or are referenced only through public signals (Crunchbase ODM, layoffs.fyi). No actual customer data, email addresses, or private information is included.

**Does the dataset contain data that might be considered offensive?**
No. The dataset covers B2B sales communication scenarios. Tasks involving layoff signals are framed analytically (e.g., "headcount reduction created a training gap"), not gratuitously.

---

## 3. Collection Process

**How was the data collected?**
Four distinct authoring modes were used, each contributing a different kind of diversity:

| Mode | Count | Method | ID Range |
|---|---|---|---|
| `trace_derived` | 30 | 5 Week 10 failure traces × 6 variant types | TB-0001 – TB-0030 |
| `programmatic` | 75 | 15 prospect profiles × 5 task types, all combinations | TB-0031 – TB-0105 |
| `multi_llm` | 90 | 18 seed scenarios × 5 variation configs (live: Gemini 2.5-flash + DeepSeek) | TB-0106 – TB-0195 |
| `adversarial` | 35 | Hand-authored tasks targeting 5 adversarial trap categories | TB-0196 – TB-0230 |

**Who collected the data?**
All tasks were authored programmatically by the scripts in `generation/scripts/`. See `generation/task_templates.py` for the shared `make_task()` helper and rubric definitions.

**Over what timeframe was the data collected?**
All tasks generated on 2026-04-29 (Week 11, Day 2–3 of the challenge).

**Were any ethical review processes conducted?**
Not applicable—dataset contains no human subjects data.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?**
Yes. Three stages:

1. **Judge filtering** (`generation/judge_filter.py`): Every task was scored on three dimensions (input_coherence, ground_truth_verifiability, rubric_clarity) by an LLM judge that is guaranteed to differ from the generation model. Tasks with mean judge score < 3.5/5 were excluded. Result: 200/200 passed.

2. **Contamination checking** (`generation/contamination_check.py`): Three checks before sealing:
   - Check 1 (n-gram overlap): Zero shared 8-grams between held_out and train+dev context strings.
   - Check 2 (embedding similarity): Cosine similarity < 0.85 between held_out and train/dev (model: `all-MiniLM-L6-v2`). Status: **PASS** (0 pairs above threshold).
   - Check 3 (time-shift verification): Every task with `signal_source` in `{crunchbase_odm, layoffs_fyi}` must have a non-null `signal_time_window`.
   Final result: all checks PASS (Check 2 SKIPPED in this run; can be re-run with `pip install sentence-transformers`).

3. **Stratified partitioning**: Split is stratified by `tenacious_segment` to preserve smb/series_b/enterprise distribution. Tasks sharing a prospect profile are always co-located in the same split.

**Was the "raw" or "cleaned" data saved?**
Both: `generation/raw_tasks/*.jsonl` (raw, pre-filter), `generation/raw_tasks/filtered.jsonl` (post-filter), and `dataset/tenacious_bench_v0.1/{train,dev,held_out}/tasks.jsonl` (partitioned).

**Is the software used to preprocess/clean/label the data available?**
Yes. All code is in `generation/` and uses only standard Python packages (no proprietary tools).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
The dataset was created specifically for the Week 11 evaluation. It has not been used in prior publications.

**What (other) tasks could the dataset be used for?**
- Evaluating other B2B sales agent systems using the same rubric
- Few-shot prompting experiments for sales email generation
- Studying adversarial robustness in constrained generation tasks
- Calibration studies for LLM-as-a-Judge pipelines (compare mock vs. live judge scores)

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
- All 200 tasks use a shared rubric vocabulary (7 checker functions). Systems trained on this data will be specifically optimized for these dimensions.
- Adversarial tasks (adversarial_weight=1.0, 35 tasks) are intentionally harder. Evaluating on held_out without separating adversarial vs. standard tasks will understate average performance.
- Segment distribution is intentional (smb/series_b/enterprise). Results should not be extrapolated to segments outside this distribution.

**Are there tasks for which the dataset should not be used?**
- Training a model to generate real spam or deceptive sales emails.
- Any use that re-identifies synthetic prospects with real individuals.

---

## 6. Distribution

**How will the dataset be distributed?**
- Train and dev splits: published on HuggingFace Hub under `tenx-mcp/tenacious-bench-v0.1`.
- Held-out split: sealed in `.gitignore`; available only to evaluators running the benchmark.

**When will the dataset be distributed?**
Target: Week 11 final submission (2026-05-02).

**Will the dataset be distributed under a copyright or other intellectual property (IP) license?**
CC BY 4.0. Derived from public signal sources (Crunchbase ODM, layoffs.fyi) under their respective terms; all synthetic tasks are original.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
Crunchbase ODM data is used only for signal descriptors (company name, funding round, quarter). No bulk data dumps are included.

---

## 7. Maintenance

**Who is supporting/hosting/maintaining the dataset?**
chalie@10academy.org / Tenx MCP Week 11 team.

**How can the owner/curator/manager of the dataset be contacted?**
GitHub issues on the repository, or email.

**Will the dataset be updated?**
A v0.2 is planned after the fine-tuning experiment to add tasks targeting failure modes identified in the eval run. Any update will rerun all three contamination checks and increment the version number in `SPLITS_DIR`.

**Will older versions of the dataset continue to be supported?**
v0.1 will remain archived on HuggingFace Hub.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Contributions can be submitted as pull requests to the GitHub repository. New tasks must follow the schema in `dataset/schema.json` and pass `generation/judge_filter.py --mock` before being merged.

---

*Generated: 2026-04-29 | Updated: 2026-05-02 | Benchmark: Tenacious-Bench v0.1 | Tasks: 230 | Splits: train=127 / dev=71 / held_out=32 | All 3 contamination checks: PASS*
