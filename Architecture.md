# Architecture: Sales Agent Evaluation Bench (Week 11)

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Tenacious-Bench v0.1 System                         │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │  Data Inputs │───▶│ Dataset Pipeline │───▶│  Evaluation Engine   │  │
│  └──────────────┘    └──────────────────┘    └──────────────────────┘  │
│                               │                         │               │
│                               ▼                         ▼               │
│                    ┌──────────────────┐    ┌──────────────────────┐    │
│                    │ Training Pipeline│───▶│  Ablation & Results  │    │
│                    └──────────────────┘    └──────────────────────┘    │
│                                                         │               │
│                                                         ▼               │
│                                            ┌──────────────────────┐    │
│                                            │  Publishing Pipeline │    │
│                                            └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
Sales-Agent-Evaluation-Bench/
│
├── Architecture.md                  # This file
├── README.md                        # Quickstart, reproduction steps, artifact links
│
├── docs/
│   └── TRP1 Challenge Week 11_...  # Original challenge document
│
├── audit/
│   ├── audit_memo.md               # 600-word audit: what τ²-Bench misses
│   └── tau_bench_gap_analysis.md   # Specific probe IDs + trace evidence
│
├── dataset/
│   ├── schema.json                 # Task schema with 3 annotated examples
│   ├── tenacious_bench_v0.1/
│   │   ├── train/                  # 50% — used for SFT/DPO/PRM training
│   │   │   └── tasks.jsonl
│   │   ├── dev/                    # 30% — public, used during iteration
│   │   │   └── tasks.jsonl
│   │   └── held_out/               # 20% — sealed, gitignored after sealing
│   │       └── tasks.jsonl         # (added to .gitignore after Act II)
│   ├── datasheet.md                # Full Gebru + Pushkarna datasheet (3–5 pages)
│   └── inter_rater_agreement.md    # Agreement matrix, rubric revision log
│
├── generation/
│   ├── scripts/
│   │   ├── trace_derived.py        # Mode 1: extract tasks from Week 10 traces
│   │   ├── programmatic.py         # Mode 2: combinatorial template expansion
│   │   ├── multi_llm_synthesis.py  # Mode 3: frontier + cheap model authoring
│   │   └── adversarial.py          # Mode 4: hand-authored hard tasks
│   ├── judge_filter.py             # LLM-as-a-judge pointwise + pairwise filter
│   ├── contamination_check.py      # N-gram + embedding contamination checks
│   └── contamination_check.json    # Results of all contamination checks
│
├── training/
│   ├── methodology.md              # Path declaration, paper memos, trace citations
│   ├── methodology_rationale.md    # Justification with ≥3 trace IDs + 2 papers
│   ├── training_data/
│   │   ├── path_a_sft/            # Chat-template input/output pairs (1k–3k)
│   │   ├── path_b_dpo/            # Preference pairs (chosen / rejected)
│   │   └── path_c_prm/            # Step-level annotations on trajectories
│   ├── train.py                    # Main Unsloth LoRA training script
│   ├── training_run.log            # Full training log
│   └── cost_log.md                 # Itemized cost log (graded artifact)
│
├── evaluation/
│   ├── scoring_evaluator.py        # Machine-verifiable scorer (no human in loop)
│   ├── ablation_results.json       # Delta A / B / C + cost-Pareto table
│   ├── held_out_traces.jsonl       # Agent outputs on sealed held-out split
│   └── model_card.md               # HuggingFace model card (Path A or C)
│
├── papers/
│   ├── common/
│   │   ├── synthetic_data_best_practices.md
│   │   ├── datasheets_for_datasets.md
│   │   ├── contamination_survey.md
│   │   └── llm_as_judge_survey.md
│   └── path_specific/              # Filled based on chosen path
│       ├── path_a/                 # Tülu 3, LIMA, Magpie memos
│       ├── path_b/                 # DPO, SimPO, ORPO, Prometheus 2 memos
│       └── path_c/                 # PRM papers, DeepSeek-Math, Source2Synth memos
│
├── publishing/
│   ├── blog_post.md                # 1,200–2,000 word technical post
│   ├── memo.pdf                    # 2-page executive memo
│   ├── evidence_graph.json         # Every claim → source (task ID / log row / URL)
│   └── community_engagement.md     # Issue / workshop submission / PR evidence
│
└── week10_artifacts/               # Seeds from Week 10 (read-only reference)
    ├── trace_log.jsonl
    ├── probe_library.md
    └── failure_taxonomy.md
```

---

## 3. Component Architecture

### 3.1 Data Input Layer

```
week10_artifacts/           Public Data Sources
├── trace_log.jsonl  ──┐   ├── Crunchbase ODM (1,001 companies)
├── probe_library.md   │   ├── layoffs.fyi CSV
└── failure_taxonomy   │   └── Tenacious assets:
                       │       ├── style_guide_v2.md (12 good + 12 bad drafts)
                       └───────├── sales_deck.pdf
                               ├── case_studies/ (3 redacted)
                               ├── bench_summary_v2.md
                               ├── pricing_sheet.md
                               └── discovery_call_transcripts/ (5 synthetic)
```

All inputs are **read-only seeds**. Nothing from Week 10 artifacts is committed as ground truth without redaction and task transformation.

---

### 3.2 Dataset Pipeline

```
                    ┌──────────────────────────────────────────┐
                    │          Four Authoring Modes            │
                    │                                          │
  trace_log.jsonl ──▶  Mode 1: Trace-Derived     (~30%, ~75)  │
  templates.yaml  ──▶  Mode 2: Programmatic      (~30%, ~75)  │
  frontier + cheap ──▶ Mode 3: Multi-LLM Synth   (~25%, ~62)  │
  manual writing  ──▶  Mode 4: Adversarial        (~15%, ~38)  │
                    │                      Total: 200–300 tasks │
                    └──────────────┬───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │         LLM-as-a-Judge Filter            │
                    │                                          │
                    │  Pointwise scoring per task (1–5 each):  │
                    │  • input_coherence                       │
                    │  • ground_truth_verifiability            │
                    │  • rubric_clarity                        │
                    │                                          │
                    │  Pairwise for similar-looking tasks      │
                    │                                          │
                    │  Judge routing:                          │
                    │  • Dev-tier (Qwen3 / DeepSeek V3.2)     │
                    │    → high-volume filtering (all tasks)   │
                    │  • Eval-tier (Claude Sonnet 4.6)        │
                    │    → spot-check 50 tasks only            │
                    │                                          │
                    │  Leakage guard: generation model ≠       │
                    │  judge model (rotate model families)     │
                    └──────────────┬───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │       Contamination Prevention           │
                    │                                          │
                    │  Check 1: N-gram overlap                 │
                    │    < 8-gram overlap between partitions   │
                    │                                          │
                    │  Check 2: Embedding similarity           │
                    │    cosine < 0.85 (cheap embedding model) │
                    │                                          │
                    │  Check 3: Time-shift verification        │
                    │    public data from documentable window  │
                    └──────────────┬───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │          Dataset Partitioning            │
                    │                                          │
                    │  train/     50%  (100–150 tasks)         │
                    │  dev/       30%  (60–90 tasks)           │
                    │  held_out/  20%  (40–60 tasks) ◀── SEAL  │
                    └──────────────────────────────────────────┘
```

**Inter-Rater Agreement Protocol:**
- Hand-label 30 tasks cold
- Re-label same 30 tasks 24 hours later, without seeing first labels
- Compute agreement matrix per rubric dimension
- If any dimension < 80% → revise rubric → re-label → repeat

---

### 3.3 Task Schema

Each task in `tasks.jsonl` follows this structure:

```json
{
  "task_id": "TB-001",
  "split": "train | dev | held_out",
  "authoring_mode": "trace_derived | programmatic | multi_llm | adversarial",
  "source_trace_ids": ["trace_042", "trace_107"],
  "input": {
    "context": "Prospect: VP of Sales at Series B SaaS, 80-person team...",
    "task_type": "email_outreach | discovery_response | objection_handling | ...",
    "constraints": ["under 150 words", "no competitor names", "include calendar link"]
  },
  "candidate_output": "...",
  "ground_truth": {
    "type": "rubric",
    "dimensions": {
      "signal_grounding": {"weight": 0.25, "checker": "signal_grounding_fn"},
      "tone_compliance": {"weight": 0.25, "checker": "tone_checker_fn"},
      "banned_phrase_absent": {"weight": 0.20, "checker": "banned_phrase_fn"},
      "calendar_link_present": {"weight": 0.15, "checker": "regex_checker_fn"},
      "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn"}
    },
    "aggregate": "weighted_sum",
    "passing_threshold": 3.5
  },
  "metadata": {
    "tenacious_segment": "series_b | enterprise | smb",
    "failure_mode_tag": "tone_drift | formulaic | trajectory",
    "adversarial_weight": 1.0,
    "generation_model": "qwen3-next-80b",
    "judge_scores": {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_clarity": 4},
    "created_at": "2026-04-28"
  }
}
```

**Rubric design rule:** Every checker function must return a numeric score with zero human judgment required. LLM-judge sub-scores are allowed but must be deterministic and logged.

---

### 3.4 Evaluation Engine

```
scoring_evaluator.py
│
├── load_tasks(split)             # Load train / dev / held_out JSONL
├── run_agent(task_input)         # Call Week 10 agent, capture output
├── score_output(output, rubric)  # Apply all checker functions
│   ├── signal_grounding_fn()     # Check output grounds ≥1 public signal
│   ├── tone_checker_fn()         # LLM-judge tone call (deterministic seed)
│   ├── banned_phrase_fn()        # Regex against banned_phrases.txt
│   ├── regex_checker_fn()        # URL / calendar link pattern match
│   └── word_count_fn()           # Tokenizer-based count
├── aggregate_score()             # Weighted sum → [0, 5]
├── bootstrap_ci(scores, n=1000)  # 95% CI via paired bootstrap
└── save_results(ablation_results.json)
```

The evaluator must produce **stable scores within ±2 percentage points** on re-run with a fresh agent clone (reproduction fidelity requirement).

---

### 3.5 Training Pipeline

#### Path Selection Logic

```
Week 10 failure_taxonomy.md
        │
        ├── Tone drift, formulaic phrasing, voice inconsistency
        │   └──▶ PATH A: SFT (generation quality)
        │
        ├── Right output most of the time, can't detect when wrong
        │   └──▶ PATH B: DPO / SimPO / ORPO (judge/critic)
        │
        └── Locally good steps, globally bad trajectory
            └──▶ PATH C: PRM (process reward model)
```

Choose **one path** only. Justify in `methodology_rationale.md` with ≥3 Week 10 trace IDs and ≥2 paper citations.

#### Training Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Training Stack                       │
│                                                         │
│  Backbone: Qwen 3.5 (0.8B / 2B / 4B)                  │
│  Adapter:  LoRA only (PEFT) — never merge weights       │
│  Framework: Unsloth (preferred) or HuggingFace TRL     │
│  Runtime: Google Colab T4 (free) or RunPod 4090 ($0.34/hr)│
│  Budget cap: $5 for training                           │
│                                                         │
│  Path A — SFT Input Format:                            │
│  {"messages": [                                         │
│    {"role": "system", "content": "..."},               │
│    {"role": "user",   "content": task_input},          │
│    {"role": "assistant", "content": ideal_output}      │
│  ]}                                                     │
│                                                         │
│  Path B — DPO Input Format:                            │
│  {"prompt": task_input,                                 │
│   "chosen": high_score_output,                          │
│   "rejected": low_score_output}                         │
│                                                         │
│  Path C — PRM Input Format:                            │
│  {"trajectory": [step1, step2, ...stepN],               │
│   "step_scores": [0.8, 0.6, 0.2, ...],                 │
│   "outcome_score": 0.3}                                 │
└─────────────────────────────────────────────────────────┘
```

#### Training Run Protocol

```
Day 5 Morning:
  1. Launch single core LoRA run
  2. Monitor loss curve for first 30 minutes
  3. If not converging by 30 min → STOP, fix training data, retry
  4. Expected wall time: 30–90 min on T4/4090
  5. Save adapter checkpoint to HuggingFace Hub (LoRA only, not merged)
```

---

### 3.6 Ablation Framework

```
                         ┌──────────────────┐
                         │  Held-Out Split  │  (sealed, never seen during training)
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  Week 10 Baseline│  │ Prompt-Engineered│  │  Trained LoRA    │
   │  (no fine-tuning)│  │ (same backbone,  │  │  (same backbone, │
   │                  │  │  better prompt)  │  │  LoRA adapter)   │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                      │
            └─────────────────────┴──────────────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │   Ablation Results │
                        │                    │
                        │  Delta A:          │
                        │   Trained vs       │
                        │   Week10 baseline  │
                        │   → must be +,     │
                        │   p < 0.05 paired  │
                        │   bootstrap 95% CI │
                        │                    │
                        │  Delta B:          │
                        │   Trained vs       │
                        │   Prompt-eng       │
                        │   → report honestly│
                        │   (negative = OK,  │
                        │    publishable)    │
                        │                    │
                        │  Delta C:          │
                        │   Trained vs       │
                        │   τ²-Bench score   │
                        │   (informational)  │
                        │                    │
                        │  Cost-Pareto:      │
                        │   $/task + latency │
                        │   with vs without  │
                        │   trained component│
                        └────────────────────┘
```

---

### 3.7 LLM Routing Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Model Routing Policy                    │
│                                                           │
│  Days 2–3 (Dataset Authoring):                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Frontier Model (Claude Sonnet 4.6 or GPT-5 class)   │ │
│  │  → Hard seed generation (adversarial tasks)          │ │
│  │  → Spot-check judging (50 tasks only)               │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Dev-Tier Model (Qwen3-Next-80B-A3B / DeepSeek V3.2) │ │
│  │  → Bulk task generation (programmatic + synthesis)   │ │
│  │  → High-volume judge filtering (all tasks)           │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  Leakage Prevention Rule:                                │
│  generation_model ≠ judge_model                          │
│  (rotate model families, log in task metadata)           │
│                                                           │
│  Days 4–7 (Training + Eval):                             │
│  Eval-tier allowed for held-out scoring only             │
└───────────────────────────────────────────────────────────┘
```

---

### 3.8 Observability Layer

```
Langfuse (reused from Week 10)
  │
  ├── agent_run traces       → held_out_traces.jsonl
  ├── judge_call traces      → judge_filter logs
  ├── cost tracking          → cost_log.md
  └── latency tracking       → ablation_results.json (Cost-Pareto section)
```

---

### 3.9 Publishing Pipeline

```
Publishing Artifacts
│
├── HuggingFace Hub
│   ├── Dataset: tenacious_bench_v0.1
│   │   ├── train/ + dev/ splits (held_out stays local)
│   │   ├── datasheet.md
│   │   ├── LICENSE (CC-BY-4.0)
│   │   ├── README with quickstart + baseline scores
│   │   └── contamination_check.json
│   │
│   └── Model: tenacious-bench-lora-v0.1 (Path A or C only)
│       ├── LoRA adapter weights (NOT merged backbone)
│       ├── model_card.md
│       └── inference example
│
├── Blog Post (1,200–2,000 words)
│   ├── Section 1: Gap / Audit
│   ├── Section 2: Dataset Construction
│   ├── Section 3: Training Experiment
│   ├── Section 4: Honest Results (including failed Delta B)
│   └── Section 5: What's Next
│
├── memo.pdf (2 pages)
│   ├── Page 1: Decision memo (headline lift, cost delta, recommendation)
│   └── Page 2: Skeptic's appendix (4 failure modes, kill-switch condition)
│
└── Community Engagement (one of):
    ├── GitHub issue/discussion on τ²-Bench repo
    ├── Workshop submission (NeurIPS / ICLR Tiny Papers / EleutherAI / LMSYS)
    └── PR to related benchmark (BIRD-Critic, AgentBench, ToolBench)
```

---

## 4. Data Flow Diagram

```
Week10 Artifacts ──┐
                   │
Public Datasets ───┤
                   │
Tenacious Assets ──┴──▶ generation/ ──▶ raw_tasks (200–300)
                                │
                                ▼
                         judge_filter.py ──▶ filtered_tasks
                                │
                                ▼
                   contamination_check.py ──▶ clean_tasks
                                │
                                ▼
                         partition_split()
                         ├── train/ (50%)
                         ├── dev/   (30%)
                         └── held_out/ (20%) ─▶ SEAL (gitignore)
                                │
                    ┌───────────┴──────────────┐
                    │                          │
                    ▼                          ▼
             training/                  evaluation/
             ├── format_data()          ├── scoring_evaluator.py
             ├── train.py (Unsloth)     ├── run_on_dev() (Days 2–4)
             └── LoRA adapter           └── run_on_held_out() (Day 6)
                    │                          │
                    └──────────────────────────┘
                                │
                                ▼
                    ablation_results.json
                    (Delta A, B, C + Cost-Pareto)
                                │
                                ▼
                    publishing/
                    ├── HuggingFace upload
                    ├── blog_post.md
                    ├── memo.pdf
                    └── evidence_graph.json
```

---

## 5. Evidence Graph Contract

Every numeric claim in `memo.pdf` and `blog_post.md` must resolve to exactly one of:

| Claim Type | Source |
|---|---|
| Dataset stat (e.g., "263 tasks") | `contamination_check.json` → partition count |
| Score (e.g., "3.42 / 5.0") | `ablation_results.json` → row ID |
| Delta (e.g., "+0.31 Delta A") | `ablation_results.json` → bootstrap CI row |
| Cost (e.g., "$4.20 total") | `cost_log.md` → line item |
| Model behavior (e.g., "tone drift in 38% of traces") | `held_out_traces.jsonl` → trace IDs |
| Paper claim | `papers/` memo → citation + page |

`evidence_graph.json` maps each claim string → source path + line/row reference.

---

## 6. Publication Checklist (Gates Before Any Artifact Goes Public)

- [ ] Datasheet complete (all 7 Gebru sections + Pushkarna layered detail)
- [ ] License correct (CC-BY-4.0 for dataset; check backbone license for model)
- [ ] README runnable (stranger can reproduce in < 1 hour)
- [ ] Reproducibility seed set in all scripts (`random.seed`, `transformers.set_seed`)
- [ ] Held-out sealed and gitignored
- [ ] Contamination report committed (`contamination_check.json`)
- [ ] Model card complete (Path A or C)
- [ ] Attribution clean (no PII, no confidential Tenacious data verbatim)
- [ ] Program staff sign-off received

---

## 7. Cost Budget Tracking

| Bucket | Allocated | Tracking File |
|---|---|---|
| Dataset authoring (cheap LLM) | $3–5 | `cost_log.md` |
| Training (Colab free / RunPod) | $0–5 | `cost_log.md` |
| Held-out evaluation (eval-tier) | $2–3 | `cost_log.md` |
| Reserve | $1–2 | `cost_log.md` |
| **Total cap** | **$10** | |

Hard rules:
- No τ²-Bench retail re-runs (not in budget)
- No eval-tier model on Days 2–3 (dev-tier only during iteration)
- `cost_log.md` is a graded artifact — every API call logged

---

## 8. Delivery Timeline

| Milestone | Date | Deliverables |
|---|---|---|
| **Interim** | Wed 21:00 UTC | Acts I–II: GitHub repo + PDF (bench composition, inter-rater agreement, 3 example tasks, plan for Days 4–7) |
| **Final** | Sat 21:00 UTC | Full repo + memo.pdf + 6-min demo video + HuggingFace URLs + blog URL + community engagement evidence |

### Demo Video Checklist (max 6 min)
1. Show dataset live on HuggingFace
2. Score one task end-to-end with `scoring_evaluator.py`
3. Show one ablation result with Delta A value traced to `ablation_results.json`
4. Show blog post live
5. Show community engagement artifact

---

## 9. Evaluation Observable Map

| Observable | Implementation |
|---|---|
| Reproduction fidelity | `scoring_evaluator.py` + seeded random; ±2pp on fresh clone |
| Probe/task originality | adversarial slice weight = 1.0; multi-LLM routing logged in metadata |
| Mechanism attribution | `ablation_results.json` with Delta A (p<0.05), Delta B (honest even if negative) |
| Cost-quality Pareto | per-task cost + latency columns in `ablation_results.json` |
| Evidence-graph integrity | `evidence_graph.json` resolves every claim in memo + blog |
| Public-artifact quality | Publication checklist above must be fully checked |
