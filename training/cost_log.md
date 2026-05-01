# API Cost Log — Week 11 Training Run

All API costs incurred during dataset generation, filtering, and evaluation are itemized here.
Costs are in USD. Model IDs are exact to enable reproducibility checks.

---

## Day 1 — Audit and Schema Design (2026-04-29)

| Item | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| Scoring evaluator demo (--demo flag) | claude-haiku-4-5-20251001 | 0 | 0 | 0 | $0.00 |
| Schema annotation examples (manual) | — | — | — | — | $0.00 |
| **Day 1 Total** | | | | | **$0.00** |

*Note: All Day 1 work was code authoring and schema design. No LLM API calls were made — scoring evaluator ran in `--mock-llm` mode.*

---

## Day 2 — Dataset Generation (2026-04-29)

| Item | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| `trace_derived.py --mock` | — (template) | 0 | 0 | 0 | $0.00 |
| `programmatic.py --mock` | — (template) | 0 | 0 | 0 | $0.00 |
| `multi_llm_synthesis.py --mock` | — (template) | 0 | 0 | 0 | $0.00 |
| `adversarial.py` (hand-authored) | — | 0 | 0 | 0 | $0.00 |
| `judge_filter.py --mock` | — (heuristic) | 0 | 0 | 0 | $0.00 |
| **Day 2 Total** | | | | | **$0.00** |

*Note: All generation scripts ran in `--mock` mode (template expansion). No API calls were made on Day 2.*

---

## Day 3 — Contamination Check + IRA (2026-04-29)

| Item | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| `contamination_check.py --skip-embedding` | — | 0 | 0 | 0 | $0.00 |
| **Day 3 Total** | | | | | **$0.00** |

---

## Day 0 — Pre-flight (2026-05-01)

| Timestamp | Item | Provider | Model / Resource | Cost (USD) |
|---|---|---|---|---|
| 2026-05-01 | τ²-Bench baseline run — 150 retail tasks | OpenAI / τ²-Bench | Agent API calls | $2.99 |
| 2026-05-01 | Unsloth starter notebook — Qwen2.5-0.5B-Instruct dummy LoRA | Google Colab T4 | T4 GPU (free tier) | $0.00 |
| 2026-05-01 | Adapter push to HuggingFace (`Chalie-lijalem/tenacious-test`) | HuggingFace | — | $0.00 |
| **Day 0 Total** | | | | **$2.99** |

*Note: τ²-Bench run ($2.99) used for baseline evaluation. Traces replaced with Tenacious B2B traces — retail domain not applicable to project. Cost logged for budget accountability.*

---

## Day 4 (Planned) — Preference Pair Generation + Training

*To be filled when training is run on Colab T4.*

| Item | Provider | Model / Resource | Duration (est.) | Cost (USD, est.) |
|---|---|---|---|---|
| `generate_preference_pairs.py --n-rejected 2` | Local | — (template) | <1 min | $0.00 |
| ORPO training run — 198 pairs, 3 epochs | Google Colab T4 | Qwen2.5-0.5B-Instruct + LoRA | ~30 min | $0.00 (free tier) |
| SimPO training run — 198 pairs, 3 epochs | Google Colab T4 | Qwen2.5-0.5B-Instruct + LoRA | ~30 min | $0.00 (free tier) |
| Adapter push ×2 to HuggingFace | HuggingFace | — | — | $0.00 |

---

## Day 5–6 (Planned) — Evaluation Runs

| Item | Model | Calls (est.) | Input tokens (est.) | Output tokens (est.) | Cost (USD, est.) |
|---|---|---|---|---|---|
| Dev eval — ORPO adapter (63 tasks) | claude-haiku-4-5-20251001 | 63 | 12,600 | 945 | ~$0.014 |
| Dev eval — SimPO adapter (63 tasks) | claude-haiku-4-5-20251001 | 63 | 12,600 | 945 | ~$0.014 |
| Held-out ablation (38 tasks × winner) | claude-haiku-4-5-20251001 | 38 | 7,600 | 570 | ~$0.008 |
| Bootstrap CI (no API calls) | — | 0 | 0 | 0 | $0.00 |

---

## Budget Summary

| Phase | Actual (USD) | Estimated remaining (USD) |
|---|---|---|
| Day 1–3 (complete) | $0.00 | — |
| Day 0 pre-flight (τ²-Bench run) | $2.99 | — |
| Day 4 training (Colab free tier) | — | $0.00 |
| Day 5–6 eval (LLM judge calls) | — | ~$0.04 |
| **Total** | **$2.99** | **~$0.04** |
| **Grand total (est.)** | | **~$3.03** |

*Well within the $10 budget stated in README.md. $2.99 already spent on τ²-Bench baseline.*

---

## Notes

- All costs are for the Tenx MCP Week 11 challenge only.
- Token counts are estimated; actual counts will be updated when live API calls are made.
- Claude Haiku 4.5 pricing as of 2026-04-29: $0.80/MTok input, $4.00/MTok output.
- DeepSeek Chat pricing (OpenRouter): $0.14/MTok input, $0.28/MTok output (if live multi_llm generation is used).
- No costs associated with open-source model weights download (Unsloth/HuggingFace).

---

*Log version: 1.1 | Last updated: 2026-05-01*
