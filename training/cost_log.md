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

## Day 4 (Planned) — Live Judge Filter

*To be filled when `judge_filter.py --live` is run. Estimated cost based on 200 tasks × ~200 input tokens + 15 output tokens at claude-haiku-4-5-20251001 rates ($0.80/MTok in, $4.00/MTok out):*

| Item | Model | Calls (est.) | Input tokens (est.) | Output tokens (est.) | Cost (USD, est.) |
|---|---|---|---|---|---|
| Live judge filter | claude-haiku-4-5-20251001 | 200 | 40,000 | 3,000 | ~$0.044 |

---

## Day 4 (Planned) — SFT Training Run

*To be filled when training is run. Estimates for LoRA fine-tuning on Google Colab T4 (free tier) or paid L4 instance.*

| Item | Provider | Duration (est.) | Cost (USD, est.) |
|---|---|---|---|
| T4 Colab training (~40 steps/min, fp16) | Google Colab | ~45 min | $0.00 (free tier) |
| L4 Colab training (bf16, if T4 OOM) | Google Colab Pro | ~20 min | ~$0.40 |

---

## Day 5–6 (Planned) — Evaluation Runs

| Item | Model | Calls (est.) | Input tokens (est.) | Output tokens (est.) | Cost (USD, est.) |
|---|---|---|---|---|---|
| Eval on held_out (38 tasks × 2 models) | claude-haiku-4-5-20251001 | 76 | 15,200 | 1,140 | ~$0.017 |
| Bootstrap CI (no API calls) | — | 0 | 0 | 0 | $0.00 |

---

## Budget Summary

| Phase | Actual (USD) | Estimated remaining (USD) |
|---|---|---|
| Day 1–3 (complete) | $0.00 | — |
| Day 4 live judge | — | ~$0.04 |
| Day 4 training | — | $0.00–$0.40 |
| Day 5–6 eval | — | ~$0.02 |
| **Total** | **$0.00** | **~$0.06–$0.46** |

*Well within the $10 budget stated in README.md.*

---

## Notes

- All costs are for the Tenx MCP Week 11 challenge only.
- Token counts are estimated; actual counts will be updated when live API calls are made.
- Claude Haiku 4.5 pricing as of 2026-04-29: $0.80/MTok input, $4.00/MTok output.
- DeepSeek Chat pricing (OpenRouter): $0.14/MTok input, $0.28/MTok output (if live multi_llm generation is used).
- No costs associated with open-source model weights download (Unsloth/HuggingFace).

---

*Log version: 1.0 | Last updated: 2026-04-29*
