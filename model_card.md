---
language: en
license: apache-2.0
base_model: unsloth/Qwen3-4B-bnb-4bit
tags:
  - lora
  - orpo
  - b2b-sales
  - preference-learning
  - tenacious
datasets:
  - Chalie-lijalem/tenacious-bench-v0.1
metrics:
  - tenacious_bench_score
---

# tenacious-orpo-qwen3-4b

LoRA adapter fine-tuned on Qwen3-4B via ORPO (Odds Ratio Preference Optimization) for B2B sales email generation. Trained on preference pairs targeting five failure modes identified in a Week 10 production audit of the Tenacious sales agent.

## Evaluation Results

Evaluated on the sealed held-out split of Tenacious-Bench v0.1 (n=32 tasks).

| Arm | Mean Score / 5.0 | 95% CI |
|---|---|---|
| Week 10 baseline (base model) | 4.008 | [3.668, 4.301] |
| Prompt-engineered baseline | 4.172 | [3.887, 4.449] |
| **This adapter (ORPO)** | **4.462** | **[4.193, 4.704]** |

| Delta | Δ | p-value | Result |
|---|---|---|---|
| vs Week 10 baseline (Delta A) | +0.454 | 0.001 | ✓ PASS |
| vs Prompt-engineered (Delta B) | +0.290 | 0.021 | ✓ PASS |

Gain by failure mode:

| Failure mode | Δ |
|---|---|
| tone_drift (38% of Week 10 failures) | +0.679 |
| signal_missing (29%) | +0.393 |
| formulaic (8%) | +0.250 |
| trajectory (21%) | +0.154 |

## Model Details

| Property | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-bnb-4bit` |
| Training method | ORPO — Odds Ratio Preference Optimization (Hong et al., 2024) |
| LoRA rank r | 32 |
| LoRA alpha α | 32 |
| Trainable parameters | 66,060,288 (1.62% of 4.09B total) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training precision | fp16 (T4 GPU) |

## Training Hyperparameters

```python
beta            = 0.1    # ORPO odds-ratio weight λ
learning_rate   = 5e-5
epochs          = 5
batch_size      = 2      # per device
grad_accum      = 8      # effective batch = 16
max_length      = 512
max_prompt_length = 256
optimizer       = "adamw_8bit"
lr_scheduler    = "cosine"
warmup_steps    = 10
seed            = 42
```

## Training Data

381 preference pairs generated from 127 training tasks in Tenacious-Bench v0.1.

- **Chosen**: Gemini Flash-generated outputs following all Tenacious style rules
- **Rejected**: Gemini Flash-generated outputs deliberately violating the task's `failure_mode_tag`
- **Token overlap** (chosen vs rejected): avg 16.8% (range 4.8–26.8%)
- **Format**: TRL canonical — `prompt` = formatted prompt, `chosen`/`rejected` = response text only

Note: Rejected outputs are Gemini-synthesized failure-mode injections, not captured production failures. Cross-family leakage prevention was not applied (Gemini both generates and judges). See dataset card for details.

## Intended Use

Generating B2B sales emails, follow-ups, and objection responses for Tenacious-style outbound workflows. Specifically optimized for:

- Signal-grounded opening lines (references verifiable prospect trigger)
- Zero banned phrases (47-phrase prohibited list)
- Single calendar CTA ([CALENDLY_LINK])
- Word-count compliance
- No pricing on first touch

## How to Use

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Chalie-lijalem/tenacious-orpo-qwen3-4b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a B2B sales assistant for Tenacious. Write direct, signal-led sales messages with no banned phrases."},
    {"role": "user",   "content": "Write an email_outreach for this prospect.\n\nContext:\nProspect: Sarah Chen, VP of Revenue at Lattice. Series C $45M 2023-Q2. LinkedIn post 3 days ago about sales rep ramp time.\n\nConstraints:\n- under 120 words\n- include [CALENDLY_LINK]\n\nWrite only the message body."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

import torch
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations

- English-only. Tested on US B2B SaaS contexts only.
- Optimized for Tenacious-specific banned phrase list and style rules. May not generalize to other sales brands.
- Evaluation dataset is small (n=32 held-out tasks). Confidence intervals are wide.
- `trajectory` failure mode shows smallest improvement (Δ=+0.154) — multi-turn consistency remains partially unresolved.
- Rejected preference pairs are synthetic (Gemini-generated) rather than captured from production failures.

## References

- Hong et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model*. https://arxiv.org/abs/2403.07691
- Dataset: [Chalie-lijalem/tenacious-bench-v0.1](https://huggingface.co/datasets/Chalie-lijalem/tenacious-bench-v0.1)
- Training code: [GitHub](https://github.com/chacha1921/Sales-Agent-Evaluation-Bench)

## License

Apache 2.0 (inherits from Qwen3 base model).
