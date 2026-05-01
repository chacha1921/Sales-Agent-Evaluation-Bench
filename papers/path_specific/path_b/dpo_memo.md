# Path B Memo: DPO — Direct Preference Optimization (Rafailov et al., NeurIPS 2023)

**Paper:** Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., NeurIPS 2023)
**Role:** Foundational algorithm — the baseline that ORPO and SimPO improve upon

---

## Key Contribution

DPO replaces the three-step RLHF pipeline (SFT → reward model → PPO) with a single closed-form objective. The key insight: the optimal policy under the KL-constrained RLHF objective can be expressed directly as a function of the log-ratio between the trained policy and a frozen reference model. This eliminates the reward model and the PPO loop entirely.

The DPO loss is:

```
L_DPO = -E[ log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x)) - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]
```

where `y_w` is the chosen response, `y_l` the rejected, `π_ref` the frozen reference, and `β` controls the KL penalty strength.

---

## Why DPO is the starting point but not the choice

DPO requires a **frozen reference model** — a copy of the base model loaded in memory throughout training. On a T4 GPU (16GB VRAM), loading Qwen2.5-0.5B twice (base + LoRA-adapted) plus optimizer states and activations for a batch size of 4 exceeds available memory at fp16.

More importantly, DPO's implicit reward is sensitive to the quality of the reference model. If the reference model already produces weak outputs (which is why we are training in the first place), the log-ratio term provides a noisy signal. ORPO and SimPO both remove this dependency.

DPO is documented here as the theoretical foundation — ORPO and SimPO are both derived from it. Understanding the DPO objective makes the modifications in each variant interpretable.

---

## Connection to preference pairs

The `training/generate_preference_pairs.py` script produces pairs in the format DPO requires: `(prompt, chosen, rejected)`. The same pairs are used by both ORPO (`ORPOTrainer`) and SimPO (`CPOTrainer` with `loss_type="simpo"`). The data format is algorithm-agnostic; only the loss function changes.

The 198 preference pairs use failure-mode-specific rejection construction:
- `tone_drift` rejections inject tier-1 banned phrases
- `signal_missing` rejections strip signal references and use generic templates
- `formulaic` rejections prepend banned openers ("My name is X and I'm reaching out from")

This is stronger than standard DPO data construction (which often uses model-sampled rejections) because the rejections target known failure modes from the Week 10 taxonomy rather than random worse outputs.

---

*~340 words | Role: foundational algorithm — not used directly due to VRAM constraints and reference model noise. ORPO and SimPO are the implementations of choice.*
