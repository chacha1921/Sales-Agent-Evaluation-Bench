# Path B Memo: ORPO and SimPO — Reference-Free Preference Optimization

**Papers:**
- ORPO: Monolithic Preference Optimization without Reference Model (Hong, Lee, and Thorne, EMNLP 2024)
- SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng, Xia, and Chen, NeurIPS 2024)

**Role:** The two training methods being compared; one will be selected based on dev split results

---

## ORPO

ORPO eliminates the reference model by combining SFT loss and preference loss into a single objective:

```
L_ORPO = L_SFT + λ · L_OR
```

The odds-ratio term `L_OR` penalises the model for assigning high likelihood to rejected responses *relative to chosen responses*. Crucially, the reference is implicit — it is the model's own current state, not a frozen copy. This means:
- No second model copy in memory
- SFT and preference alignment happen simultaneously
- Training is more stable than DPO because the signal is anchored to the current policy

Hong et al. report ORPO outperforms DPO on AlpacaEval and MT-Bench with the same data, and at lower memory cost.

**Key hyperparameter:** `β` (odds-ratio weight, `lambda` in the paper). Default 0.1. Higher values increase preference signal strength; lower values weight SFT more heavily. For a small dataset (198 pairs), β = 0.1 is conservative and appropriate.

---

## SimPO

SimPO replaces DPO's log-ratio reward with a **length-normalised average log-likelihood**:

```
r_SimPO(x, y) = (1/|y|) * Σ log π_θ(y_t | x, y_{<t})
```

and adds a **target reward margin γ** that the chosen response must exceed the rejected by:

```
L_SimPO = -E[ log σ( β/|y_w| * log π(y_w|x) - β/|y_l| * log π(y_l|x) - γ ) ]
```

The length normalisation prevents the model from preferring shorter responses simply because they have higher raw log-likelihood. The margin γ creates a minimum gap — the model must be clearly more confident about chosen than rejected, not just marginally so.

Meng et al. report SimPO outperforms DPO and RAFT on AlpacaEval 2 by 6–10 points with no reference model. SimPO tends to produce more decisive, confident outputs; ORPO tends to be more conservative.

**Key hyperparameters:** `β` (temperature, default 2.0) and `γ` (margin, default 1.0). For our domain, `γ = 1.0` means the model must assign at least 1.0 higher normalised log-likelihood to a rubric-compliant email than to a banned-phrase-filled one — a reasonable margin given how different the outputs are.

---

## Why both are being run

The choice between ORPO and SimPO is empirical for this dataset:
- ORPO's joint SFT+preference loss may be better for a small dataset (198 pairs) where SFT signal helps prevent catastrophic forgetting
- SimPO's length normalisation may be more important for Tenacious tasks, where word-count limits mean correct outputs are often shorter than rejected outputs (rejected outputs include banned phrases, injected padding, or pricing text that inflates length)

`training/compare_methods.py` scores both adapters on the 63-task dev split and reports bootstrap CIs. The winner is used for the held-out ablation.

---

*~450 words | ORPO: joint SFT+preference, implicit reference, β=0.1. SimPO: length-normalised reward, explicit margin γ=1.0, more decisive outputs. Empirical comparison on dev split selects the final method.*
