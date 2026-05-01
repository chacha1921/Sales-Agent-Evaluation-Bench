# Methodology Rationale — Path B: Preference Learning (ORPO + SimPO)

**Version:** 2.0 | **Date:** 2026-05-02 | **Author:** chalie@10academy.org

---

## Path Selection: Why B over A and C

The Week 10 failure taxonomy across 230 tasks shows a **consistency problem, not a capability gap**:

| Failure mode | Frequency | Evidence of existing capability |
|---|---|---|
| `tone_drift` | 38% | Probe P-013: agent passes 5/10 identical trials, fails 5/10 — same task, same model |
| `signal_missing` | 29% | Probe P-005: agent correctly references the layoff signal in `trace_003` (4.0/5) but ignores it in `trace_012` (1.1/5) |
| `trajectory` | 21% | Agent acknowledges objections correctly on 6/10 trials; fails on the other 4 |
| `formulaic` | 8% | Banned opener ("just checking in") appears only in ~40% of follow-up outputs |
| `constraint_violation` | 4% | Word-count failures are near-threshold misses, not categorical |

A 40–60% trigger rate confirms the model already produces correct outputs on these task types — it simply does not do so reliably. **Path A (SFT) is the wrong tool**: SFT teaches new behaviors from demonstration data. When the model already knows the correct behavior, injecting more demonstrations shifts the output distribution but does not resolve the preference inconsistency. Lambert et al. (2024, Tülu 3) show SFT accounts for ≥85% of quality on non-verifiable generation tasks, but the residual inconsistency — exactly what Week 10 exposes — is not addressable by SFT alone.

**Path C (PRM)** targets multi-turn trajectory failures with step-level process rewards. Trajectory failures are 21% of the total. Building a process reward model for 21% of failures while ignoring the dominant 67% (tone_drift + signal_missing) would invert the training priority.

**Path B (preference learning)** directly addresses inconsistency: given the same prompt, the model learns to prefer the correct behavior over the incorrect one it also sometimes produces. This is the problem Week 10 defines.

---

## Why ORPO and SimPO Specifically

Standard DPO requires a frozen reference model to compute the KL penalty, doubling GPU memory. For a T4 (16 GB), this rules out any 7B base.

**ORPO** (Hong et al., 2024) eliminates the reference model entirely by folding the preference loss into the SFT cross-entropy step via a log-odds ratio penalty. The combined objective trains signal-grounding and tone compliance in a single pass without the reference model overhead:

```
L_ORPO = L_SFT + λ · L_OR
L_OR   = -log σ(log(p_θ(chosen)/1−p_θ(chosen)) − log(p_θ(rejected)/1−p_θ(rejected)))
```

**SimPO** (Meng et al., 2024) replaces the reference-model log-ratio with a length-normalized reward and a target margin γ, making the implicit reward proportional to average token log-probability rather than raw sequence probability. This matters for email tasks: a rejected output that violates the word-count constraint is systematically longer; SimPO's length normalization prevents the model from gaming the reward by shortening outputs rather than improving quality. The target margin γ = 1.0 sets a minimum gap between chosen and rejected rewards, preventing near-degenerate pairs from producing trivially small gradient updates.

LIMA (Zhou et al., 2023) establishes that ~1,000 high-quality preference pairs substantially shift output style without degrading general capability. Our 254 pairs (127 train tasks × 2 rejected variants) fall below this, but each pair is targeted at a specific Week 10 failure mode rather than broad-coverage curation — failure-mode density outperforms task-type diversity when the training goal is to suppress a strong pretraining prior (banned phrases are common in base model pretraining).

---

## Training Data Summary

**Format:** TRL/Unsloth ORPOTrainer-compatible `{prompt, chosen, rejected}` with chat messages.

| Property | Value |
|---|---|
| Pairs | 254 (127 train tasks × 2 rejected variants) |
| Source | train split only — dev and held_out untouched |
| Chosen quality | 254/254 (100%) zero banned phrases, signal grounded |
| Rejected signal | 232/254 (91%) contain ≥1 banned phrase |

| Failure mode | Pairs | Week 10 frequency |
|---|---|---|
| `signal_missing` | 122 | 29% |
| `tone_drift` | 68 | 38% |
| `trajectory` | 32 | 21% |
| `formulaic` | 20 | 8% |
| `constraint_violation` | 12 | 4% |

---

## Contamination Status

All three checks PASS (run 2026-05-02):
- **Check 1 (n-gram):** Zero shared 8-grams between held_out (32) and train+dev (198). Five apparent dev-vs-train overlaps are confirmed false positives: generic signal-type phrases ("posted a head of sales job on linkedin") shared across unrelated companies and prospects — not prospect-specific context leakage.
- **Check 2 (embedding):** Zero pairs with cosine similarity ≥ 0.85 between held_out and train+dev (`all-MiniLM-L6-v2`).
- **Check 3 (time-shift):** All 108 tasks with public signal sources (crunchbase_odm, layoffs_fyi) have non-null `signal_time_window`.

---

## Expected Outcomes (Act IV)

| Metric | Week 10 baseline | Target (post-ORPO) |
|---|---|---|
| `signal_grounding_fn` mean (dev) | ~0.50 | ≥ 0.70 |
| `tone_checker_fn` mean (dev) | ~0.60 | ≥ 0.80 |
| `banned_phrase_fn` mean (dev) | ~0.75 | ≥ 0.90 |
| Aggregate score mean (dev) | ~2.5 / 5.0 | ≥ 3.5 / 5.0 |

Success criterion: Δ aggregate score > 0, bootstrap CI p < 0.05 on dev split. Winner of ORPO vs SimPO comparison evaluated once on held_out (Day 6).

---

## References

- Lambert et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv:2411.15124. [`papers/path_specific/path_b/tulu3_memo.md`]
- Zhou et al. (2023). *LIMA: Less Is More for Alignment.* arXiv:2305.11206. [`papers/path_specific/path_b/lima_memo.md`]
- Hong et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model.* arXiv:2403.07691.
- Meng et al. (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward.* arXiv:2405.14734.
