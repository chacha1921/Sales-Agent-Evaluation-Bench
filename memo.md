# Tenacious Sales Agent — Fine-Tuning Results

<!-- **To:** CEO, CFO — Tenacious &nbsp;|&nbsp; **From:** AI/ML Engineering &nbsp;|&nbsp; **Date:** 2026-05-02

--- -->

**Executive Summary.** We built Tenacious-Bench v0.1 (230 tasks, 5 failure modes) and fine-tuned a Qwen3-4B LoRA adapter via ORPO on 381 live preference pairs. On 32 sealed held-out tasks the adapter scored 4.462/5.0 vs. 4.008 baseline (+0.454, 95% CI [0.153, 0.787], p=0.001, paired bootstrap n=2000), beating a prompt-engineered baseline on the same backbone by +0.290 (p=0.021). **Recommendation: deploy with caveat.**

| Arm | Mean / 5.0 | 95% CI |
|---|---|---|
| Week 10 baseline | 4.008 | [3.668, 4.301] |
| Prompt-engineered (same backbone) | 4.172 | [3.887, 4.449] |
| **Trained ORPO adapter** | **4.462** | **[4.193, 4.704]** |

**Delta A (+0.454, p=0.001):** paired bootstrap, 2000 resamples, 32 tasks. CI [0.153, 0.787] excludes zero; passes pre-registered p<0.05 threshold.

**Delta B (+0.290, p=0.021):** The prompt-engineered baseline used the identical Qwen3-4B backbone with a system prompt listing all 47 banned phrases — same model, same intervention shape — and scored 4.172 (+0.164 alone). Training adds +0.290 on top; prompt engineering accounts for ~36% of total lift.

| | Cost/task | Latency |
|---|---|---|
| Week 10 baseline (API) | $0.0004 | 4.6 s |
| Trained adapter (LoRA, T4) | $0.00024 (−40%) | 9.1 s (+98%) |

LoRA runs locally vs. a pay-per-token API. At current volume 9.1 s is acceptable; a dedicated T4 endpoint is warranted at 10× scale.

**Recommendation: Deploy with caveat.** Two conditions: (1) banned-phrase regex filter active before go-live; (2) dedicated inference endpoint before high-concurrency use. Basis: Delta A (+0.454, p=0.001) exceeds the threshold; Delta B (+0.290, p=0.021) confirms training beats prompt engineering; 40% cost reduction makes LoRA economically dominant.

<div style="page-break-after: always;"></div>

## Page 2 — Skeptic's Appendix

**Four Uncaptured Failure Modes**

1. **Multi-turn factual consistency.** Bench cannot grade self-contradiction across turns (no task spans >1 exchange). v0.2: 12 four-turn tasks with `factual_consistency_fn` checking turn 4 vs. a claim from turn 1.
2. **Competitor-named objections.** No task names a specific competitor; accuracy and disparagement in comparisons are ungraded. v0.2: 10 tasks with named-competitor context and disparagement flag.
3. **Warm-intro acknowledgment.** All tasks are cold-outreach; no referral-signal task exists. v0.2: 8 tasks with "referred by [Name]" context, graded on opener acknowledgment.
4. **Closing recovery.** Closing tasks test single-message CTAs only. v0.2: 10 two-turn tasks where turn 1 is a soft decline and turn 2 is graded on the de-escalated alternative.

**Signal Lossiness.** The 108 signal-grounded tasks use public signals with a 30–90 day lag. The `signal_missing` checker rewards referencing any signal regardless of staleness, systematically over-rewarding agents that cite stale funding rounds. The +0.393 signal_missing Delta A contribution is a ceiling, not a floor.

**Unresolved Training Failure.** Trajectory (Δ=+0.154, smallest gain): on 3 of 4 trajectory-tagged held-out tasks the adapter wrote "that makes sense" then restarted the pitch unchanged — surface acknowledgment without substantive response. Tried: single-exchange preference pairs. Not resolved: model learned the phrase, not the reasoning. Fix requires multi-turn preference pairs spanning 3-turn conversations.

**Kill-Switch.** Metric: banned-phrase detection rate, 200-email rolling window (`dataset/banned_phrases.txt`), via pre-send regex — no benchmark re-run needed. Threshold: **>5%** (≥11 emails). Justification: pre-training tone_drift rate was ~38%; adapter improved it by +0.679 (largest gain); 5% is 13× above near-zero post-training level. Action: disable adapter, revert to prompt-engineered base model (4.172/5.0). Do not revert to the unmodified base model. Cost: $0.
