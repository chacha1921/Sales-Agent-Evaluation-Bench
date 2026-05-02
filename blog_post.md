# When 0.73 Is a Lie: Building a Sales-Specific Eval Bench from Production Failures

*Technical blog post — Tenacious-Bench v0.1 | 2026-05-02 | [Published on Substack](https://chalielijalem.substack.com/p/building-the-sales-evaluation-bench)*

---

## The Gap

Our B2B sales agent scored 0.7267 on τ²-Bench retail held-out. On the surface, that looks reasonable — above 70%, within the confidence interval of competitive agents on the leaderboard. We shipped it.

Then we looked at what it was actually writing.

In one trace, the agent sent this email to a VP of Revenue at a Series B SaaS company: *"I hope this email finds you well. Tenacious can help you leverage your existing data to synergize across your sales stack. Our end-to-end, best-in-class platform is game-changing for revenue teams. I wanted to reach out to see if you'd be open to a quick chat. [CALENDLY_LINK]"*

Six banned phrases. No reference to the $32M Series B or the open Head of RevOps role in the context. τ²-Bench score: 0.82 (PASS). Tenacious internal rubric score: 0.8/5 (FAIL).

This is the failure mode a general benchmark is structurally unable to catch. τ²-Bench was built for e-commerce slot-filling — hotel bookings, shopping cart flows, customer service resolution. Its rubric grades whether the agent reached the correct terminal state (booking confirmed, item added, ticket resolved). It has no dimension for whether the agent's *words* are appropriate for the task. A sales email that books a meeting via manipulation still passes τ²-Bench. A sales email that fails to book a meeting but builds trust does not.

Across 30 probe runs, we found the agent had five systematic failure categories: **tone drift** (banned phrases and jargon, 38% frequency), **missing signal grounding** (ignoring the prospect trigger, 29%), **trajectory inconsistency** (ignoring objection history, 21%), **formulaic openers** (self-introduction, "just checking in", 8%), and **constraint violations** (over word limit, pricing on first touch, 4%). τ²-Bench caught zero of these. Mean τ²-Bench score during those failures: 0.82.

---

## The Audit Method

The audit started with a probe library: 30 test cases constructed around the five failure modes. Each probe contained a realistic prospect context (segment, role, company, funding signal, pain point), a task type, explicit constraints, and a known failure trigger — a piece of context the agent reliably mishandled.

We ran the production agent against all 30 probes and scored each output on both τ²-Bench and a hand-calibrated Tenacious rubric. The divergence was consistent: τ²-Bench scores clustered around 0.71–0.88 (all PASS); Tenacious rubric scores ranged from 0.6 to 1.4/5 (all FAIL). The agent was passing a benchmark that could not see its failures.

The five gaps were not corner cases. Tone drift was present in 38% of runs on the same task type it passed 62% of the time. This is a consistency problem, not a capability gap. The model knew how to write clean, signal-led emails — it just did not do so reliably.

---

## The Dataset

To measure this reliably at scale, we needed a benchmark. We built **Tenacious-Bench v0.1**: 230 tasks across five failure modes, three prospect segments (SMB, Series B, Enterprise), five task types (email outreach, follow-up, discovery response, objection handling, closing), and four authoring modes.

**Authoring modes and why they matter:**

*Trace-derived (30 tasks)* — Started from five real production failures. Each became six task variants, ensuring the benchmark tests exactly the scenarios the agent already failed on. These are the hardest tasks.

*Programmatic (75 tasks)* — 15 prospect profiles × 5 task types. Maximizes task-type diversity while controlling for scenario confounds. Every profile was manually reviewed against the Style Guide before inclusion.

*Multi-LLM synthesis (90 tasks)* — 18 seed scenarios × 5 variation configs. Bulk seeds via Gemini 2.5-flash; high-adversarial seeds via DeepSeek Chat via OpenRouter. Cross-family routing is intentional: Gemini-generated tasks are judged by DeepSeek, and vice versa, to prevent model-specific idiom from gaming the judge filter.

*Adversarial (35 tasks)* — Hand-authored traps. Six tasks with "leverage" and "synergy" buried mid-sentence. Eight tasks with a word limit of exactly 75 words. Seven tasks where the conversation history contains a lie the agent told two turns ago.

**Two hard design choices:**

The judge filter rejected zero tasks (230/230 passed ≥3.5/5). That sounds suspicious, so it is worth naming: we ran the filter on the output of a *second* LLM, not the generating LLM. The filter did reject 12 tasks during development — tasks where the rubric dimension was ambiguous enough that a machine could not reliably verify it. We rewrote those rubrics before generating final tasks.

The IRA protocol failed the first time. Our `tone_checker_fn` mock heuristic was giving partial credit to "just checking in" and missing "My name is Alex" openers. Cohen's κ on the 30-task calibration sample was 0.662 — below our 0.70 threshold. We rewrote the function with a two-tier rule: tier-1 phrases return 0.0 immediately with no partial credit. Round 2 κ = 1.000 across all rater pairs.

Total dataset cost: **$0.021**. All three contamination checks (n-gram overlap, embedding similarity ≥0.85, time-shift verification) passed on the sealed held-out partition.

---

## The Training Experiment

We chose Path B: preference learning via ORPO and SimPO. The reasoning: the agent already knew how to write correct emails (pass rate was 40–60%, not 0%). SFT teaches new behaviors; preference learning teaches the model to *prefer its correct outputs over its incorrect ones*. That is the right tool for a consistency problem.

**What worked:**

The data format bug took five training runs to find. Runs 1 through 3 produced perfect NLL loss curves but zero preference separation — `rewards/accuracies` stayed random (0.10–0.20) for all 45–110 steps. The root cause: our `to_hf_dataset` function was passing full conversations in `chosen` and `rejected`. Both included the same system prompt and user message (~70% of tokens). The per-token average log-probability was nearly identical for chosen and rejected regardless of the output content, so the preference gradient was zero. The fix was one line: pass response text only in `chosen`/`rejected`, not the full conversation.

Run 4 — response-only format, live preference pairs (16.8% avg token overlap), r=32 — produced the breakthrough. `rewards/accuracies` hit 1.000 by epoch 2 for both ORPO and SimPO. `log_odds_chosen` rose from -0.561 to +1.741 across 5 epochs.

**What did not work (and is worth naming):**

Preference pairs built from Python templates (mock mode) are useless for preference training. At 90% token overlap between chosen and rejected, the odds-ratio loss gets no gradient signal. Three rounds and six training runs wasted on mock data. The lesson: run `generate_preference_pairs.py --live` before training, not after.

SimPO with γ=2.0 on mock pairs proved this mathematically: loss plateau = log(1 + e²) = 2.127, exactly. The plateau is the theoretical floor when chosen and rejected are identical sequences. It never moves.

---

## The Honest Result

Evaluated on 32 sealed held-out tasks, with three comparison arms.

| Arm | Mean / 5.0 | 95% CI |
|---|---|---|
| Week 10 baseline | 4.008 | [3.668, 4.301] |
| Prompt-engineered baseline | 4.172 | [3.887, 4.449] |
| Trained ORPO adapter | **4.462** | **[4.193, 4.704]** |

**Delta A (trained vs Week 10): Δ=+0.454, 95% CI [0.153, 0.787], p=0.001.**

This passes the primary requirement. The improvement is real, not noise, and the confidence interval excludes zero.

**Delta B (trained vs prompt-engineered): Δ=+0.290, 95% CI [0.012, 0.583], p=0.021.**

This is the more interesting result. A carefully written system prompt that lists every banned phrase and style rule lifted the score to 4.172 — already significantly above baseline. Training then added another +0.290 on top of that. The honest read: prompt engineering gets you 60% of the way there for free. Training gets you the remaining 40%, at a cost of one T4 training run (~44 minutes, ~$0 on Colab free tier).

Biggest improvement on `tone_drift` (+0.679) — the most frequent Week 10 failure mode and the one most directly targeted by the preference pairs. Smallest on `trajectory` (+0.154) — multi-turn consistency remains partially unresolved because the preference pairs used a single-exchange proxy, not real conversation trajectories.

τ²-Bench retail comparison (Delta C, informational): We have the Week 10 τ²-Bench score on file (pass@1=0.7267). We did not re-run τ²-Bench this week. The Tenacious-Bench improvement does not tell us whether the adapter also improves general retail task completion — that is a question for a future experiment.

---

## What Is Next

Three things remain genuinely unresolved:

**Real probe-triggered rejected pairs.** The current preference pairs use Gemini to simulate bad outputs. Real rejected examples should come from the production agent's actual failures — the outputs it generated during the Week 10 probes. These are richer, more naturalistic, and harder for the model to "cheat" on.

**Cross-family leakage prevention.** Gemini both generated the chosen outputs and (when API key is available) judges the tone dimension. This is a methodological gap. The next training round should route generation through Claude or DeepSeek and keep Gemini as the judge, or vice versa.

**Trajectory.** Four of our 32 held-out tasks test multi-turn consistency. The adapter improved trajectory scores by only +0.154. To fix this properly, we need preference pairs that span full conversation histories, not single exchanges.

The dataset and adapter are publicly available. If you are building domain-specific evals for a different sales or support context, the pipeline (authoring modes, judge filter calibration, contamination protocol, IRA) is fully reproducible from the repo.

---

*Dataset: [huggingface.co/datasets/Chalie-lijalem/tenacious-bench-v0.1](https://huggingface.co/datasets/Chalie-lijalem/tenacious-bench-v0.1)*  
*Model: [huggingface.co/Chalie-lijalem/tenacious-orpo-qwen3-4b](https://huggingface.co/Chalie-lijalem/tenacious-orpo-qwen3-4b)*  
*Code: [github.com/chacha1921/Sales-Agent-Evaluation-Bench](https://github.com/chacha1921/Sales-Agent-Evaluation-Bench)*
