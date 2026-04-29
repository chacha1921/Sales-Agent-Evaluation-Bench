# Week 10 Failure Taxonomy

Systematic classification of agent failure modes observed during Week 10 evaluation.
Used as seed for Tenacious-Bench v0.1 task design and training path selection.

---

## Failure Mode 1: tone_drift

**Frequency:** 38% of failing traces
**Description:** Agent output contains banned phrases, pushy language, or deviates from
Tenacious voice (direct, signal-led, non-pushy, concise, human).
**Root cause:** Base model default toward filler phrases and corporate jargon.
**τ²-Bench visibility:** Not graded (τ²-Bench has no voice compliance dimension).
**Representative probes:** PROBE-002, PROBE-003, PROBE-006, PROBE-007
**Representative traces:** trace_042, trace_107

---

## Failure Mode 2: signal_missing

**Frequency:** 29% of failing traces
**Description:** Agent sends generic pitch without grounding claims in a verifiable
prospect signal (funding, headcount, job posting, layoff event).
**Root cause:** Context window compression drops signal details; agent falls back to template.
**τ²-Bench visibility:** Not graded.
**Representative probes:** PROBE-001, PROBE-004
**Representative traces:** trace_212, trace_315

---

## Failure Mode 3: trajectory

**Frequency:** 21% of failing traces
**Description:** Locally good steps compound into globally off-brand outcomes. The most
common pattern is a strong signal-led opening that degrades into a generic close.
**Root cause:** Agent optimises per-turn without modeling the arc of the full conversation.
**τ²-Bench visibility:** Partially captured (terminal state), but early trajectory drift is invisible.
**Representative probes:** PROBE-008, PROBE-010
**Representative traces:** trace_401

---

## Failure Mode 4: formulaic

**Frequency:** 8% of failing traces
**Description:** Agent uses segment-inappropriate messaging (enterprise copy to SMB or
vice versa) or structurally identical emails regardless of prospect context.
**Root cause:** No segment-conditioning in the system prompt.
**τ²-Bench visibility:** Not graded.
**Representative probes:** PROBE-005, PROBE-007
**Representative traces:** trace_107

---

## Failure Mode 5: constraint_violation

**Frequency:** 4% of failing traces
**Description:** Agent violates explicit task constraints (word count, forbidden content).
**Root cause:** Instructions in long system prompts are under-attended.
**τ²-Bench visibility:** Partially graded (output format), but word count not checked.
**Representative probes:** PROBE-009
**Representative traces:** trace_042

---

## Training Path Recommendation

Based on frequency and severity:
- **tone_drift** (38%) → addressable with Path A (SFT) or Path B (DPO)
- **signal_missing** (29%) → addressable with Path A (SFT on signal-grounded examples)
- **trajectory** (21%) → addressable with Path C (PRM)

**Recommended path for Week 11:** Path A (SFT) — tone_drift and signal_missing account
for 67% of failures and are generation-quality problems, not consistency or trajectory
problems. Path A is also lowest cost and most interpretable.
