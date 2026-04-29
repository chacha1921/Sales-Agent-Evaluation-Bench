# Audit Memo: What τ²-Bench Retail Fails to Grade for Tenacious B2B Sales

**Word count:** 598 | **Date:** 2026-04-29

---

## The Question

τ²-Bench retail was built for consumer-facing tool-use agents: booking flows, shopping carts,
customer service resolution. Its rubric grades slot-filling accuracy, API call correctness,
and terminal-state validity. These dimensions are structurally irrelevant to Tenacious's
outbound B2B sales context. Five categories of evidence from Week 10 prove the gap is not
cosmetic—it is complete.

---

## Gap 1: No Signal-Grounding Dimension

τ²-Bench never checks whether an email grounds its claim in a verifiable prospect signal.
In trace_042 (PROBE-001), the agent sent a fully generic pitch to a VP of Revenue whose
LinkedIn post three days earlier described her exact problem. τ²-Bench scored 0.82 (PASS).
Tenacious scored 1.2/5 (FAIL). The benchmark was blind to the most important quality
signal in B2B sales: did the agent prove it did its homework?

---

## Gap 2: No Voice or Banned-Phrase Compliance Check

Tenacious maintains a 47-phrase prohibited list covering filler language, corporate jargon,
and manipulation patterns. τ²-Bench has no rubric dimension for brand voice compliance.
In trace_107 (PROBE-002), the agent used "leverage," "synergy," and "synergising" in a
discovery follow-up. τ²-Bench scored 0.88 (PASS). Tenacious scored 0.8/5 (FAIL). In
trace_315 (PROBE-003), "just checking in" appeared in the subject line and body of a
follow-up. Again: τ²-Bench PASS, Tenacious FAIL. Voice compliance is binary and
machine-verifiable. τ²-Bench simply does not ask the question.

---

## Gap 3: No Segment-Aware Rubric

Tenacious targets Series B, Enterprise, and SMB segments with materially different
messaging strategies. In trace_212 (PROBE-005), the agent sent enterprise-tier copy—
referencing "org-wide rollout," "procurement cycle," and "legal review"—to a bootstrapped
12-person agency. τ²-Bench scored 0.79 (PASS). Tenacious scored 1.0/5 (FAIL). τ²-Bench
grades all tasks on a single universal rubric with no segment conditioning. Tenacious's
scoring is segment-stratified by design.

---

## Gap 4: No Multi-Turn Trajectory Integrity Check

τ²-Bench evaluates terminal state. A Tenacious agent that opens with a strong, signal-led
paragraph but closes with "let me know if you're interested" degrades the full conversation.
In trace_401 (PROBE-008), the agent executed turns 1–4 cleanly, then reverted to a generic
close on turn 5. τ²-Bench scored 0.91 (PASS—meeting was scheduled). Tenacious scored
2.1/5 (PARTIAL). PROBE-010 confirmed the same pattern: the agent pivoted on an objection
without acknowledging it, which τ²-Bench could not detect.

---

## Gap 5: No CTA or Constraint Verification

τ²-Bench does not check whether an outreach email includes a calendar link or respects
a word-count constraint. In trace_042 (PROBE-004), the email ended with "let me know if
you'd like to connect" and no link. In trace_212 (PROBE-009), the agent produced a
287-word email against a 120-word constraint. Both passed τ²-Bench. Both failed Tenacious.
These are deterministic, zero-ambiguity checks. The fact that τ²-Bench omits them reflects
its retail origins, not a scoring philosophy.

---

## What This Means

Across probes PROBE-001 through PROBE-010 and traces trace_042, trace_107, trace_212,
trace_315, and trace_401, the Week 10 agent's τ²-Bench scores (mean 0.85) hid a Tenacious-
Bench score range of 0.8–2.1/5.0—every task below the 3.5 passing threshold. The
benchmark was not measuring what matters. Tenacious-Bench v0.1 fixes all five gaps with
machine-verifiable rubric dimensions. No human in the loop is required to produce a score.
