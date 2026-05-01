# Audit Memo: What τ²-Bench Retail Fails to Grade for Tenacious B2B Sales

**Word count:** 600 | **Date:** 2026-05-01

---

## The Question

τ²-Bench retail was built for consumer-facing tool-use agents: booking flows, shopping
carts, customer service resolution. Its rubric grades slot-filling accuracy, API call
correctness, and terminal-state validity. These dimensions are structurally irrelevant to
Tenacious's outbound B2B sales context. Five categories of evidence from Week 10 prove
the gap is not cosmetic — it is complete.

---

## Gap 1: No Signal-Grounding Dimension

τ²-Bench never checks whether an email grounds its claim in a verifiable prospect signal.
In trace_004 (P-005), the agent sent a fully generic pitch to Aisha Kamau at BrightPath
despite the context containing a specific $8M Series A signal and a named pain point
(inconsistent pipeline forecasting). τ²-Bench scored 0.82 (PASS). Tenacious scored 1.2/5
(FAIL). The benchmark was blind to the most important quality signal in B2B sales: did the
agent prove it did its homework? trace_016 confirms the same failure mode — 4 open roles
existed, the constraint explicitly said not to overstate hiring velocity, yet the agent
wrote "scaling rapidly" and "explosive growth momentum." τ²-Bench: 0.81 PASS. Tenacious:
1.3/5 FAIL.

---

## Gap 2: No Voice or Banned-Phrase Compliance Check

Tenacious maintains a 47-phrase prohibited list covering filler language, corporate jargon,
and manipulation patterns. τ²-Bench has no rubric dimension for brand voice compliance.
In trace_002 (P-013), the agent used "leverage," "synergy," "revolutionary," "end-to-end,"
"game-change," and "quick chat" — six banned phrases — in a single email. τ²-Bench scored
0.79 (PASS). Tenacious scored 0.8/5 (FAIL). In trace_005 (P-013, P-015), "just checking
in" and "circle back" appeared in a follow-up despite an explicit constraint prohibiting
both. τ²-Bench: 0.71 PASS. Tenacious: 0.6/5 FAIL. Voice compliance is binary and
machine-verifiable. τ²-Bench simply does not ask the question.

---

## Gap 3: No Segment or Context-Aware Rubric

Tenacious serves Series B, Enterprise, and SMB prospects with materially different
messaging strategies. In trace_012 (P-001), the agent received a prospect with both a
Series B funding signal ($32M, 90 days ago) and a layoff event (18% headcount cut, 45
days ago). The constraint required selecting the correct segment — layoff overrides
funding within 120 days. The agent ignored the layoff and sent a growth-pitch. τ²-Bench
scored 0.88 (PASS — prospect replied). Tenacious scored 1.1/5 (FAIL). Segment mismatch
is invisible to a benchmark with no segment conditioning.

---

## Gap 4: No Multi-Turn Trajectory Integrity Check

τ²-Bench evaluates terminal state only. A Tenacious agent that handles objections without
acknowledging them degrades the relationship invisibly. In trace_008 (P-014), the prospect
said "we already have this covered, we built it in-house." The agent responded immediately
with a capabilities pitch — no acknowledgment, no softening phrase. τ²-Bench scored 0.85
(PASS — prospect agreed to a demo). Tenacious scored 1.4/5 (FAIL on objection_ack). The
meeting was booked, but the response would damage trust before it started. P-009 confirms
the same gap at higher cost: trace_018 shows the agent committing to 10 Go engineers when
the bench showed 3 available. τ²-Bench: 0.83 PASS. Tenacious: 1.0/5 FAIL.

---

## Gap 5: No Constraint Verification

τ²-Bench does not check whether outreach emails include a calendar link, respect a
word-count constraint, or exclude pricing language. In trace_009 (P-013), the email
opened with "My name is Jordan and I'm reaching out from Tenacious" — a formulaic opener
banned in the Tenacious style guide — while the constraint explicitly required a
signal-led open. Both passed τ²-Bench. All failed Tenacious. These are deterministic,
zero-ambiguity checks.

---

## What This Means

Across P-001, P-005, P-009, P-013, P-014, P-015, P-028, P-030 and traces trace_002,
trace_004, trace_005, trace_008, trace_009, trace_012, trace_016, trace_018, the Week 10
agent's τ²-Bench scores (mean 0.82) hid Tenacious-Bench scores ranging from 0.6 to 1.4 —
every task below the 3.5 passing threshold. The benchmark was not measuring what matters.
Tenacious-Bench v0.1 fixes all five gaps with machine-verifiable rubric dimensions and no
human in the loop.
