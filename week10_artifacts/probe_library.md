# Week 10 Probe Library

Probes used to stress-test the Tenacious B2B sales agent during Week 10 evaluation.
Each probe is a targeted test input designed to expose a specific failure mode.

---

## PROBE-001 — Signal Grounding: No Public Signal Referenced

**Category:** signal_missing
**Segment:** Series B
**Input:** Prospect Sarah Chen, VP of Revenue at Lattice (Series C, $45M, 2023-Q2). LinkedIn post 3 days ago about sales rep ramp time.
**Expected behavior:** Email opens by referencing the LinkedIn signal or funding round.
**Observed behavior:** Agent sent a fully generic pitch with no reference to any signal.
**τ²-Bench result:** PASS (slot-filling correct, terminal state valid)
**Tenacious result:** FAIL

---

## PROBE-002 — Tone Drift: Banned Phrase "Leverage"

**Category:** tone_drift
**Segment:** Enterprise
**Input:** Discovery follow-up for Mark Rivera, Head of Sales Ops at 500-person SaaS.
**Expected behavior:** Direct, human follow-up referencing discovery notes.
**Observed behavior:** Agent used "leverage your existing stack" and "synergy with your RevOps motion."
**τ²-Bench result:** PASS
**Tenacious result:** FAIL (2 banned phrases)

---

## PROBE-003 — Banned Phrase: "Just Checking In"

**Category:** tone_drift
**Segment:** SMB
**Input:** Third follow-up to unresponsive prospect.
**Expected behavior:** Value-add follow-up referencing a new signal or offering an out.
**Observed behavior:** "Hi James, just checking in to see if you had a chance to review..."
**τ²-Bench result:** PASS
**Tenacious result:** FAIL (banned phrase in subject line and body)

---

## PROBE-004 — Missing CTA: No Calendar Link

**Category:** cta_missing
**Segment:** Series B
**Input:** Cold outreach to first-touch prospect with clear buying signal.
**Expected behavior:** Email ends with Calendly link or explicit meeting offer.
**Observed behavior:** Email ends with "Let me know if you'd like to connect." No link.
**τ²-Bench result:** PASS
**Tenacious result:** FAIL

---

## PROBE-005 — Segment Mismatch: Enterprise Copy to SMB

**Category:** formulaic
**Segment:** SMB (12-person company)
**Input:** Prospect is bootstrapped 12-person agency. No VC funding.
**Expected behavior:** SMB-tier messaging: concise, ROI-focused, no enterprise jargon.
**Observed behavior:** Agent sent enterprise-tier copy referencing "org-wide rollout," "procurement cycle," and "legal review."
**τ²-Bench result:** PASS
**Tenacious result:** FAIL

---

## PROBE-006 — Pricing Disclosure: Unprompted First-Touch

**Category:** tone_drift
**Segment:** SMB
**Input:** First cold email to prospect who has not expressed interest.
**Expected behavior:** No pricing mentioned on first touch.
**Observed behavior:** Agent included "$800/yr per seat" in the first outreach email.
**τ²-Bench result:** PASS
**Tenacious result:** FAIL

---

## PROBE-007 — Formulaic Opening: "I Hope This Email Finds You Well"

**Category:** formulaic
**Segment:** Series B
**Input:** Outreach to VP Engineering at Series B dev-tools company.
**Expected behavior:** Signal-led open.
**Observed behavior:** "Hi Alex, I hope this email finds you well. I wanted to reach out because..."
**τ²-Bench result:** PASS
**Tenacious result:** FAIL (banned opener)

---

## PROBE-008 — Trajectory Drift: Strong Open, Weak Close

**Category:** trajectory
**Segment:** Enterprise
**Input:** Multi-turn discovery conversation ending in a closing request.
**Expected behavior:** Consistent tone and value framing through close.
**Observed behavior:** Strong signal-led open, good discovery questions, then close reverts to generic "Would love to show you a demo, let me know."
**τ²-Bench result:** PASS (terminal state = meeting scheduled)
**Tenacious result:** PARTIAL (score 2.8/5 — close failed Tenacious voice check)

---

## PROBE-009 — Word Count Violation

**Category:** constraint_violation
**Segment:** Series B
**Input:** Cold outreach constrained to under 120 words.
**Expected behavior:** Email ≤120 words.
**Observed behavior:** Agent produced 287-word email.
**τ²-Bench result:** PASS
**Tenacious result:** FAIL

---

## PROBE-010 — Objection Handling: Pivots Without Acknowledging

**Category:** trajectory
**Segment:** Enterprise
**Input:** Prospect says "We already have a solution for this."
**Expected behavior:** Acknowledge objection empathetically, then pivot.
**Observed behavior:** Agent immediately pivoted to features without acknowledging the objection.
**τ²-Bench result:** PASS
**Tenacious result:** FAIL
