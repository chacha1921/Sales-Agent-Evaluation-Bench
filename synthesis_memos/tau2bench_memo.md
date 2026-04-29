# Synthesis Memo: τ²-Bench (Yao et al., 2024)

**Paper:** τ²-Bench: Benchmarking Tool-Use of AI Agents in Real-World Tasks
**Design choice critiqued:** Terminal-state evaluation as the universal success criterion (§3.2, "Evaluation Protocol")

---

## The Design Decision

τ²-Bench defines task success entirely by terminal state: an agent succeeds if and only if the final world-state matches the annotated target (§3.2). The paper explicitly argues that terminal-state evaluation is superior to step-level evaluation because it "avoids the difficulty of specifying correct intermediate actions" and "remains valid regardless of what path the agent took to reach the goal" (§3.2, para. 3). This is not an omission — it is a deliberate design choice, stated as a methodological contribution.

The paper also grounds its task design in e-commerce and retail web workflows: adding items to a cart, completing a checkout flow, booking appointments. This domain selection is presented as enabling "real-world grounding" (§2.1).

---

## Why I Disagree

Terminal-state evaluation is the right choice for state-transformation tasks. For communication-centric tasks, it is the wrong metric entirely — and using it without qualification produces misleading scores.

In B2B sales, there is no unambiguous terminal "state." A meeting booked is one possible success outcome, but the *manner* of the communication determines whether the relationship can be built on. Week 10 trace_401 makes this concrete: the agent executed turns 1–4 cleanly, with signal-led openers and non-pushy follow-ups. On turn 5, it reverted to a generic close — "let me know if you're interested in chatting." The prospect replied and a call was scheduled. τ²-Bench scored this 0.91 (PASS). Tenacious scored it 2.1/5 (PARTIAL FAIL). The meeting was booked, but the final message undermined the relationship before it started.

τ²-Bench's terminal-state criterion cannot detect this failure because communication quality is not represented in the terminal state at all. The "state" after turn 5 is identical whether the agent wrote a confident, specific close or a generic filler — both resulted in a calendar invite.

The same failure mode appeared in PROBE-010 (trace_401): the agent answered an objection by pivoting immediately to a counter-claim, without acknowledging the prospect's concern. The τ²-Bench outcome was identical — a follow-up meeting was still scheduled. From a relationship standpoint, this type of unanswered objection is the primary cause of late-stage deal collapse in B2B sales. Terminal-state metrics simply have no vocabulary for it.

The deeper problem is domain transfer. τ²-Bench's retail grounding is not incidental — it shapes the metric design. Adding an item to a cart has a clear binary terminal state. Booking an introductory call does not predict ARR, retention, or deal health. By treating "meeting scheduled" as equivalent to "task succeeded," τ²-Bench imports the ontology of retail task completion into a context where it does not apply.

---

## What Should Have Been Done Differently

The paper would have been more general if the evaluation protocol had included a brief scope statement: "terminal-state evaluation is appropriate for tasks where task quality is fully encoded in the final state, not for tasks where communication quality is the outcome." Limiting the benchmark's domain claim to tool-use agents in state-transformation workflows — rather than positioning it as a general agent evaluation framework — would have prevented overgeneralization.

The Tenacious-Bench design addresses this directly: rubric dimensions are defined at the message level (signal grounding, tone compliance, constraint satisfaction), and the aggregate score reflects communication quality independent of whether any downstream calendar event occurs.

---

*~450 words | Week 11 evidence: trace_401, PROBE-010, PROBE-008*
