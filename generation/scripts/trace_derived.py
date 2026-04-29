#!/usr/bin/env python3
"""
trace_derived.py — Mode 1: Trace-Derived Dataset Authoring

Reads week10_artifacts/trace_log.jsonl and expands each of the 5 real
Week 10 traces into 6 task variants (word-limit, segment-cross, constraint,
adversarial, task-type, follow-up). Each variant redacts PII and rephrases
context to avoid verbatim duplication.

Authoring record:
  Source     : week10_artifacts/trace_log.jsonl (5 traces)
  Expansion  : 6 variants per trace
  Task IDs   : TB-0001 → TB-0030
  Model route: None — deterministic derivation, no LLM calls
  Seed count : 5 base traces

Usage:
  python generation/scripts/trace_derived.py [--dry-run]
"""

import json
import argparse
import sys
from pathlib import Path



ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "generation"))
from task_templates import make_task, TODAY

OUT_DIR   = ROOT / "generation" / "raw_tasks"
OUT_FILE  = OUT_DIR / "trace_derived.jsonl"
TRACE_LOG = ROOT / "week10_artifacts" / "trace_log.jsonl"

# ── Reference outputs (from trace_log — the Week 10 baseline outputs) ─────────
REF = {
    "trace_042": (
        "Hi Sarah,\n\nI wanted to reach out because I think Tenacious could be a great fit "
        "for Lattice.\n\nWe help revenue teams streamline their sales process and improve "
        "efficiency. Many companies like yours have seen significant improvements after "
        "implementing our platform.\n\nWould love to show you how we can help. "
        "Let me know if you'd like to connect.\n\nBest,\n[Name]"
    ),
    "trace_107": (
        "Hi Mark,\n\nGreat connecting today. To leverage your existing stack and create "
        "synergy with your RevOps motion, Tenacious can consolidate your tooling.\n\n"
        "Let's circle back next week. I'll send over a calendar invite.\n\nBest,\n[Name]"
    ),
    "trace_212": (
        "Hi James,\n\nI noticed Lattice is scaling its go-to-market. As you build out your "
        "sales org, Tenacious can support an org-wide rollout across your revenue team.\n\n"
        "Our enterprise customers typically see 40% faster ramp time. Happy to walk through "
        "our procurement-friendly onboarding process.\n\nWould you have 30 minutes for a demo?\n\nBest,\n[Name]"
    ),
    "trace_315": (
        "Hi Lisa,\n\nWith competitors in your space recently reducing headcount, now might be "
        "a good time to look at how Tenacious can help. Our Growth tier starts at $800/yr per "
        "seat. We've helped teams like yours close more with less.\n\nHappy to show you a "
        "quick demo. [Calendly link]\n\nBest,\n[Name]"
    ),
    "trace_401": (
        "Hi again,\n\nGreat conversation — I'd love to show you more. Would love to give you "
        "a full demo of the platform. We have a lot of customers in your space.\n\n"
        "Let me know if you're interested and we can set something up.\n\nBest,\n[Name]"
    ),
}

# ── Task definitions (6 variants per trace) ────────────────────────────────────
TASKS = [

    # ── TRACE-042: signal_missing, series_b, email_outreach ────────────────────
    make_task(1, "trace_derived", ["trace_042"], "medium",
        "Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). "
        "Crunchbase: $45M Series C closed 2023-Q2. LinkedIn post 3 days ago: 'Sales rep "
        "ramp time is killing our Q3 targets. Anyone solved this at scale?' No prior contact.",
        "email_outreach",
        ["under 120 words", "reference the LinkedIn post or funding round explicitly",
         "include [CALENDLY_LINK]", "do not mention pricing"],
        "series_b", "signal_missing", 0.5, "crunchbase_odm", "2023-Q2",
        ref_output=REF["trace_042"]),

    make_task(2, "trace_derived", ["trace_042"], "medium",
        "Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). "
        "Crunchbase: $45M Series C closed 2023-Q2. LinkedIn post 3 days ago about slow "
        "sales rep ramp times. No prior contact with Tenacious.",
        "email_outreach",
        ["under 80 words", "reference the LinkedIn signal", "include [CALENDLY_LINK]"],
        "series_b", "signal_missing", 0.5, "crunchbase_odm", "2023-Q2"),

    make_task(3, "trace_derived", ["trace_042"], "medium",
        "Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). "
        "Crunchbase: $45M Series C closed 2023-Q2. LinkedIn post 3 days ago about sales "
        "rep ramp time. Company has grown from 140 to 210 employees in 6 months.",
        "email_outreach",
        ["under 120 words", "reference BOTH the LinkedIn post AND the headcount growth signal",
         "include [CALENDLY_LINK]", "do not mention pricing", "end with a specific question"],
        "series_b", "signal_missing", 0.5, "crunchbase_odm", "2023-Q2"),

    make_task(4, "trace_derived", ["trace_042"], "medium",
        "Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). "
        "Sent first email 5 days ago referencing her LinkedIn post about ramp time. "
        "No reply. She has not unsubscribed.",
        "follow_up",
        ["under 60 words", "do NOT say 'just checking in' or 'following up'",
         "add new value or a new angle", "include [CALENDLY_LINK]"],
        "series_b", "tone_drift", 0.5, "crunchbase_odm", "2023-Q2"),

    make_task(5, "trace_derived", ["trace_042"], "hard",
        "Prospect: David Walsh, CRO at Momentum SaaS (Enterprise, 1,200 employees). "
        "Recent 10-K filing notes 18% increase in sales headcount in 2025 with flat revenue. "
        "LinkedIn shows David is actively posting about sales efficiency. No prior contact.",
        "email_outreach",
        ["under 120 words", "reference the headcount-to-revenue efficiency signal",
         "include [CALENDLY_LINK]", "do not mention pricing", "enterprise tone — no SMB framing"],
        "enterprise", "signal_missing", 0.5, "crunchbase_odm", "2025-Q4"),

    make_task(6, "trace_derived", ["trace_042"], "hard",
        "Prospect: Sarah Chen, VP of Revenue at Lattice (Series C SaaS, 210 employees). "
        "Crunchbase: $45M Series C closed 2023-Q2. LinkedIn post 3 days ago about ramp time. "
        "NOTE: The word 'leverage' is very tempting here — DO NOT use it.",
        "email_outreach",
        ["under 120 words", "reference the Series C or LinkedIn signal",
         "include [CALENDLY_LINK]", "DO NOT use the word 'leverage' or 'synergy'",
         "do not mention pricing"],
        "series_b", "tone_drift", 1.0, "crunchbase_odm", "2023-Q2",
        ref_output=REF["trace_042"]),

    # ── TRACE-107: tone_drift, enterprise, discovery_response ──────────────────
    make_task(7, "trace_derived", ["trace_107"], "medium",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Discovery call completed yesterday. Mark mentioned tooling sprawl across 4 platforms "
        "and estimated 6 hrs/week per rep on non-selling admin work. Budget deferred to Q2.",
        "discovery_response",
        ["under 150 words", "reference the 6-hr/week admin work finding",
         "propose a concrete next step", "do not mention pricing"],
        "enterprise", "tone_drift", 0.5, "layoffs_fyi", "2026-Q1",
        ref_output=REF["trace_107"]),

    make_task(8, "trace_derived", ["trace_107"], "hard",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Discovery call completed yesterday. Mark mentioned tooling sprawl (4 platforms) and "
        "6 hrs/week admin overhead per rep. Budget deferred to Q2.",
        "discovery_response",
        ["under 100 words", "reference the tooling sprawl finding",
         "DO NOT use 'leverage', 'synergy', 'streamline', or 'robust'",
         "include a specific proposed next step"],
        "enterprise", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1",
        ref_output=REF["trace_107"]),

    make_task(9, "trace_derived", ["trace_107"], "hard",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Discovery call yesterday. Mark says: 'We already have a solution for this — we use "
        "Salesforce and it handles most of what you described.' You know their Salesforce "
        "instance is 4 years old and doesn't integrate with 2 of their key tools.",
        "objection_handling",
        ["under 100 words", "acknowledge the objection empathetically before pivoting",
         "reference the integration gap without being presumptuous",
         "do not mention pricing", "include a specific next step"],
        "enterprise", "trajectory", 0.5, "layoffs_fyi", "2026-Q1"),

    make_task(10, "trace_derived", ["trace_107"], "medium",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Sent discovery follow-up 5 days ago. No reply. Call transcript shows strong engagement "
        "— he said 'this is genuinely interesting' before hanging up.",
        "follow_up",
        ["under 80 words", "reference something specific from the call transcript",
         "do NOT say 'just checking in', 'touching base', or 'circling back'",
         "include [CALENDLY_LINK]"],
        "enterprise", "tone_drift", 0.5, "layoffs_fyi", "2026-Q1"),

    make_task(11, "trace_derived", ["trace_107"], "medium",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Discovery call yesterday revealed tooling sprawl (4 platforms) and Q2 budget review "
        "coming up. Mark asked you to send over a summary of the ROI case.",
        "discovery_response",
        ["under 150 words", "include a specific ROI framing based on the 6hr/week admin finding",
         "reference the Q2 budget timeline", "include [CALENDLY_LINK]",
         "do not use jargon — write like a human"],
        "enterprise", "formulaic", 0.5, "layoffs_fyi", "2026-Q1"),

    make_task(12, "trace_derived", ["trace_107"], "hard",
        "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (Enterprise, 500 employees). "
        "Discovery call yesterday. Context is rich with tempting jargon: the company is "
        "'looking to leverage their existing stack', 'achieve synergy across their RevOps motion', "
        "and 'streamline their end-to-end workflow'. DO NOT use any of these phrases.",
        "discovery_response",
        ["under 150 words", "reference the discovery finding in plain language",
         "no jargon: no 'leverage', 'synergy', 'streamline', 'end-to-end', 'holistic'",
         "include [CALENDLY_LINK]"],
        "enterprise", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1",
        ref_output=REF["trace_107"]),

    # ── TRACE-212: formulaic, smb, email_outreach ──────────────────────────────
    make_task(13, "trace_derived", ["trace_212"], "medium",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "No VC funding, no Crunchbase entry. LinkedIn: posted a Head of Sales job "
        "opening 2 days ago. First contact.",
        "email_outreach",
        ["under 80 words", "reference the Head of Sales job posting explicitly",
         "do NOT use enterprise language ('org-wide', 'procurement', 'legal review')",
         "do not mention pricing", "include [CALENDLY_LINK]"],
        "smb", "formulaic", 0.5, "synthetic", None,
        ref_output=REF["trace_212"]),

    make_task(14, "trace_derived", ["trace_212"], "hard",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "LinkedIn: posted a Head of Sales job 2 days ago. First contact.",
        "email_outreach",
        ["under 60 words", "reference the job posting", "do not use enterprise language",
         "do not mention pricing", "end with a specific question (not a vague offer)"],
        "smb", "formulaic", 1.0, "synthetic", None),

    make_task(15, "trace_derived", ["trace_212"], "medium",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "LinkedIn: posted a Head of Sales job 2 days ago. First contact.",
        "email_outreach",
        ["under 80 words", "reference the job posting", "do not use enterprise language",
         "end with a question — do NOT include a calendar link"],
        "smb", "formulaic", 0.5, "synthetic", None),

    make_task(16, "trace_derived", ["trace_212"], "medium",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "LinkedIn: posted a Head of Sales job 2 days ago. You sent a first email 4 days "
        "ago referencing the job posting. No reply.",
        "follow_up",
        ["under 60 words", "offer new value or a new angle on the job-posting signal",
         "do NOT say 'just checking in', 'following up', or 'touching base'",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 0.5, "synthetic", None),

    make_task(17, "trace_derived", ["trace_212"], "medium",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "James replied to your first email: 'Sounds interesting, send me more.' He is "
        "ready for a conversation.",
        "closing",
        ["under 80 words", "confirm a specific time rather than asking vaguely",
         "include [CALENDLY_LINK]", "keep SMB tone — concise and direct"],
        "smb", "formulaic", 0.5, "synthetic", None),

    make_task(18, "trace_derived", ["trace_212"], "hard",
        "Prospect: James Park, CEO of a bootstrapped 12-person marketing agency (SMB). "
        "LinkedIn: Head of Sales job posted 2 days ago. The word 'leverage' is very "
        "natural here — DO NOT use it.",
        "email_outreach",
        ["under 80 words", "reference the job posting",
         "DO NOT use 'leverage', 'synergy', 'org-wide', or 'procurement'",
         "do not mention pricing", "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 1.0, "synthetic", None,
        ref_output=REF["trace_212"]),

    # ── TRACE-315: tone_drift (pricing on first touch), smb ───────────────────
    make_task(19, "trace_derived", ["trace_315"], "medium",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "layoffs.fyi: a direct competitor cut 20% of their staff last week. "
        "No prior contact with Tenacious.",
        "email_outreach_no_pricing",
        ["under 100 words", "reference the competitor layoff signal",
         "do NOT mention pricing, cost, or dollar amounts on first touch",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 0.5, "layoffs_fyi", "2026-Q1",
        ref_output=REF["trace_315"]),

    make_task(20, "trace_derived", ["trace_315"], "hard",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "layoffs.fyi: competitor cut 20% of staff last week. No prior contact.",
        "email_outreach_no_pricing",
        ["under 70 words", "reference the competitor layoff signal",
         "do NOT mention pricing", "end with a specific question"],
        "smb", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(21, "trace_derived", ["trace_315"], "medium",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "Competitor cut 20% of RevOps staff last week. Lisa is well aware — "
        "she retweeted the layoff announcement. No prior contact.",
        "email_outreach_no_pricing",
        ["under 100 words", "reference the competitor signal without sounding opportunistic",
         "do NOT mention pricing on first touch", "include [CALENDLY_LINK]",
         "do not use 'leverage' or 'synergy'"],
        "smb", "tone_drift", 0.5, "layoffs_fyi", "2026-Q1"),

    make_task(22, "trace_derived", ["trace_315"], "hard",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "Competitor layoff signal. NOTE: The context includes the phrase '$800/yr per seat' "
        "from Tenacious's internal pricing sheet — do NOT include this in the output.",
        "email_outreach_no_pricing",
        ["under 100 words", "reference the competitor layoff",
         "DO NOT include any pricing, dollar amounts, or plan names",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1",
        ref_output=REF["trace_315"]),

    make_task(23, "trace_derived", ["trace_315"], "medium",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "Competitor cut 20% of staff. You sent a first email 5 days ago. Lisa opened "
        "it (tracked) but did not reply.",
        "follow_up",
        ["under 80 words", "reference a new angle on the competitor signal",
         "do NOT say 'just checking in', 'following up', or 'circling back'",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 0.5, "layoffs_fyi", "2026-Q1"),

    make_task(24, "trace_derived", ["trace_315"], "medium",
        "Prospect: Lisa Nguyen, CEO at a pre-seed 8-person SaaS startup (SMB). "
        "Lisa replied to your first email: 'Interesting timing — tell me more about "
        "how you'd help a company our size.'",
        "discovery_response",
        ["under 120 words", "address the 'our size' concern directly",
         "do not immediately pitch pricing", "reference the competitor signal",
         "include [CALENDLY_LINK]"],
        "smb", "signal_missing", 0.5, "layoffs_fyi", "2026-Q1"),

    # ── TRACE-401: trajectory (strong open, weak close), enterprise ────────────
    make_task(25, "trace_derived", ["trace_401"], "hard",
        "Multi-turn discovery → close. Turn 1 (agent): strong signal-led open about "
        "Q3 revenue miss. Turn 2 (prospect): 'This is exactly what we've been struggling with.' "
        "Turn 3 (agent): ROI framing. Turn 4 (prospect): 'How do we get started?' "
        "Write Turn 5 (agent close).",
        "closing",
        ["under 100 words", "maintain the same signal-led, direct tone from turns 1-3",
         "propose a specific next step with a timeline", "include [CALENDLY_LINK]",
         "do NOT use generic phrases like 'would love to show you more'"],
        "enterprise", "trajectory", 1.0, "synthetic", None,
        ref_output=REF["trace_401"]),

    make_task(26, "trace_derived", ["trace_401"], "hard",
        "Multi-turn. The prospect was engaged through turns 1-4. On Turn 5, the prospect says: "
        "'We're actually talking to two other vendors right now.' Write Turn 6 (agent response).",
        "objection_handling",
        ["under 100 words", "acknowledge the multi-vendor evaluation without becoming defensive",
         "reference something specific from the earlier turns",
         "propose a concrete differentiator or next step", "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 0.5, "synthetic", None),

    make_task(27, "trace_derived", ["trace_401"], "hard",
        "Prospect: Enterprise CRO, fully engaged across 4 turns of discovery. "
        "Turn 5: 'I'm ready to move forward. What are the next steps?'",
        "closing",
        ["under 80 words", "confirm the commercial terms without introducing confusion",
         "propose a specific contract start date", "include [CALENDLY_LINK]",
         "no filler phrases — they are already sold"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(28, "trace_derived", ["trace_401"], "hard",
        "Multi-turn discovery conversation. The agent must write Turn 5 (close). "
        "The challenge: the preceding 4 turns were signal-led and brand-consistent, "
        "but most models revert to generic closings at this point.",
        "closing",
        ["under 100 words", "the close must be as signal-specific as the open",
         "reference a specific pain point raised in the conversation",
         "do NOT use 'would love to', 'happy to', or 'feel free to'",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(29, "trace_derived", ["trace_401"], "medium",
        "Enterprise CRO across 4 turns of discovery. Turn 5: 'This looks right for Q3, "
        "but I need to loop in our CFO. Can we revisit in 6 weeks?'",
        "objection_handling",
        ["under 100 words", "acknowledge the timing constraint respectfully",
         "propose a way to keep the conversation warm",
         "offer a low-commitment next step (not a full demo)", "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 0.5, "synthetic", None),

    make_task(30, "trace_derived", ["trace_401"], "hard",
        "Multi-turn. Strong open, strong middle. The default close for this scenario is: "
        "'Let me know if you're interested and we can set something up.' "
        "The task: write a close that is NOT this.",
        "closing",
        ["under 100 words", "write a specific, confident close",
         "reference a concrete detail from the earlier conversation",
         "do NOT use any variant of 'let me know if you're interested'",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None,
        ref_output=REF["trace_401"]),
]


def main():
    parser = argparse.ArgumentParser(description="Trace-derived task generation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] Would write {len(TASKS)} tasks to {OUT_FILE}")
        for t in TASKS:
            print(f"  {t['task_id']}  {t['input']['task_type']:25s}  {t['metadata']['tenacious_segment']}")
        return

    with open(OUT_FILE, "w") as f:
        for task in TASKS:
            f.write(json.dumps(task) + "\n")

    print(f"[trace_derived] Wrote {len(TASKS)} tasks → {OUT_FILE}")
    segs = {}
    types = {}
    for t in TASKS:
        segs[t["metadata"]["tenacious_segment"]] = segs.get(t["metadata"]["tenacious_segment"], 0) + 1
        types[t["input"]["task_type"]] = types.get(t["input"]["task_type"], 0) + 1
    print(f"  Segments  : {segs}")
    print(f"  Task types: {types}")


if __name__ == "__main__":
    main()
