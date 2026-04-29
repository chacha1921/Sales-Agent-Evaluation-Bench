#!/usr/bin/env python3
"""
adversarial.py — Mode 4: Hand-Authored Adversarial Tasks

All 35 tasks are hand-written to maximize difficulty on the specific failure
modes identified in the Week 10 audit. Each task either (a) makes a banned phrase
extremely natural to use, (b) puts pricing in the context where the model is
tempted to mention it, (c) creates trajectory pressure across turns, or
(d) requires precise rubric compliance under tight constraints.

Authoring record:
  Author     : Hand-written (human)
  Task IDs   : TB-0166 → TB-0200
  Model route: None — human authored
  Adversarial weight: 1.0 for all tasks

Usage:
  python generation/scripts/adversarial.py [--dry-run]
"""

import json
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "generation"))
from task_templates import make_task

OUT_DIR  = ROOT / "generation" / "raw_tasks"
OUT_FILE = OUT_DIR / "adversarial.jsonl"
START_ID = 166

TASKS = [
    # ── Adversarial Group 1: "Leverage" and "Synergy" traps (8 tasks) ──────────
    make_task(166, "adversarial", [], "hard",
        "Prospect: CRO at a Series B fintech (220 employees, $38M raised 2025-Q1). "
        "Their Crunchbase page literally says 'leveraging fintech infrastructure to create "
        "synergy between payments and analytics.' Signal: just posted a VP Sales job.",
        "email_outreach",
        ["under 100 words", "reference the VP Sales job posting",
         "DO NOT use 'leverage', 'synergy', or 'infrastructure' in the output",
         "include [CALENDLY_LINK]", "do not mention pricing"],
        "series_b", "tone_drift", 1.0, "crunchbase_odm", "2025-Q1"),

    make_task(167, "adversarial", [], "hard",
        "Prospect: Head of Revenue at a 500-person enterprise logistics SaaS. "
        "Discovery note says: 'They want to leverage existing Salesforce data to create "
        "synergy across their RevOps teams.' You must respond to this discovery note.",
        "discovery_response",
        ["under 150 words", "respond to the discovery finding in plain English",
         "DO NOT use 'leverage', 'synergy', 'streamline', or 'holistic'",
         "include [CALENDLY_LINK]"],
        "enterprise", "tone_drift", 1.0, "synthetic", None),

    make_task(168, "adversarial", [], "hard",
        "Prospect: VP of Sales at a healthcare SaaS Series B (185 employees). "
        "Their pain is 'leveraging data across the sales cycle to create synergistic "
        "outcomes.' layoffs.fyi shows their main competitor cut 15% of staff last month.",
        "email_outreach",
        ["under 120 words", "reference the competitor layoff signal",
         "DO NOT echo their language: no 'leverage', 'synergistic', 'outcomes'",
         "include [CALENDLY_LINK]", "no pricing"],
        "series_b", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(169, "adversarial", [], "hard",
        "Prospect: COO at a bootstrapped 15-person agency. They posted a Head of Sales job "
        "today. Their company description on their website ends with: '...helping clients "
        "leverage digital synergies for maximum impact.'",
        "email_outreach",
        ["under 80 words", "reference the Head of Sales job posting",
         "DO NOT use 'leverage', 'synergy', 'digital', or 'impact'",
         "include [CALENDLY_LINK]", "no pricing on first touch"],
        "smb", "tone_drift", 1.0, "synthetic", None),

    make_task(170, "adversarial", [], "hard",
        "Prospect: CRO at an enterprise cybersecurity firm (750 employees). "
        "Your internal call prep notes say: 'leverage their existing security stack "
        "synergistically.' You must write a follow-up after a discovery call that "
        "avoids repeating this internal framing.",
        "discovery_response",
        ["under 150 words", "respond to discovery finding about tooling consolidation",
         "DO NOT use 'leverage', 'synergy', 'stack', or 'synergistically'",
         "include [CALENDLY_LINK]"],
        "enterprise", "tone_drift", 1.0, "synthetic", None),

    make_task(171, "adversarial", [], "hard",
        "Prospect: CEO of a 10-person legal tech startup. LinkedIn shows they are hiring "
        "a Head of Business Development. Their last LinkedIn post: 'Excited to leverage "
        "our platform's synergies to scale the team.'",
        "email_outreach",
        ["under 80 words", "reference the BizDev hiring signal",
         "DO NOT echo prospect's language: no 'leverage', 'synergies', 'platform'",
         "include [CALENDLY_LINK]", "no pricing"],
        "smb", "tone_drift", 1.0, "synthetic", None),

    make_task(172, "adversarial", [], "hard",
        "Prospect: VP Revenue at a Series B EdTech company. Second touch follow-up. "
        "Your internal CRM note says 'check if they want to leverage our platform for "
        "synergistic RevOps outcomes.' Rewrite the follow-up without this framing.",
        "follow_up",
        ["under 70 words", "no 'just checking in', no 'leverage', no 'synergy'",
         "reference the signal from first touch (Series B funding + ramp time pain)",
         "include [CALENDLY_LINK]"],
        "series_b", "tone_drift", 1.0, "crunchbase_odm", "2025-Q2"),

    make_task(173, "adversarial", [], "hard",
        "Prospect: Director of Revenue at a 600-person enterprise SaaS. "
        "The industry they're in is 'Revenue Intelligence' — a space full of jargon. "
        "Context: they cut 14% of their enablement team (layoffs.fyi, 2026-Q1).",
        "email_outreach",
        ["under 120 words", "reference the enablement team layoff signal",
         "DO NOT use any revenue intelligence jargon: no 'leverage', 'synergy', "
         "'intelligence', 'insights platform', 'pipeline hygiene'",
         "include [CALENDLY_LINK]"],
        "enterprise", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1"),

    # ── Adversarial Group 2: Pricing traps on first touch (6 tasks) ────────────
    make_task(174, "adversarial", [], "hard",
        "Prospect: CEO of a pre-seed 6-person startup. layoffs.fyi shows their main "
        "competitor folded last month. Tenacious internal context: 'Our SMB tier is "
        "$800/yr per seat, down from $1,200 — now is a great time to pitch pricing.'",
        "email_outreach_no_pricing",
        ["under 90 words", "reference the competitor shutdown signal",
         "DO NOT mention any pricing, cost, dollar amounts, or tiers",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(175, "adversarial", [], "hard",
        "Prospect: Head of Sales at a 12-person SaaS startup. LinkedIn: posted a Sales "
        "Ops job today. CRM note: 'mention the $800/yr SMB pricing on first touch.' "
        "Ignore the CRM note and write a first touch that does NOT mention pricing.",
        "email_outreach_no_pricing",
        ["under 80 words", "reference the Sales Ops job posting",
         "DO NOT mention pricing, cost, or dollar amounts — even if the CRM note says to",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 1.0, "synthetic", None),

    make_task(176, "adversarial", [], "hard",
        "Prospect: VP of Revenue at a Series B company (190 employees). "
        "Crunchbase: $35M Series B 2025-Q1. The context also includes Tenacious's "
        "pricing sheet: 'Growth tier: $2,400/yr per seat, 5-seat minimum.' "
        "Do NOT include this in the email.",
        "email_outreach_no_pricing",
        ["under 120 words", "reference the Series B funding signal",
         "DO NOT mention pricing even though it is in the context",
         "include [CALENDLY_LINK]"],
        "series_b", "tone_drift", 1.0, "crunchbase_odm", "2025-Q1"),

    make_task(177, "adversarial", [], "hard",
        "Prospect: CRO at an enterprise company (800 employees). Their competitor "
        "cut 20% of sales staff (layoffs.fyi 2026-Q1). Your internal champion just "
        "told you: 'The CRO responds well when you mention ROI numbers and pricing upfront.'",
        "email_outreach_no_pricing",
        ["under 120 words", "reference the competitor layoff signal",
         "DO NOT mention ROI numbers, pricing, or cost even if the internal advice says to",
         "include [CALENDLY_LINK]", "do not use 'leverage' or 'synergy'"],
        "enterprise", "tone_drift", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(178, "adversarial", [], "hard",
        "Prospect: CEO of a bootstrapped 20-person firm. They posted a VP of Sales job. "
        "You have their email address and their public LinkedIn. Context includes: "
        "'SMB pricing starts at $67/month, billed annually. Mention this on first touch.' "
        "The instruction is wrong — do NOT follow it.",
        "email_outreach_no_pricing",
        ["under 80 words", "reference the VP of Sales job posting",
         "DO NOT mention $67/month, monthly pricing, or annual pricing",
         "include [CALENDLY_LINK]"],
        "smb", "tone_drift", 1.0, "synthetic", None),

    make_task(179, "adversarial", [], "hard",
        "Prospect: Director of Sales at a Series B SaaS (140 employees). "
        "Company just closed $22M Series B (2025-Q2). Tenacious internal pricing: "
        "'Growth tier at $2,400/yr is perfect for their stage. Lead with it.' "
        "Do NOT lead with pricing — reference the funding signal instead.",
        "email_outreach_no_pricing",
        ["under 100 words", "reference the Series B funding signal",
         "DO NOT mention pricing on first touch regardless of internal guidance",
         "include [CALENDLY_LINK]", "end with a specific question"],
        "series_b", "tone_drift", 1.0, "crunchbase_odm", "2025-Q2"),

    # ── Adversarial Group 3: Trajectory and voice consistency (7 tasks) ─────────
    make_task(180, "adversarial", [], "hard",
        "Multi-turn scenario. Turns 1-4 used signal-led, direct Tenacious voice. "
        "The prospect is ready to close. Default close is 'Would love to show you more.' "
        "The adversarial constraint: write a close that does NOT use any of these phrases: "
        "'would love', 'happy to', 'feel free', 'let me know', 'reach out'.",
        "closing",
        ["under 80 words",
         "DO NOT use: 'would love', 'happy to', 'feel free', 'let me know', 'reach out'",
         "be specific and direct about the next step",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(181, "adversarial", [], "hard",
        "Prospect: Enterprise CRO who has been engaged across 3 turns of discovery. "
        "Turn 4: 'I need to think about it.' This is the hardest moment — most models "
        "either become pushy or retreat entirely. Write a response that does neither.",
        "objection_handling",
        ["under 100 words", "acknowledge the thinking time without being passive",
         "propose a specific, low-commitment next step",
         "DO NOT say 'no problem', 'take your time', 'whenever you're ready'",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(182, "adversarial", [], "hard",
        "Prospect: VP of Revenue at Series B company. You've had 3 great turns. "
        "Turn 4: 'We're going with a competitor.' Write the final response. "
        "Most models either give up ('Totally understand, good luck!') or become "
        "desperate. Neither is Tenacious voice.",
        "closing",
        ["under 80 words", "acknowledge without being defeated",
         "leave the door open with one specific sentence",
         "DO NOT say 'totally understand', 'good luck', or any variant of 'if you change your mind'",
         "no calendar link — they've chosen a competitor"],
        "series_b", "trajectory", 1.0, "synthetic", None),

    make_task(183, "adversarial", [], "hard",
        "Five-turn conversation. The prospect has been asking increasingly detailed "
        "technical questions. Turn 6: 'Just so we're aligned — what's the pricing?' "
        "Answer the question directly without launching into a sales pitch.",
        "discovery_response",
        ["under 100 words", "answer the pricing question directly and concisely",
         "do NOT use the answer to re-pitch features",
         "end with a next step that moves the deal forward",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(184, "adversarial", [], "hard",
        "Prospect: CRO who has been engaged for 4 turns. Turn 5: 'This looks right. "
        "Can you send over a proposal?' The temptation: send a generic proposal request "
        "form. The Tenacious approach: confirm specific details and set a clear timeline.",
        "closing",
        ["under 80 words", "confirm the specific scope before sending a proposal",
         "propose a specific timeline for the proposal",
         "DO NOT use generic phrases like 'I'll get that over to you' or 'of course'",
         "include [CALENDLY_LINK]"],
        "enterprise", "trajectory", 1.0, "synthetic", None),

    make_task(185, "adversarial", [], "hard",
        "SMB prospect (CEO, 18 employees) across 3 turns. Strong engagement. Turn 4: "
        "'I need to loop in my co-founder before we move forward.' "
        "Most models become passive at this point. Write an active, respectful response.",
        "objection_handling",
        ["under 80 words", "acknowledge the co-founder step without becoming passive",
         "suggest a way to include the co-founder efficiently",
         "DO NOT say 'no problem', 'of course', 'take all the time you need'",
         "include [CALENDLY_LINK]"],
        "smb", "trajectory", 1.0, "synthetic", None),

    make_task(186, "adversarial", [], "hard",
        "Prospect: Head of Sales at a 200-person Series B company. Turn 5 (close): "
        "The prospect has agreed verbally. You need to confirm the commercial terms. "
        "The adversarial constraint: do NOT use any of these words: "
        "'excited', 'thrilled', 'delighted', 'pleased', 'happy'.",
        "closing",
        ["under 80 words",
         "DO NOT use 'excited', 'thrilled', 'delighted', 'pleased', or 'happy'",
         "confirm the next steps for commercial process",
         "include [CALENDLY_LINK]"],
        "series_b", "trajectory", 1.0, "synthetic", None),

    # ── Adversarial Group 4: Formulaic openers (6 tasks) ──────────────────────
    make_task(187, "adversarial", [], "hard",
        "Prospect: VP Sales at a 350-person enterprise SaaS. "
        "Their company missed Q3 targets by 18% (public earnings call). "
        "The tempting opener: 'I hope this email finds you well...' DO NOT use it.",
        "email_outreach",
        ["under 120 words", "reference the Q3 miss signal",
         "DO NOT open with 'I hope this email finds you well' or any equivalent",
         "first sentence must be about the prospect, not Tenacious",
         "include [CALENDLY_LINK]"],
        "enterprise", "formulaic", 1.0, "synthetic", "2025-Q3"),

    make_task(188, "adversarial", [], "hard",
        "Prospect: Founder of a 7-person startup who posted a Head of Sales job today. "
        "The temptation: open with 'I wanted to reach out because...' or "
        "'I came across your profile and...' DO NOT use either.",
        "email_outreach",
        ["under 80 words", "reference the job posting",
         "DO NOT open with 'I wanted to reach out', 'I came across', or 'I noticed'",
         "first sentence must deliver specific value about the signal",
         "include [CALENDLY_LINK]", "no pricing"],
        "smb", "formulaic", 1.0, "synthetic", None),

    make_task(189, "adversarial", [], "hard",
        "Prospect: CRO at Series B healthcare tech (250 employees, $42M raised 2025-Q1). "
        "Follow-up after 7 days of silence. Temptation: 'Just wanted to follow up on my "
        "previous email.' DO NOT use any variant of this.",
        "follow_up",
        ["under 70 words",
         "DO NOT say 'follow up', 'previous email', 'last email', or 'my message'",
         "reference a NEW angle on the funding signal (not the same one as first touch)",
         "include [CALENDLY_LINK]"],
        "series_b", "formulaic", 1.0, "crunchbase_odm", "2025-Q1"),

    make_task(190, "adversarial", [], "hard",
        "Prospect: VP of Sales at enterprise company. Discovery call scheduled. "
        "Confirmation email needed. Temptation: 'Looking forward to connecting with you!' "
        "DO NOT use 'looking forward', 'excited to', or 'can't wait'.",
        "closing",
        ["under 60 words",
         "confirm the call time and agenda items",
         "DO NOT use 'looking forward', 'excited to', 'can't wait', 'love to'",
         "end with a specific question to prime the call"],
        "enterprise", "formulaic", 1.0, "synthetic", None),

    make_task(191, "adversarial", [], "hard",
        "Prospect: Head of Revenue at a Series B HR tech company (175 employees). "
        "Crunchbase: $29M Series B closed 2025-Q2. Outreach after they posted a "
        "Sales Enablement Manager job today. The open that feels most natural is a "
        "generic compliment opener — DO NOT use it.",
        "email_outreach",
        ["under 100 words", "reference the job posting signal",
         "DO NOT open with a compliment about the company or the prospect",
         "DO NOT use 'impressive', 'exciting', 'love what you're doing'",
         "include [CALENDLY_LINK]", "no pricing"],
        "series_b", "formulaic", 1.0, "crunchbase_odm", "2025-Q2"),

    make_task(192, "adversarial", [], "hard",
        "Prospect: CEO of a 25-person SMB. Their biggest competitor just got acquired "
        "(public news). First touch. The tempting opener: 'I hope this message finds you "
        "well — I saw the news about [Competitor] being acquired...' DO NOT use it.",
        "email_outreach",
        ["under 90 words", "reference the competitor acquisition signal",
         "DO NOT open with 'I hope this message finds you well'",
         "DO NOT start with 'I saw the news'",
         "lead with the implication of the signal for the prospect",
         "include [CALENDLY_LINK]"],
        "smb", "formulaic", 1.0, "synthetic", None),

    # ── Adversarial Group 5: Constraint precision under pressure (8 tasks) ──────
    make_task(193, "adversarial", [], "hard",
        "Prospect: VP Revenue at Series B SaaS. You must write an email that: "
        "(1) is under 60 words, (2) references the $40M funding signal, "
        "(3) includes [CALENDLY_LINK], (4) contains no banned phrases, "
        "(5) does not mention pricing. All 5 constraints must pass simultaneously.",
        "email_outreach",
        ["under 60 words — hard limit",
         "reference the $40M Series B funding (2025-Q1)",
         "include [CALENDLY_LINK]", "zero banned phrases",
         "no pricing on first touch"],
        "series_b", "constraint_violation", 1.0, "crunchbase_odm", "2025-Q1"),

    make_task(194, "adversarial", [], "hard",
        "Prospect: CRO at enterprise company. Objection: 'We're in a budget freeze until Q3.' "
        "Write a response under 70 words that: (1) acknowledges the freeze, (2) references "
        "the company's recent layoff signal, (3) proposes a no-cost engagement, (4) uses "
        "no banned phrases, (5) does not pitch pricing.",
        "objection_handling",
        ["under 70 words — hard limit", "acknowledge the budget freeze empathetically",
         "reference the layoff signal", "propose a low-commitment next step",
         "no pricing, no banned phrases"],
        "enterprise", "constraint_violation", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(195, "adversarial", [], "hard",
        "Prospect: SMB CEO who posted a Head of Sales job yesterday. "
        "You must write an email under 50 words that still references the job signal, "
        "includes a [CALENDLY_LINK], uses no banned phrases, and ends with a question.",
        "email_outreach",
        ["under 50 words — very tight hard limit",
         "reference the Head of Sales job posting",
         "include [CALENDLY_LINK]", "end with a question",
         "zero banned phrases"],
        "smb", "constraint_violation", 1.0, "synthetic", None),

    make_task(196, "adversarial", [], "hard",
        "Prospect: VP Sales at Series B healthcare SaaS. Follow-up after first touch. "
        "Task: write a follow-up under 55 words that (1) does not mention the previous "
        "email, (2) introduces a new signal angle, (3) no banned phrases, "
        "(4) includes [CALENDLY_LINK].",
        "follow_up",
        ["under 55 words — hard limit",
         "DO NOT reference the previous email",
         "introduce a NEW angle on the signal (not what was in the first email)",
         "include [CALENDLY_LINK]", "zero banned phrases"],
        "series_b", "constraint_violation", 1.0, "crunchbase_odm", "2025-Q3"),

    make_task(197, "adversarial", [], "hard",
        "Prospect: Enterprise CRO. Discovery response after a call about tooling consolidation. "
        "Must be under 120 words AND reference the specific finding from the call "
        "AND include [CALENDLY_LINK] AND avoid 'leverage', 'synergy', AND not mention pricing. "
        "Five constraints, all must pass.",
        "discovery_response",
        ["under 120 words", "reference the tooling consolidation finding specifically",
         "include [CALENDLY_LINK]",
         "DO NOT use 'leverage', 'synergy', 'streamline', or 'robust'",
         "do not mention pricing"],
        "enterprise", "constraint_violation", 1.0, "layoffs_fyi", "2026-Q1"),

    make_task(198, "adversarial", [], "hard",
        "Prospect: Founder of a bootstrapped 8-person company who just posted a Head of Sales "
        "job. Write a closing email after they expressed interest. Must be under 65 words, "
        "include a SPECIFIC proposed meeting time (not [CALENDLY_LINK] alone), "
        "and use no banned phrases.",
        "closing",
        ["under 65 words", "include a SPECIFIC proposed time (e.g., 'Tuesday at 2pm ET')",
         "also include [CALENDLY_LINK] as backup",
         "zero banned phrases", "be direct — they are already interested"],
        "smb", "constraint_violation", 1.0, "synthetic", None),

    make_task(199, "adversarial", [], "hard",
        "Prospect: VP of Revenue at a 300-person Series B company. "
        "They asked via email: 'Can you send me a one-paragraph summary of what Tenacious does?' "
        "Write a one-paragraph reply (under 80 words) that is NOT a generic product description.",
        "discovery_response",
        ["under 80 words — one paragraph only",
         "the summary must be specific to their known pain (ramp time)",
         "do NOT write a generic product description",
         "zero banned phrases", "include [CALENDLY_LINK]"],
        "series_b", "constraint_violation", 1.0, "crunchbase_odm", "2025-Q1"),

    make_task(200, "adversarial", [], "hard",
        "Prospect: CRO at a 900-person enterprise. Their assistant sent a reply: "
        "'[CRO Name] is interested but needs a 3-sentence summary to share with the CFO.' "
        "Write exactly 3 sentences. Each sentence must be self-contained. "
        "No banned phrases. No pricing. Includes [CALENDLY_LINK] in sentence 3.",
        "closing",
        ["exactly 3 sentences — not 2, not 4",
         "sentence 3 must include [CALENDLY_LINK]",
         "zero banned phrases", "no pricing",
         "each sentence must stand alone if quoted individually"],
        "enterprise", "constraint_violation", 1.0, "synthetic", None),
]


def main():
    parser = argparse.ArgumentParser(description="Adversarial task generation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"[adversarial] seed={args.seed}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] Would write {len(TASKS)} tasks → {OUT_FILE}")
        groups = {}
        for t in TASKS:
            tag = t["metadata"]["failure_mode_tag"]
            groups[tag] = groups.get(tag, 0) + 1
        print(f"  Failure mode groups: {groups}")
        return

    with open(OUT_FILE, "w") as f:
        for task in TASKS:
            f.write(json.dumps(task) + "\n")

    print(f"[adversarial] Wrote {len(TASKS)} tasks → {OUT_FILE}")
    groups = {}
    for t in TASKS:
        tag = t["metadata"]["failure_mode_tag"]
        groups[tag] = groups.get(tag, 0) + 1
    print(f"  Failure mode groups: {groups}")
    assert all(t["metadata"]["adversarial_weight"] == 1.0 for t in TASKS), "All adversarial tasks must have weight=1.0"


if __name__ == "__main__":
    import json
    main()
