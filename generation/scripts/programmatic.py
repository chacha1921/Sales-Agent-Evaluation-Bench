#!/usr/bin/env python3
"""
programmatic.py — Mode 2: Programmatic / Template-Expansion Dataset Authoring

Generates tasks by combinatorial expansion of 15 prospect profiles × 5 task types.
All tasks are fully deterministic (no LLM calls). Templates ensure structural
diversity across segments, roles, signal types, and task types.

Authoring record:
  Method     : Combinatorial template expansion
  Profiles   : 15 prospect profiles (defined below)
  Task types : 5 (email_outreach, follow_up, discovery_response,
                    objection_handling, closing)
  Task IDs   : TB-0031 → TB-0105
  Model route: None — no LLM calls, fully deterministic
  Seed count : 15 profiles × 5 types = 75 tasks

Usage:
  python generation/scripts/programmatic.py [--dry-run]
"""

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "generation"))
from task_templates import make_task

OUT_DIR  = ROOT / "generation" / "raw_tasks"
OUT_FILE = OUT_DIR / "programmatic.jsonl"
START_ID = 31

# ── Prospect profiles ──────────────────────────────────────────────────────────
PROFILES = [
    dict(name="Alex Thompson",  role="VP of Revenue",             company="FinEdge",      segment="series_b",  employees=180,
         signal="closed $32M Series B in 2024-Q3",                signal_type="funding_round",
         signal_src="crunchbase_odm", signal_win="2024-Q3",       pain="inconsistent pipeline visibility across 8 AEs"),
    dict(name="Maria Santos",   role="Chief Revenue Officer",      company="HealthGrid",   segment="enterprise", employees=850,
         signal="cut 12% of sales headcount in 2026-Q1",          signal_type="layoff_event",
         signal_src="layoffs_fyi",  signal_win="2026-Q1",         pain="60% of reps miss quota in their first 6 months"),
    dict(name="James Park",     role="Founder & CEO",             company="Proxify",      segment="smb",       employees=14,
         signal="posted a Head of Sales job on LinkedIn 2 days ago", signal_type="job_posting",
         signal_src="synthetic",    signal_win=None,               pain="founder-led sales is not scaling past $500k ARR"),
    dict(name="Rachel Kim",     role="VP of Sales",               company="LearnUp",      segment="series_b",  employees=95,
         signal="launched enterprise product tier in 2025-Q4",    signal_type="product_launch",
         signal_src="synthetic",    signal_win="2025-Q4",          pain="enterprise reps lack structured playbooks"),
    dict(name="David Chen",     role="Head of Sales Operations",  company="FlowRoute",    segment="enterprise", employees=620,
         signal="missed Q3 revenue target by 14% per earnings call", signal_type="earnings_miss",
         signal_src="synthetic",    signal_win="2025-Q3",          pain="sales cycle length increased from 45 to 68 days"),
    dict(name="Lisa Patel",     role="Director of Business Dev",  company="LexPath",      segment="series_b",  employees=65,
         signal="closed $18M Series A in 2025-Q1",                signal_type="funding_round",
         signal_src="crunchbase_odm", signal_win="2025-Q1",       pain="BD team has no systematic outreach process"),
    dict(name="Carlos Rivera",  role="Founder & COO",             company="BrandSpark",   segment="smb",       employees=22,
         signal="posted a VP of Sales job on LinkedIn 3 days ago", signal_type="job_posting",
         signal_src="synthetic",    signal_win=None,               pain="first sales hire needs onboarding framework"),
    dict(name="Emily Zhang",    role="VP of Customer Success",    company="ShieldOps",    segment="enterprise", employees=400,
         signal="cut 18% of CS headcount in 2026-Q1",             signal_type="layoff_event",
         signal_src="layoffs_fyi",  signal_win="2026-Q1",          pain="expansion revenue declining as CS team shrinks"),
    dict(name="Michael Brown",  role="Head of Growth",            company="PropStack",    segment="series_b",  employees=130,
         signal="raised $22M Series B in 2025-Q2",                signal_type="funding_round",
         signal_src="crunchbase_odm", signal_win="2025-Q2",       pain="growth team running manual outreach with no tooling"),
    dict(name="Nina Johnson",   role="CEO & Co-founder",          company="MakeWise",     segment="smb",       employees=9,
         signal="posted a Sales Manager job on LinkedIn yesterday", signal_type="job_posting",
         signal_src="synthetic",    signal_win=None,               pain="closing rate is 8% from qualified leads"),
    dict(name="Tom Wilson",     role="VP of Marketing",           company="DataHaven",    segment="series_b",  employees=155,
         signal="launched new enterprise pricing tier in 2025-Q3", signal_type="product_launch",
         signal_src="synthetic",    signal_win="2025-Q3",          pain="marketing-sourced pipeline converts at half the rate of outbound"),
    dict(name="Sofia Martinez", role="Chief Revenue Officer",      company="CoverBase",    segment="enterprise", employees=780,
         signal="missed Q4 targets by 11% per investor update",   signal_type="earnings_miss",
         signal_src="synthetic",    signal_win="2025-Q4",          pain="AE ramp is 110 days vs 60-day industry benchmark"),
    dict(name="Kevin Lee",      role="Founder & CEO",             company="DevPulse",     segment="smb",       employees=18,
         signal="posted a Head of Revenue job on LinkedIn 4 days ago", signal_type="job_posting",
         signal_src="synthetic",    signal_win=None,               pain="dev-led GTM needs a systematic sales layer"),
    dict(name="Amanda Foster",  role="Head of Revenue",           company="TalentBridge", segment="series_b",  employees=200,
         signal="closed $28M Series B in 2025-Q1",                signal_type="funding_round",
         signal_src="crunchbase_odm", signal_win="2025-Q1",       pain="new reps take 100+ days to first close"),
    dict(name="Robert Kim",     role="VP of Sales",               company="ShopFront",    segment="enterprise", employees=950,
         signal="cut 22% of RevOps and enablement staff in 2026-Q1", signal_type="layoff_event",
         signal_src="layoffs_fyi",  signal_win="2026-Q1",          pain="sales tooling now unsupported after team reductions"),
]

TASK_CONFIGS = [
    dict(task_type="email_outreach",
         word_limit=120, constraints_extra=[
             "reference the {signal_type} signal explicitly",
             "include [CALENDLY_LINK]", "do not mention pricing"],
         failure_tag="signal_missing", adv_weight=0.5),
    dict(task_type="follow_up",
         word_limit=80,  constraints_extra=[
             "this is a second touch — 5 days after a first email referencing the {signal_type} signal",
             "do NOT say 'just checking in', 'touching base', or 'circling back'",
             "include [CALENDLY_LINK]"],
         failure_tag="tone_drift",    adv_weight=0.5),
    dict(task_type="discovery_response",
         word_limit=150, constraints_extra=[
             "prospect asked: 'How would Tenacious specifically help with {pain}?'",
             "address the question directly with evidence",
             "include [CALENDLY_LINK]"],
         failure_tag="formulaic",     adv_weight=0.5),
    dict(task_type="objection_handling",
         word_limit=100, constraints_extra=[
             "prospect says: 'We already have something in place for this'",
             "acknowledge the objection before pivoting",
             "reference the {signal_type} signal as context",
             "do not mention pricing"],
         failure_tag="trajectory",    adv_weight=0.5),
    dict(task_type="closing",
         word_limit=80,  constraints_extra=[
             "prospect has expressed interest — write the closing message",
             "include a specific proposed start date or timeline",
             "include [CALENDLY_LINK]", "no filler — they are already engaged"],
         failure_tag="tone_drift",    adv_weight=0.5),
]

DIFFICULTY_MAP = {
    "series_b":  "medium",
    "enterprise": "hard",
    "smb":        "easy",
}


def build_tasks():
    tasks = []
    tid = START_ID
    for profile in PROFILES:
        for cfg in TASK_CONFIGS:
            context = (
                f"Prospect: {profile['name']}, {profile['role']} at {profile['company']} "
                f"({profile['segment'].replace('_', ' ').title()}, {profile['employees']} employees). "
                f"Signal: {profile['company']} {profile['signal']}. "
                f"Known pain point: {profile['pain']}."
            )
            constraints = [f"under {cfg['word_limit']} words"]
            for c in cfg["constraints_extra"]:
                constraints.append(
                    c.replace("{signal_type}", profile["signal_type"])
                     .replace("{pain}", profile["pain"])
                )
            tasks.append(make_task(
                tid=tid,
                authoring_mode="programmatic",
                source_traces=[],
                difficulty=DIFFICULTY_MAP[profile["segment"]],
                context=context,
                task_type=cfg["task_type"],
                constraints=constraints,
                segment=profile["segment"],
                failure_tag=cfg["failure_tag"],
                adv_weight=cfg["adv_weight"],
                signal_source=profile["signal_src"],
                signal_time_window=profile["signal_win"],
                generation_model="template_expansion",
            ))
            tid += 1
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Programmatic task generation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] Would write {len(tasks)} tasks to {OUT_FILE}")
        segs = {}
        for t in tasks:
            s = t["metadata"]["tenacious_segment"]
            segs[s] = segs.get(s, 0) + 1
        print(f"  Segments: {segs}")
        return

    with open(OUT_FILE, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    print(f"[programmatic] Wrote {len(tasks)} tasks → {OUT_FILE}")
    segs = {}
    types = {}
    for t in tasks:
        segs[t["metadata"]["tenacious_segment"]] = segs.get(t["metadata"]["tenacious_segment"], 0) + 1
        types[t["input"]["task_type"]] = types.get(t["input"]["task_type"], 0) + 1
    print(f"  Segments  : {segs}")
    print(f"  Task types: {types}")


if __name__ == "__main__":
    import json
    main()
