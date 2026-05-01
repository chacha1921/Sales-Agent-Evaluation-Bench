#!/usr/bin/env python3
"""
generate_sft_data.py — Generate gold SFT training pairs for Path A.

For each task in the training split, generates N rubric-compliant gold
outputs per task. Each (task, output) pair becomes one SFT training example
in OpenAI chat (ChatML) format, ready for Unsloth/TRL fine-tuning.

Usage:
    # Mock mode — template-based, no API key needed (~990 pairs, ~30s)
    python training/generate_sft_data.py --mock

    # Live mode — Claude Haiku, requires ANTHROPIC_API_KEY (~990 pairs, ~$0.30)
    python training/generate_sft_data.py --live

    # Adjust pairs per task
    python training/generate_sft_data.py --live --n-per-task 5

    # Seed for reproducibility
    python training/generate_sft_data.py --mock --seed 42
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "dataset" / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
OUT_DIR   = ROOT / "training" / "training_data" / "path_a_sft"
OUT_FILE  = OUT_DIR / "sft_pairs.jsonl"
STATS_FILE = OUT_DIR / "generation_stats.json"

_DEFAULT_SEED = 42

# ── System prompt (Tenacious brand voice) ────────────────────────────────────

TENACIOUS_SYSTEM_PROMPT = """You are a B2B sales assistant for Tenacious, an AI-assisted revenue intelligence platform. You write outbound sales emails, follow-ups, and objection responses on behalf of account executives.

Your outputs must follow these rules without exception:

TONE AND VOICE
- Direct and specific. Open with the prospect's context, not a greeting.
- Human. Write like a thoughtful person, not a marketing template.
- No urgency or pressure. Never say things like "act now" or "don't miss out".
- One ask per message. Do not stack requests.

SIGNAL GROUNDING
- Always reference the specific trigger that prompted outreach: funding round, headcount change, leadership hire, job posting, layoff, or product launch.
- Use exact details: amounts, percentages, quarter, role title, or company name from the context.

HARD BANS — never write any of these:
just checking in · touching base · circling back · following up · I hope this email finds you well · I hope you're doing well · I wanted to reach out · per our conversation · as per my last email · let's connect · would love to connect · hop on a call · quick chat · quick call · pick your brain · thought leader · thought leadership · leverage · synergy · synergize · solution · utilize · end-to-end · best-in-class · world-class · game-changer · game-changing · revolutionary · cutting-edge · state-of-the-art · paradigm shift · move the needle · low-hanging fruit · boil the ocean · deep dive · take it offline · disruptive · holistic · streamline your workflow · empower your team · at the end of the day · seamlessly integrates

FORMATTING
- Include [CALENDLY_LINK] exactly once as the calendar booking link.
- Respect the word limit in the constraints.
- No pricing or cost language of any kind.
- Use the prospect's first name, not full name, after the opening line."""

# ── Context parser ────────────────────────────────────────────────────────────

def parse_context(context: str) -> dict:
    """Extract structured fields from a free-text context string."""
    info = {
        "name": "", "first_name": "", "title": "", "company": "",
        "segment": "", "size": "", "signal": "", "pain_point": "",
        "raw": context,
    }

    m = re.match(r"Prospect:\s*([^,]+),\s*([^a]+at)\s+([^(]+)\(([^)]+)\)", context)
    if m:
        info["name"] = m.group(1).strip()
        info["first_name"] = info["name"].split()[0]
        info["title"] = m.group(2).replace(" at", "").strip()
        info["company"] = m.group(3).strip()
        raw_seg = m.group(4).strip()
        # Clean segment to just the tier label (strip employee count)
        info["segment"] = raw_seg.split(",")[0].strip()

    size_m = re.search(r"(\d+)\s+employees", context)
    if size_m:
        info["size"] = size_m.group(1)

    signal_m = re.search(r"Signal:\s*(.+?)(?:\.|Known pain)", context, re.DOTALL)
    if signal_m:
        info["signal"] = signal_m.group(1).strip().rstrip(".")

    pain_m = re.search(r"[Kk]nown pain point:\s*(.+?)(?:\.|$)", context, re.DOTALL)
    if pain_m:
        info["pain_point"] = pain_m.group(1).strip().rstrip(".")

    return info


def build_user_prompt(task: dict) -> str:
    ctx = task["input"]["context"]
    task_type = task["input"]["task_type"]
    constraints = task["input"].get("constraints", [])
    constraint_str = "\n".join(f"- {c}" for c in constraints)
    return f"""Write a {task_type.replace('_', ' ')} for this prospect.

Context:
{ctx}

Constraints:
{constraint_str}

Write only the message body. No subject line unless the constraints explicitly ask for one. No explanatory text."""


# ── Mock gold output generation ───────────────────────────────────────────────

TEMPLATES_EMAIL_OUTREACH = [
    "{first} — {signal}. For a {size}-person team, {pain_point} often intensifies at this stage. We've helped similar {segment} companies address this without adding headcount. Worth 20 minutes? {link}",
    "{first}, saw that {signal}. One thing teams at your stage consistently flag is {pain_point} — usually right when pipeline pressure spikes. Happy to show you how we've tackled this for comparable teams. {link}",
    "{first}, {company} recently {signal}. I work with {title}s at {segment} companies who are dealing with {pain_point} around this milestone. Can I share what's worked? {link}",
    "{first} — {signal}. The {title}s I talk with at this stage say {pain_point} is the highest-friction problem going into their next review. I have a specific angle on this — 20 minutes? {link}",
    "Quick note, {first}: {signal}. Most {segment} teams find {pain_point} is where they lose time and deals at this stage. Here's a {link} to grab 20 minutes if this resonates.",
    "{first}, one data point from working with {segment} teams: after {signal}, {pain_point} is the number-one thing their {title}s flag in pipeline reviews. We address exactly that. {link}",
    "{first} — {signal}. I work with revenue leaders facing {pain_point} at this stage — curious if you've found a good answer yet. {link} for a brief conversation if you haven't.",
    "{first} — {signal} is exactly the moment when {pain_point} starts showing up in forecast calls. We help teams fix that before it costs deals. {link} if you'd like to see how.",
    "{first}, saw that {signal}. {pain_point} is usually what {segment} revenue teams are solving at this point. One idea here that's worked well. {link}",
    "{first}, {signal}. {pain_point} is a common ceiling for teams at your stage. I have 20 minutes blocked — worth a look? {link}",
]

TEMPLATES_FOLLOW_UP = [
    "{first} — wanted to add one thing since my last note: {signal}. That makes {pain_point} even more worth a conversation now. {link}",
    "{first}, {signal} is still relevant to what I mentioned. {pain_point} doesn't get easier at this stage. {link} for 20 minutes when you have bandwidth.",
    "{first}, a second note: {signal} — for a team your size, {pain_point} is the kind of problem that compounds. Still worth 20 minutes. {link}",
    "Still relevant, {first}: {signal}. The teams I work with who fix {pain_point} at this stage avoid a lot of friction later. {link}",
    "{first} — {signal}. One specific approach to {pain_point} I didn't mention: [specific angle]. Worth 20 minutes? {link}",
    "{first} — one more note on {signal}: this makes {pain_point} more time-sensitive, not less. Still worth 20 minutes. {link}",
    "{first}, one more angle here: {signal} means {pain_point} affects forecast accuracy directly. Happy to walk through a 15-minute demo. {link}",
    "{first} — still thinking about {signal}. The teams I work with who addressed {pain_point} early saved roughly 20% of AE time. {link} if that's worth exploring.",
    "{first}, brief note: {signal} moves {pain_point} up the priority list for most {title}s I work with. Still interested in a quick look? {link}",
    "{first} — {signal}. {pain_point} is something we've helped {segment} teams address directly. {link}",
]

TEMPLATES_OBJECTION_HANDLING = [
    "That makes sense, {first}. {pain_point} doesn't always feel urgent until it's the thing slowing down Q{q} close. Here's what teams in your situation have done — {link} to discuss.",
    "Understood — {pain_point} is a real constraint. What I hear from {segment} {title}s is that {signal} often shifts the timeline. Worth 15 minutes to see if the timing changes? {link}",
    "Hear you, {first}. Most {title}s I talk with say the same thing before {signal}. After that milestone, {pain_point} tends to move to the top of the list. Still think 20 minutes is worth it. {link}",
    "Fair point, {first} — {signal} means priorities are shifting. That said, {pain_point} is usually what makes the next stage harder. I'll keep this brief: {link}.",
    "I hear you. {signal} makes the timing tricky. That said, {pain_point} is the one thing that typically doesn't wait. Can we spend 15 minutes on just that? {link}",
    "That's a reasonable concern. What I've seen at companies after {signal} is that {pain_point} becomes the constraint that slows the next round of growth. Happy to show specifically how — {link}.",
    "Completely understand the hesitation, {first}. {signal} is exactly when addressing {pain_point} has the highest ROI. One conversation to decide? {link}",
    "That makes complete sense given {signal}. The question I'd ask is whether {pain_point} is something you're managing today or hoping to manage later. Happy to be concrete: {link}.",
    "Understood, {first}. {signal} is a lot to absorb. I'll keep this one question: is {pain_point} on your list for this half? If yes, 20 minutes might matter. {link}",
    "Fair. Here's what I'd focus on: {pain_point} is the one thing I've seen derail {segment} teams at your stage after {signal}. Brief conversation to see if it applies? {link}",
]

TEMPLATES_CLOSING = [
    "{first}, based on what you shared about {pain_point} and {signal}, I think we have a clear fit. Can we confirm timing this week? {link}",
    "{first}, {pain_point} at {company} is exactly where we're strongest. I've blocked time this week — {link} to confirm.",
    "{first}, we've covered {pain_point} at {company} in detail. The next step is a brief legal/procurement check on our end. Can we schedule that this week? {link}",
    "Ready to move forward, {first}. {signal} and the {pain_point} discussion make the case clear. {link} to set the kickoff.",
    "{first} — where we've landed: {signal} creates the right moment to address {pain_point}. I'm confident in the fit. {link} to confirm the next step.",
    "{first}, {pain_point} at {company} is exactly what we built this for. Based on our last call, I'd suggest moving to a trial this week. {link} to align on scope.",
    "One last question before we close, {first}: is {pain_point} still the top priority for Q{q}? If yes, I'm confident we can help — {link} to lock this in.",
    "{first}, we've aligned on {pain_point} and the {signal} timing makes sense. Let's formalise this — {link}.",
    "Based on everything, {first}: {signal} + {pain_point} = a strong case for moving now. {link} to confirm the deal structure.",
    "{first} — {pain_point} at {company} is solvable, and the timing with {signal} is good. {link} to close this out.",
]

TEMPLATES_DISCOVERY_RESPONSE = [
    "Good question, {first}. Most {segment} teams I work with after {signal} say {pain_point} is the first thing they'd address. I'd ask: how are you measuring it today? {link} to continue this conversation.",
    "{first}, great point about {pain_point}. Given {signal}, I'd focus on two things: [X] and [Y]. Worth going deeper? {link}",
    "To answer directly, {first}: {pain_point} after {signal} is a data-visibility problem more than a process problem. Here's why that matters for {company}. {link} to discuss.",
    "{first} — the short answer is that {pain_point} at {company}'s stage typically manifests as [specific symptom]. Does that match what you're seeing? {link}",
    "Good question. For {company} specifically — after {signal} — {pain_point} usually surfaces in [two specific places]. Happy to walk through what we've seen. {link}",
    "{first}, {signal} means {pain_point} is going to hit your team differently than it did before. Let me show you how. {link}",
    "On {pain_point}: what I've seen with {segment} teams at {company}'s stage is [specific pattern]. Does that align with what your AEs are reporting? {link} to keep going.",
    "Fair question, {first}. The honest answer: {pain_point} is where most {segment} revenue teams feel the most friction after {signal}. I can show you three things that help. {link}",
    "{first}, to be specific: {signal} means your {title} team is probably dealing with {pain_point} in [two specific ways]. Let's test that assumption. {link}",
    "Great question. {signal} creates a specific version of {pain_point} that I've seen before. Here's what worked. {link}",
]

TEMPLATES_OUTREACH_NO_PRICING = [
    "{first} — quick note: {signal} is the kind of milestone where {pain_point} gets harder before it gets easier. Happy to share what's worked without any commercial conversation first. {link}",
    "{first}, saw that {signal}. Before we talk about anything else, I'd like to share what {segment} teams use to address {pain_point} at your stage — no pricing attached. {link}",
    "{first}, {signal} put you on my radar. I keep this initial conversation strictly about the problem: {pain_point}. {link} for 20 minutes on that alone.",
]

TEMPLATES_BY_TYPE = {
    "email_outreach": TEMPLATES_EMAIL_OUTREACH,
    "follow_up": TEMPLATES_FOLLOW_UP,
    "objection_handling": TEMPLATES_OBJECTION_HANDLING,
    "closing": TEMPLATES_CLOSING,
    "discovery_response": TEMPLATES_DISCOVERY_RESPONSE,
    "email_outreach_no_pricing": TEMPLATES_OUTREACH_NO_PRICING,
}


def _pick_quarter() -> str:
    return str(random.choice([1, 2, 3, 4]))


def mock_gold_output(task: dict, variant_idx: int) -> str:
    """Generate a template-based gold output that passes all rubric checks."""
    info = parse_context(task["input"]["context"])
    task_type = task["input"]["task_type"]
    templates = TEMPLATES_BY_TYPE.get(task_type, TEMPLATES_EMAIL_OUTREACH)
    template = templates[variant_idx % len(templates)]

    filled = template.format(
        first=info["first_name"] or "Hi",
        name=info["name"] or "there",
        title=info["title"] or "VP",
        company=info["company"] or "your company",
        segment=info["segment"] or "growth-stage",
        size=info["size"] or "your",
        signal=info["signal"] or "recent growth signal",
        pain_point=info["pain_point"] or "pipeline visibility",
        link="[CALENDLY_LINK]",
        q=_pick_quarter(),
    )

    # Enforce word count constraints
    constraints = task["input"].get("constraints", [])
    for c in constraints:
        m = re.search(r"under\s+(\d+)\s+words?", c, re.I)
        if m:
            limit = int(m.group(1))
            words = filled.split()
            if len(words) > limit:
                filled = " ".join(words[: limit - 3]) + " [CALENDLY_LINK]"
            break

    return filled


# ── Live generation (Claude Haiku) ────────────────────────────────────────────

LIVE_GENERATION_PROMPT = """\
Write {n} distinct variations of this sales message. Each variation must:
- Use a different opening approach (insight, question, data point, contrast, specific outcome)
- Reference the signal explicitly using exact details from the context
- Follow all Tenacious brand-voice rules in the system prompt
- Stay under the specified word limit
- Include [CALENDLY_LINK] exactly once per variation

Return a JSON array of strings — one string per variation. No other text.

Task:
{task_prompt}
"""


def live_gold_outputs(task: dict, n: int, client) -> list[str]:
    """Generate gold outputs via Gemini Flash (leakage-safe: judge is DeepSeek)."""
    from google.genai import types as genai_types
    task_prompt = build_user_prompt(task)
    user_msg = LIVE_GENERATION_PROMPT.format(n=n, task_prompt=task_prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=genai_types.GenerateContentConfig(
            system_instruction=TENACIOUS_SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=2048,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip()

    # Parse JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            outputs = json.loads(m.group(0))
            return [str(o) for o in outputs if o]
        except json.JSONDecodeError:
            pass

    # Fallback: split on numbered list
    outputs = re.split(r"\n\d+[\.\)]\s+", raw)
    return [o.strip() for o in outputs if o.strip()][:n]


# ── SFT pair builder ──────────────────────────────────────────────────────────

def build_sft_pair(task: dict, gold_output: str, variant_idx: int) -> dict:
    """Format one (task, output) pair as an OpenAI-chat-style training example."""
    return {
        "task_id": task["task_id"],
        "variant": variant_idx,
        "messages": [
            {"role": "system",    "content": TENACIOUS_SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_prompt(task)},
            {"role": "assistant", "content": gold_output},
        ],
        "metadata": {
            "segment":      task["metadata"]["tenacious_segment"],
            "task_type":    task["input"]["task_type"],
            "failure_mode": task["metadata"]["failure_mode_tag"],
            "source":       "mock" if variant_idx >= 0 else "live",
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SFT training pairs for Path A")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Template-based generation (no API key)")
    mode.add_argument("--live", action="store_true", help="Gemini Flash generation (requires GOOGLE_API_KEY)")
    parser.add_argument("--n-per-task", type=int, default=10,
                        help="Gold outputs per task (default: 10 → ~990 total pairs)")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit to first N tasks (for testing)")
    args = parser.parse_args()

    random.seed(args.seed)
    mode_str = "mock" if args.mock else "live"
    print(f"[generate_sft_data] mode={mode_str}  seed={args.seed}  n_per_task={args.n_per_task}")

    if not TRAIN_FILE.exists():
        print(f"[ERROR] {TRAIN_FILE} not found. Run contamination_check.py first.")
        sys.exit(1)

    tasks = []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    print(f"  Tasks loaded: {len(tasks)}")

    client = None
    if args.live:
        try:
            import os
            from pathlib import Path as _Path
            _env = _Path(__file__).parent.parent / ".env"
            if _env.exists():
                for _line in _env.read_text().splitlines():
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        os.environ.setdefault(_k.strip(), _v.strip())
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("[ERROR] GOOGLE_API_KEY not set. Use --mock or set the environment variable.")
                sys.exit(1)
            client = genai.Client(api_key=api_key)
        except ImportError:
            print("[ERROR] google-genai not installed. Run: pip install google-genai")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = []
    failed = 0

    for i, task in enumerate(tasks):
        if args.mock:
            for v in range(args.n_per_task):
                output = mock_gold_output(task, v)
                pairs.append(build_sft_pair(task, output, v))
        else:
            try:
                outputs = live_gold_outputs(task, args.n_per_task, client)
                for v, output in enumerate(outputs):
                    pairs.append(build_sft_pair(task, output, v))
                time.sleep(0.2)  # rate limit
            except Exception as e:
                print(f"  [WARN] Task {task['task_id']} failed: {e}")
                failed += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tasks)} tasks ({len(pairs)} pairs so far)")

    # Write output
    with open(OUT_FILE, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode_str,
        "seed": args.seed,
        "n_per_task": args.n_per_task,
        "tasks_processed": len(tasks),
        "tasks_failed": failed,
        "total_pairs": len(pairs),
        "out_file": str(OUT_FILE),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DONE] {len(pairs)} SFT pairs → {OUT_FILE}")
    print(f"  Stats → {STATS_FILE}")
    if args.mock:
        print(f"\n  Note: mock outputs are template-based. Run --live with ANTHROPIC_API_KEY")
        print(f"  for model-generated gold outputs before the actual fine-tuning run.")


if __name__ == "__main__":
    main()
