#!/usr/bin/env python3
"""
generate_preference_pairs.py — Build chosen/rejected preference pairs for ORPO and SimPO.

Each training task gets:
  - chosen:   a rubric-compliant output (signal-led, no banned phrases, CTA present)
  - rejected: a deliberately corrupted output matching the task's failure_mode_tag

Modes:
  Mock (default): Python templates — fast, no API key, but token overlap ~90%
                  (ORPO/SimPO preference loss gets no gradient signal)
  Live (--live):  Gemini Flash writes genuinely diverse chosen + rejected outputs.
                  Token overlap ~20-30% — activates full preference gradient.
                  Cost: ~$0.03 for 254 pairs. Requires GOOGLE_API_KEY in .env.

Output format (TRL/Unsloth ORPOTrainer / CPOTrainer compatible):
  {"prompt": [...messages...], "chosen": [...], "rejected": [...]}

Usage:
    python training/generate_preference_pairs.py              # mock (default)
    python training/generate_preference_pairs.py --live       # Gemini Flash live
    python training/generate_preference_pairs.py --live --n-rejected 3
    python training/generate_preference_pairs.py --also-dev   # include dev split
    python training/generate_preference_pairs.py --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT       = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "dataset" / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
DEV_FILE   = ROOT / "dataset" / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
SFT_FILE   = ROOT / "training" / "training_data" / "path_a_sft" / "sft_pairs.jsonl"
OUT_DIR    = ROOT / "training" / "training_data" / "path_b_dpo"
OUT_FILE   = OUT_DIR / "preference_pairs.jsonl"
STATS_FILE = OUT_DIR / "generation_stats.json"

_DEFAULT_SEED = 42

# ── System prompt (same as SFT — consistent across all training) ──────────────

SYSTEM_PROMPT = """You are a B2B sales assistant for Tenacious, an AI-assisted revenue intelligence platform. You write outbound sales emails, follow-ups, and objection responses on behalf of account executives.

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

# ── Rejection generators (one per failure mode) ───────────────────────────────

_TONE_DRIFT_OPENERS = [
    "I hope this email finds you well. ",
    "I hope you're doing well. ",
]

_TONE_DRIFT_INSERTS = [
    " Our platform helps you leverage your existing data to synergize across teams.",
    " We offer a best-in-class, end-to-end solution that will revolutionize your workflow.",
    " Our cutting-edge approach moves the needle for revenue teams at scale.",
    " We have a game-changing platform that helps teams streamline their workflows.",
]

_FORMULAIC_OPENERS = [
    "My name is Jordan and I work at Tenacious. ",
    "My name is Alex and I'm reaching out from Tenacious. ",
    "I'm reaching out from Tenacious to introduce our platform. ",
]

_SIGNAL_FREE_BODIES = [
    "{first}, I wanted to reach out about how Tenacious can help {company} improve pipeline visibility. Our solution is used by revenue teams across the industry. Would love to connect for a quick chat — [CALENDLY_LINK]",
    "Hi {first}, I'm touching base to introduce Tenacious — we help sales teams streamline their workflows and leverage data insights. Let me know if you'd like to hop on a call. [CALENDLY_LINK]",
    "Hope you're doing well, {first}. I wanted to reach out because Tenacious helps companies like {company} with end-to-end revenue intelligence. Would love to connect. [CALENDLY_LINK]",
]

_PRICING_INSERTS = [
    " Pricing starts at $500/month per seat.",
    " Our plans start from $299/month with annual contracts available.",
    " We offer flexible pricing starting at $1,500/month for teams your size.",
]


def _parse_first_name(context: str) -> str:
    m = re.match(r"Prospect:\s*([A-Z][a-z]+)", context)
    return m.group(1) if m else "there"


def _parse_company(context: str) -> str:
    m = re.search(r"at\s+([^(]+)\s*\(", context)
    return m.group(1).strip() if m else "your company"


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_overlimit(text: str, limit: int) -> str:
    """Return text that is at least 20 words over the limit."""
    words = text.split()
    padding = " We are excited to share more about our innovative, world-class platform and its game-changing capabilities for your organization. Please don't hesitate to let us know if you'd like to learn more about our comprehensive end-to-end solution."
    while _word_count(text) <= limit + 5:
        text = text + padding
    return text


def make_rejected(task: dict, chosen: str, variant: int = 0) -> str:
    """Generate a rejected output by injecting the task's failure mode."""
    failure = task["metadata"]["failure_mode_tag"]
    context = task["input"]["context"]
    constraints = task["input"].get("constraints", [])
    first = _parse_first_name(context)
    company = _parse_company(context)

    rng_choice = lambda lst: lst[variant % len(lst)]

    if failure == "tone_drift":
        opener = rng_choice(_TONE_DRIFT_OPENERS)
        insert = rng_choice(_TONE_DRIFT_INSERTS)
        # Prepend formulaic opener and inject banned phrases mid-text
        return opener + chosen + insert

    elif failure == "signal_missing":
        # Replace with a generic, signal-free version
        template = rng_choice(_SIGNAL_FREE_BODIES)
        return template.format(first=first, company=company)

    elif failure == "formulaic":
        opener = rng_choice(_FORMULAIC_OPENERS)
        return opener + chosen

    elif failure == "constraint_violation":
        # Find word limit and blow past it, or inject pricing
        word_limit = None
        for c in constraints:
            m = re.search(r"under\s+(\d+)\s+words?", c, re.I)
            if m:
                word_limit = int(m.group(1))
                break
        if word_limit:
            return _enforce_overlimit(chosen, word_limit)
        else:
            pricing = rng_choice(_PRICING_INSERTS)
            return chosen + pricing

    else:
        # trajectory or unknown — use a generic off-topic response
        return (
            f"Hi {first}, I hope you're doing well. I wanted to reach out and "
            f"introduce Tenacious. We help companies like {company} with revenue "
            f"intelligence. Just checking in to see if you'd be interested in a "
            f"quick chat. Let me know! [CALENDLY_LINK]"
        )


# ── Live generation (Gemini Flash) ────────────────────────────────────────────

_FAILURE_MODE_INSTRUCTIONS = {
    "tone_drift": (
        "Write a sales email for the same prospect that deliberately VIOLATES Tenacious "
        "brand rules. Start with 'I hope this email finds you well.' and include at least "
        "two of these banned phrases somewhere in the body: leverage, synergy, best-in-class, "
        "cutting-edge, game-changing, streamline your workflow. Still include [CALENDLY_LINK]."
    ),
    "signal_missing": (
        "Write a completely generic sales pitch for the same prospect that IGNORES the "
        "specific trigger signal entirely — do not mention it at all. Use filler openers "
        "like 'I wanted to reach out' or 'just checking in'. Make it feel like a mass "
        "blast that could be sent to anyone. Include [CALENDLY_LINK]."
    ),
    "formulaic": (
        "Write a sales email for the same prospect that starts with a self-introduction "
        "opener: 'My name is [name] and I work at Tenacious.' Then continue with a "
        "generic pitch. Include [CALENDLY_LINK]."
    ),
    "trajectory": (
        "Write a sales email for the same prospect that completely IGNORES any conversation "
        "history, prior objection, or previous message from the prospect. Respond as if "
        "this is a brand-new cold outreach — the first contact ever. Include [CALENDLY_LINK]."
    ),
    "constraint_violation": (
        "Write a sales email for the same prospect that VIOLATES the word-count constraint "
        "by writing at least 30 words over the limit. Also mention pricing somewhere "
        "(e.g., 'Pricing starts at $X/month'). Include [CALENDLY_LINK]."
    ),
}


def _call_gemini(client, system_prompt: str, user_prompt: str) -> str:
    from google.genai import types as genai_types
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8,
                    max_output_tokens=512,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return resp.text.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                raise e


def live_generate_pair(task: dict, client, variant: int) -> tuple[str, str]:
    """Call Gemini Flash twice: once for chosen (clean), once for rejected (failure mode)."""
    ctx = task["input"]["context"]
    task_type = task["input"]["task_type"]
    constraints = task["input"].get("constraints", [])
    constraint_str = "\n".join(f"- {c}" for c in constraints)
    failure_mode = task["metadata"]["failure_mode_tag"]

    base_user = f"""Write a {task_type.replace('_', ' ')} for this prospect.

Context:
{ctx}

Constraints:
{constraint_str}

Write only the message body. No subject line unless constraints ask for one. No explanatory text."""

    # Chosen: clean, rules-compliant output
    chosen = _call_gemini(client, SYSTEM_PROMPT, base_user)
    time.sleep(0.3)

    # Rejected: deliberately bad version matching the failure mode
    failure_instruction = _FAILURE_MODE_INSTRUCTIONS.get(
        failure_mode,
        "Write a generic, low-quality version that ignores the signal and uses clichés.",
    )
    rejected_user = f"""{failure_instruction}

Context:
{ctx}

Constraints (word limit still applies unless you are testing constraint_violation):
{constraint_str}

Write only the message body. No explanatory text."""

    rejected = _call_gemini(client, "", rejected_user)
    return chosen, rejected


def _init_gemini_client():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    try:
        from google import genai
    except ImportError:
        print("[ERROR] google-genai not installed. Run: pip install google-genai")
        sys.exit(1)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set in .env or environment.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# ── Preference pair builder ───────────────────────────────────────────────────

def build_prompt_messages(task: dict) -> list:
    ctx = task["input"]["context"]
    task_type = task["input"]["task_type"]
    constraints = task["input"].get("constraints", [])
    constraint_str = "\n".join(f"- {c}" for c in constraints)
    user_content = f"""Write a {task_type.replace('_', ' ')} for this prospect.

Context:
{ctx}

Constraints:
{constraint_str}

Write only the message body. No subject line unless the constraints explicitly ask for one. No explanatory text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_pair(task: dict, chosen: str, rejected: str, variant: int) -> dict:
    return {
        "task_id":  task["task_id"],
        "variant":  variant,
        "prompt":   build_prompt_messages(task),
        "chosen":   [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {
            "segment":      task["metadata"]["tenacious_segment"],
            "task_type":    task["input"]["task_type"],
            "failure_mode": task["metadata"]["failure_mode_tag"],
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_chosen_map(sft_file: Path) -> dict:
    """Return {task_id: {variant: output_text}} from SFT pairs."""
    chosen_map = {}
    if not sft_file.exists():
        return chosen_map
    for line in open(sft_file):
        p = json.loads(line)
        tid = p["task_id"]
        v   = p["variant"]
        txt = p["messages"][2]["content"]
        chosen_map.setdefault(tid, {})[v] = txt
    return chosen_map


def process_tasks(tasks: list, chosen_map: dict, n_rejected: int) -> list:
    pairs = []
    missing_chosen = 0
    for task in tasks:
        tid = task["task_id"]
        # Use variant 0 as the primary chosen; fall back to variant 1, 2...
        chosen_variants = chosen_map.get(tid, {})
        if not chosen_variants:
            missing_chosen += 1
            continue

        for rej_idx in range(n_rejected):
            chosen_v = rej_idx % max(len(chosen_variants), 1)
            chosen = chosen_variants.get(chosen_v) or chosen_variants[0]
            rejected = make_rejected(task, chosen, variant=rej_idx)
            pairs.append(build_pair(task, chosen, rejected, rej_idx))

    if missing_chosen:
        print(f"  [WARN] {missing_chosen} tasks had no chosen output — "
              f"run generate_sft_data.py first.")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs for ORPO/SimPO")
    parser.add_argument("--live", action="store_true",
                        help="Call Gemini Flash for diverse chosen+rejected outputs "
                             "(requires GOOGLE_API_KEY in .env). Recommended for real training.")
    parser.add_argument("--also-dev", action="store_true",
                        help="Include dev split tasks (adds ~63 tasks)")
    parser.add_argument("--n-rejected", type=int, default=3,
                        help="Rejected variants per task (default: 3 → 381+ pairs)")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    mode = "live" if args.live else "mock"
    print(f"[generate_preference_pairs] mode={mode}  seed={args.seed}  "
          f"also_dev={args.also_dev}  n_rejected={args.n_rejected}")

    if not TRAIN_FILE.exists():
        print(f"[ERROR] {TRAIN_FILE} not found.")
        sys.exit(1)

    tasks = [json.loads(l) for l in open(TRAIN_FILE) if l.strip()]
    print(f"  Train tasks: {len(tasks)}")

    if args.also_dev and DEV_FILE.exists():
        dev_tasks = [json.loads(l) for l in open(DEV_FILE) if l.strip()]
        tasks += dev_tasks
        print(f"  Dev tasks added: {len(dev_tasks)} → total: {len(tasks)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = []
    failed = 0

    if args.live:
        client = _init_gemini_client()
        print(f"  Gemini Flash client ready — generating {len(tasks) * args.n_rejected} pairs...")
        for i, task in enumerate(tasks):
            for rej_idx in range(args.n_rejected):
                try:
                    chosen, rejected = live_generate_pair(task, client, rej_idx)
                    pairs.append(build_pair(task, chosen, rejected, rej_idx))
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  [WARN] {task['task_id']} variant {rej_idx} failed: {e}")
                    failed += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks done ({len(pairs)} pairs)")
    else:
        chosen_map = load_chosen_map(SFT_FILE)
        print(f"  Chosen outputs from SFT pairs: "
              f"{sum(len(v) for v in chosen_map.values())} for {len(chosen_map)} tasks")
        pairs = process_tasks(tasks, chosen_map, args.n_rejected)

    with open(OUT_FILE, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    from collections import Counter
    mode_counts = Counter(p["metadata"]["failure_mode"] for p in pairs)
    stats = {
        "timestamp":              datetime.utcnow().isoformat() + "Z",
        "mode":                   mode,
        "seed":                   args.seed,
        "n_rejected":             args.n_rejected,
        "tasks_used":             len(tasks),
        "tasks_failed":           failed,
        "total_pairs":            len(pairs),
        "out_file":               str(OUT_FILE),
        "failure_mode_breakdown": dict(mode_counts),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DONE] {len(pairs)} preference pairs ({mode}) → {OUT_FILE}")
    print(f"  Breakdown: {dict(mode_counts)}")
    if failed:
        print(f"  [WARN] {failed} pairs failed — check API quota or retry")
    print(f"\n  Next: python training/train_orpo.py")
    print(f"         python training/train_simpo.py")
    print(f"         python training/compare_methods.py")


if __name__ == "__main__":
    main()
