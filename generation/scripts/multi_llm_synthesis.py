#!/usr/bin/env python3
"""
multi_llm_synthesis.py — Mode 3: Multi-LLM Synthesis

Generates tasks using LLM seed prompts. In --mock mode (default), seeds are
expanded via deterministic templates — no API calls. In --live mode, seeds
are sent to a frontier model (hard seeds) and a dev-tier model (bulk).

Authoring record:
  Seed count : 18 scenario seeds × 5 variations = 90 tasks
  Task IDs   : TB-0106 → TB-0195
  Model route (mock) : template expansion (no LLM)
  Model route (live) :
    Hard seed variants (adv_weight=1.0) → deepseek/deepseek-chat    via OpenRouter
    Bulk variants (adv_weight=0.5)      → gemini/gemini-2.0-flash   via Google GenAI
  Leakage guard : generation_model logged per task; judge_model must differ
                  gemini tasks → judged by claude-haiku (different family)

Usage:
  python generation/scripts/multi_llm_synthesis.py --mock   (no API calls)
  python generation/scripts/multi_llm_synthesis.py --live   (requires ANTHROPIC_API_KEY + GOOGLE_API_KEY)
"""

import json
import argparse
import random
import sys
import os
import urllib.request
from pathlib import Path

def _load_env():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

_load_env()

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "generation"))
from task_templates import make_task, TODAY

OUT_DIR     = ROOT / "generation" / "raw_tasks"
OUT_FILE    = OUT_DIR / "multi_llm.jsonl"
PROMPTS_DIR = ROOT / "generation" / "prompts"
START_ID    = 106   # TB-0106 → TB-0195 (18 seeds × 5 variations = 90 tasks)

# ── System prompt loaded from generation/prompts/generation_system_prompt.md ──
def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()

GENERATION_SYSTEM_PROMPT = _load_prompt("generation_system_prompt.md")

# ── 12 seed scenarios ──────────────────────────────────────────────────────────
SEEDS = [
    dict(id="S01", segment="series_b",  role="Head of Sales",        company="Narvar",      industry="logistics SaaS",
         signal="Series B $40M closed 2025-Q2", src="crunchbase_odm", win="2025-Q2",
         pain="no structured sales playbook after Series B"),
    dict(id="S02", segment="enterprise", role="CRO",                  company="PivotDesk",   industry="real estate tech",
         signal="cut 25% of sales headcount 2026-Q1", src="layoffs_fyi", win="2026-Q1",
         pain="remaining reps have no ramp support after layoffs"),
    dict(id="S03", segment="smb",        role="CEO",                   company="Folio",        industry="content agency",
         signal="posted VP of Sales job LinkedIn 3 days ago", src="synthetic", win=None,
         pain="founder closing all deals personally, not scalable"),
    dict(id="S04", segment="series_b",  role="VP of Revenue",        company="Solvvy",       industry="AI customer support",
         signal="launched enterprise tier 2025-Q3 per press release", src="synthetic", win="2025-Q3",
         pain="enterprise ramp taking 4x longer than SMB ramp"),
    dict(id="S05", segment="enterprise", role="Head of Revenue Ops",  company="Acuity",       industry="insurance tech",
         signal="missed Q3 targets by 16% per investor call", src="synthetic", win="2025-Q3",
         pain="RevOps has 4 systems that don't talk to each other"),
    dict(id="S06", segment="smb",        role="Founder",               company="Sparq",        industry="dev tools",
         signal="posted Head of Growth job LinkedIn yesterday", src="synthetic", win=None,
         pain="product-led growth hitting ceiling, need sales motion"),
    dict(id="S07", segment="series_b",  role="Director of Sales",    company="Lumen",        industry="HR tech",
         signal="$25M Series A closed 2025-Q1", src="crunchbase_odm", win="2025-Q1",
         pain="sales team grew 3x but process is still ad hoc"),
    dict(id="S08", segment="enterprise", role="VP of Sales",          company="StrataBridge", industry="supply chain SaaS",
         signal="cut 20% of enablement team 2026-Q1", src="layoffs_fyi", win="2026-Q1",
         pain="ramp programs eliminated, new reps have no support"),
    dict(id="S09", segment="smb",        role="Co-founder & CEO",     company="Drift",        industry="B2B marketplace",
         signal="posted a Sales Lead job LinkedIn 2 days ago", src="synthetic", win=None,
         pain="first sales hire needs structured process from day one"),
    dict(id="S10", segment="series_b",  role="VP of Business Dev",   company="Pulsar",       industry="analytics SaaS",
         signal="raised $15M Seed extension 2025-Q4", src="crunchbase_odm", win="2025-Q4",
         pain="BD team has 3 AEs sharing one untracked spreadsheet"),
    dict(id="S11", segment="enterprise", role="Chief Revenue Officer", company="Meridian AI", industry="enterprise AI platform",
         signal="missed Q4 NRR target by 9%", src="synthetic", win="2025-Q4",
         pain="expansion ARR declining as enterprise AEs lack upsell playbooks"),
    dict(id="S12", segment="smb",        role="CEO",                   company="Bloom",        industry="e-commerce enablement",
         signal="posted Head of Sales job LinkedIn this morning", src="synthetic", win=None,
         pain="CEO-led sales closed $800k ARR but can't scale past $1.2M"),
    # ── 6 additional seeds (S13–S18) for Gemini live generation ──────────────
    dict(id="S13", segment="series_b",  role="VP of Sales",           company="Cohere",       industry="enterprise AI",
         signal="Series B $125M closed 2025-Q3", src="crunchbase_odm", win="2025-Q3",
         pain="post-funding AE team doubled overnight with no onboarding process"),
    dict(id="S14", segment="enterprise", role="Head of Revenue Enablement", company="Pendo",  industry="product analytics",
         signal="cut 18% of go-to-market headcount 2026-Q1", src="layoffs_fyi", win="2026-Q1",
         pain="surviving reps covering 2x territories with no revised playbook"),
    dict(id="S15", segment="smb",        role="Co-founder",            company="Coda",         industry="collaborative docs",
         signal="posted first Sales Development Rep job LinkedIn yesterday", src="synthetic", win=None,
         pain="product-led acquisition stalling, need outbound motion from scratch"),
    dict(id="S16", segment="series_b",  role="CRO",                   company="Ironclad",     industry="contract lifecycle",
         signal="$100M Series D closed 2025-Q4", src="crunchbase_odm", win="2025-Q4",
         pain="contract sales cycle 3x longer than industry average post-expansion"),
    dict(id="S17", segment="enterprise", role="SVP Sales",             company="Veeva",        industry="life sciences SaaS",
         signal="missed Q2 quota by 22% per analyst call 2025-Q2", src="synthetic", win="2025-Q2",
         pain="enterprise AEs pitching product features instead of business outcomes"),
    dict(id="S18", segment="smb",        role="Head of Business Dev",  company="Linear",       industry="project management",
         signal="posted Account Executive role LinkedIn 2 days ago", src="synthetic", win=None,
         pain="engineering-led culture, no formal sales process or talk tracks"),
]

VARIATION_CONFIGS = [
    dict(v=1, task_type="email_outreach",      word_limit=120, adv=0.5,
         constraints_suffix=["reference the signal explicitly", "include [CALENDLY_LINK]", "do not mention pricing"]),
    dict(v=2, task_type="email_outreach",      word_limit=80,  adv=0.5,
         constraints_suffix=["reference the signal", "include [CALENDLY_LINK]", "no pricing", "end with a question"]),
    dict(v=3, task_type="follow_up",           word_limit=70,  adv=0.5,
         constraints_suffix=["5 days no reply", "no 'just checking in'", "include [CALENDLY_LINK]"]),
    dict(v=4, task_type="objection_handling",  word_limit=100, adv=0.5,
         constraints_suffix=["acknowledge before pivoting", "reference the signal", "do not mention pricing"]),
    dict(v=5, task_type="email_outreach",      word_limit=100, adv=1.0,
         constraints_suffix=["reference the signal", "do NOT use 'leverage' or 'synergy'",
                              "do NOT mention pricing", "include [CALENDLY_LINK]"]),
]

DIFFICULTY_MAP = {"series_b": "medium", "enterprise": "hard", "smb": "easy"}


def build_mock_tasks():
    tasks = []
    tid = START_ID
    for seed in SEEDS:
        for vc in VARIATION_CONFIGS:
            context = (
                f"Prospect: {seed['role']} at {seed['company']} "
                f"({seed['segment'].replace('_', ' ').title()}, {seed['industry']}). "
                f"Signal: {seed['company']} {seed['signal']}. "
                f"Pain: {seed['pain']}."
            )
            constraints = [f"under {vc['word_limit']} words"] + vc["constraints_suffix"]
            gen_model = "deepseek/deepseek-chat" if vc["adv"] == 1.0 else "gemini/gemini-2.0-flash"
            tasks.append(make_task(
                tid=tid,
                authoring_mode="multi_llm",
                source_traces=[],
                difficulty="hard" if vc["adv"] == 1.0 else DIFFICULTY_MAP[seed["segment"]],
                context=context,
                task_type=vc["task_type"],
                constraints=constraints,
                segment=seed["segment"],
                failure_tag="tone_drift" if vc["adv"] == 1.0 else "signal_missing",
                adv_weight=vc["adv"],
                signal_source=seed["src"],
                signal_time_window=seed["win"],
                generation_model=gen_model,
            ))
            tid += 1
    return tasks


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini Flash for bulk task generation. Returns raw text response."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=400),
    )
    return response.text.strip()


def _call_openrouter(model_id: str, system_prompt: str, user_prompt: str) -> str:
    """Call a cheap model via OpenRouter for adversarial (hard) task generation."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def build_live_tasks(tasks_so_far):
    """Call LLM APIs to generate tasks. Requires OPENROUTER_API_KEY + GOOGLE_API_KEY."""
    tasks = list(tasks_so_far)
    tid = START_ID
    mock_tasks = build_mock_tasks()

    for seed in SEEDS:
        for vc in VARIATION_CONFIGS:
            seed_prompt = (
                f"Scenario: {seed['role']} at {seed['company']} ({seed['segment']}, {seed['industry']}). "
                f"Signal: {seed['signal']}. Pain: {seed['pain']}.\n"
                f"Task type: {vc['task_type']}. Word limit: {vc['word_limit']}. "
                f"Adversarial: {vc['adv'] == 1.0}.\n"
                f"Additional constraints: {', '.join(vc['constraints_suffix'])}"
            )
            is_adversarial = vc["adv"] == 1.0
            # Hard seeds: DeepSeek Chat via OpenRouter (cheap, good quality adversarial)
            # Bulk seeds: Gemini Flash (very cheap, different model family for leakage prevention)
            HARD_MODEL = "deepseek/deepseek-chat"
            gen_model_id = HARD_MODEL if is_adversarial else "gemini/gemini-2.0-flash"
            try:
                if is_adversarial:
                    raw_text = _call_openrouter(HARD_MODEL, GENERATION_SYSTEM_PROMPT, seed_prompt)
                else:
                    raw_text = _call_gemini(GENERATION_SYSTEM_PROMPT, seed_prompt)

                # Strip markdown code fences if present
                raw_text = raw_text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]

                raw = json.loads(raw_text)
                task = make_task(
                    tid=tid,
                    authoring_mode="multi_llm",
                    source_traces=[],
                    difficulty=raw.get("difficulty", DIFFICULTY_MAP[seed["segment"]]),
                    context=raw.get("context", ""),
                    task_type=raw.get("task_type", vc["task_type"]),
                    constraints=raw.get("constraints", [f"under {vc['word_limit']} words"]),
                    segment=seed["segment"],
                    failure_tag="tone_drift" if is_adversarial else "signal_missing",
                    adv_weight=vc["adv"],
                    signal_source=seed["src"],
                    signal_time_window=seed["win"],
                    generation_model=gen_model_id,
                )
                tasks.append(task)
                print(f"  Generated {task['task_id']} via {gen_model_id}")
            except Exception as e:
                print(f"  [WARN] Seed {seed['id']} v{vc['v']} failed: {e} — mock fallback")
                fallback = mock_tasks[tid - START_ID]
                tasks.append(fallback)
            tid += 1

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM synthesis task generation")
    parser.add_argument("--mock",    action="store_true", default=True, help="Use mock mode (no API calls)")
    parser.add_argument("--live",    action="store_true", help="Call LLM APIs (overrides --mock)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if args.live:
        args.mock = False

    random.seed(args.seed)
    print(f"[multi_llm] seed={args.seed}  prompt=generation/prompts/generation_system_prompt.md")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mock:
        tasks = build_mock_tasks()
        gen_note = "mock (template expansion)"
    else:
        tasks = build_live_tasks([])
        gen_note = "live (LLM API calls)"

    if args.dry_run:
        print(f"[dry-run] Would write {len(tasks)} tasks ({gen_note}) → {OUT_FILE}")
        return

    with open(OUT_FILE, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    segs = {}
    for t in tasks:
        s = t["metadata"]["tenacious_segment"]
        segs[s] = segs.get(s, 0) + 1
    print(f"[multi_llm] Wrote {len(tasks)} tasks ({gen_note}) → {OUT_FILE}")
    print(f"  Segments: {segs}")


if __name__ == "__main__":
    main()
