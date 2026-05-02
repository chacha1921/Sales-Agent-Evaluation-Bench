#!/usr/bin/env python3
"""
scoring_evaluator.py — Machine-verifiable scorer for Tenacious-Bench v0.1.

Every checker function returns float [0,1] with no human in the loop.
LLM-judge calls use a deterministic seed and can be bypassed with --mock-llm.

Usage:
    # Run against 3 hand-built dummy tasks (no API key needed)
    python evaluation/scoring_evaluator.py --demo

    # Score a split
    python evaluation/scoring_evaluator.py --split dev --mock-llm

    # Full eval with live LLM judge (requires ANTHROPIC_API_KEY)
    python evaluation/scoring_evaluator.py --split dev

    # Compare two agents on held-out
    python evaluation/scoring_evaluator.py --split held_out --results-a results_week10.json --results-b results_trained.json
"""

import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path

def _load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

_load_env()

import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
BANNED_PHRASES_FILE = ROOT / "dataset" / "banned_phrases.txt"


def _load_banned_phrases() -> list[str]:
    if BANNED_PHRASES_FILE.exists():
        return [p.strip().lower() for p in BANNED_PHRASES_FILE.read_text().splitlines() if p.strip()]
    return []


BANNED_PHRASES = _load_banned_phrases()


# ─── Checker Functions ────────────────────────────────────────────────────────
#
# RUBRIC CALIBRATION (applies to all seven dimensions below)
# Each checker returns float [0, 1]. Aggregate score = weighted_sum * 5 → [0, 5].
#
# What different aggregate scores look like:
#   5.0 (perfect)  — all constraints met, specific signal referenced, zero banned phrases,
#                    calendly link present, no pricing, tone is direct/human/signal-led
#   3.5 (pass)     — minor constraint miss or weak signal reference; banned phrases absent;
#                    CTA present; overall message is on-brand and coherent
#   2.0 (partial)  — meaningful failure on ≥1 dimension: signal completely absent, or
#                    banned phrase present, or no CTA in an email_outreach task
#   1.0 (fail)     — multiple dimension failures: generic template, banned phrases, no CTA,
#                    and/or no signal; or word count >2× limit
#
# Per-dimension calibration:
#   signal_grounding_fn:  1.0 = explicit verifiable signal (funding amount, layoff %, etc.)
#                         0.5 = company/name mention only, no signal
#                         0.0 = no connection to context whatsoever
#   banned_phrase_fn:     1.0 = zero prohibited phrases; 0.0 = any hit (binary)
#   cta_checker_fn:       1.0 = calendly/calendar link or explicit book-a-time request; 0.0 = absent
#   word_count_fn:        1.0 = at or under limit; linear decay to 0.0 at 2× the limit
#   pricing_mention_fn:   1.0 = no pricing language; 0.0 = any pricing/cost reference (binary)
#   objection_ack_fn:     1.0 = empathetic acknowledgment before any pivot; 0.0 = direct pivot
#   tone_checker_fn:      1.0 = direct, signal-led, non-pushy, concise, human
#                         0.5 = acceptable tone with one tier-2 jargon hit
#                         0.0 = any tier-1 phrase (absolute brand violation per Style Guide v2)


def signal_grounding_fn(output: str, context: str) -> float:
    """
    Returns 1.0 if output references at least one verifiable signal from context.
    Signals: funding rounds, headcount numbers, job posting titles, layoff events,
    specific dates/quarters, named companies/roles when present in context.
    Returns 0.5 for partial match (named entity overlap), 0.0 for no match.
    """
    output_lower = output.lower()
    context_lower = context.lower()

    signal_patterns = [
        r'\bseries [abcde]\b',
        r'\$\d+[mk]?\b',
        r'\b\d{1,3}[,\-]\d{3}\b',
        r'\b\d+[\s\-]?(?:person|employee|member|seat)',
        r'\b(?:head of|vp of|director of|chief)\b',
        r'\b(?:funding|raised|round|invest|crunchbase)\b',
        r'\blayoff|restructur|reorg|headcount\b',
        r'\b20\d{2}[-\s]?q[1-4]\b',
        r'\bjob posting|hiring|open role|new hire\b',
        r'\blinkedin\b',
    ]

    for pattern in signal_patterns:
        in_output = re.search(pattern, output_lower)
        in_context = re.search(pattern, context_lower)
        if in_output and in_context:
            return 1.0

    # Soft check: named entities from context appearing in output
    context_names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', context))
    output_names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', output))
    stopwords = {'The', 'This', 'That', 'Our', 'Your', 'With', 'From', 'Have', 'Will'}
    overlap = (context_names & output_names) - stopwords
    if len(overlap) >= 2:
        return 0.5

    return 0.0


def banned_phrase_fn(output: str) -> float:
    """Returns 1.0 if output contains zero banned phrases, 0.0 if any found."""
    output_lower = output.lower()
    violations = [p for p in BANNED_PHRASES if p in output_lower]
    if violations:
        return 0.0
    return 1.0


def cta_checker_fn(output: str) -> float:
    """Returns 1.0 if a calendar link, placeholder, or explicit booking CTA is present."""
    patterns = [
        r'calendly\.com',
        r'cal\.com',
        r'\[calendly[_\s]?link\]',
        r'\[calendar[_\s]?link\]',
        r'book a time',
        r'schedule a (?:call|meeting|chat)',
        r'grab \d+[\s-]?min',
        r'pick a time',
        r'find a time',
        r'https?://\S*(?:meet|cal|schedule)\S*',
    ]
    output_lower = output.lower()
    for p in patterns:
        if re.search(p, output_lower):
            return 1.0
    return 0.0


def word_count_fn(output: str, max_words: int) -> float:
    """
    Returns 1.0 if word count <= max_words.
    Applies linear penalty beyond limit: score = max(0, 1 - overage_ratio).
    """
    count = len(output.split())
    if count <= max_words:
        return 1.0
    overage_ratio = (count - max_words) / max_words
    return max(0.0, 1.0 - overage_ratio)


def pricing_mention_fn(output: str) -> float:
    """Returns 1.0 if no pricing/cost language is present, 0.0 otherwise."""
    patterns = [
        r'\$\d',
        r'\bpric(?:e|ing|ed)\b',
        r'\bcost(?:s|ing)?\b',
        r'\bper seat\b',
        r'\bper month\b',
        r'\bper year\b',
        r'\bsubscription\b',
        r'\btier\b',
        r'\bplan\b',
        r'\bfree trial\b',
    ]
    output_lower = output.lower()
    for p in patterns:
        if re.search(p, output_lower):
            return 0.0
    return 1.0


def objection_ack_fn(output: str, mock: bool = False) -> float:
    """
    Checks whether the output acknowledges the prospect's objection before pivoting.
    Mock mode: keyword heuristic. Live mode: LLM binary judge.
    """
    if mock:
        ack_keywords = [
            'understand', 'makes sense', 'fair point', 'hear you',
            'appreciate', 'noted', 'that said', 'even so', 'of course',
            'totally get', 'you\'re right', 'that\'s fair',
        ]
        output_lower = output.lower()
        return 1.0 if any(k in output_lower for k in ack_keywords) else 0.0

    return _llm_judge_binary(
        output,
        "Does this B2B sales response first acknowledge the prospect's objection "
        "empathetically before making any counter-argument or pivot? Answer YES or NO only.",
    )


def tone_checker_fn(output: str, mock: bool = False) -> float:
    """
    Scores output on 5 Tenacious tone markers (1–5 each), normalised to [0,1].
    Mock mode: two-tier penalty heuristic. Live mode: LLM multi-dim judge.

    IRA revision (2026-04-29): tier-1 phrases now return 0.0 immediately.
    Original gradual-penalty approach gave PASS to outputs with "just checking in"
    and "My name is X" — both of which human raters consistently scored as FAIL.
    κ rose from 0.662 to 1.000 after this change. See dataset/inter_rater_agreement.md.
    """
    if mock:
        # Tier 1: immediate FAIL — Style Guide v2 absolute violations
        tier1 = [
            # Formulaic openers
            "just checking in", "just following up", "circle back", "circling back",
            "touching base", "i hope this email finds you well", "i hope this finds you well",
            "i wanted to reach out", "my name is", "i'm reaching out from",
            # Offshore/agency clichés [SG]
            "top talent", "a-players", "rockstar", "ninja", "wizard", "world-class",
            # Fake urgency [SG]
            "you'll regret missing this", "don't miss out",
            # Passive-aggressive [SG]
            "per my last email", "i noticed you're a",
            # Black-box claims [SG]
            "our proprietary", "our ai-powered",
            # Performative brevity [SG]
            "i'll keep this brief",
            # Vague asks [SG]
            "quick question", "quick chat",
        ]
        # Tier 2: gradual penalty — jargon that weakens but doesn't always fail
        tier2 = [
            "act now", "limited time", "last chance", "skyrocket", "supercharge",
            "synergy", "leverage", "ecosystem", "game-changer", "revolutionary",
            "disruptor", "paradigm shift", "gold standard",
        ]
        output_lower = output.lower()
        if any(p in output_lower for p in tier1):
            return 0.0
        penalty = sum(1 for p in tier2 if p in output_lower)
        raw = max(1, 5 - penalty)
        return (raw - 1) / 4.0

    return _llm_judge_score(
        output,
        "Score this B2B sales email on EACH of these 5 Tenacious tone markers from 1 to 5:\n"
        "1. Direct — no filler phrases or preamble\n"
        "2. Signal-led — opens with or references a prospect insight\n"
        "3. Non-pushy — no urgency manipulation or pressure tactics\n"
        "4. Concise — every sentence earns its place\n"
        "5. Human — sounds like a real person, not a template\n\n"
        "Reply with exactly this format: SCORES: d,d,d,d,d",
    )


# ─── LLM Judge Calls ─────────────────────────────────────────────────────────

def _gemini_call(user_content: str, max_tokens: int = 20) -> str:
    """Shared Gemini Flash call for LLM judge dimensions. Temperature 0 for reproducibility."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set — run with --mock-llm or set GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_content,
        config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens),
    )
    return (response.text or "").strip()


def _llm_judge_binary(output: str, prompt: str) -> float:
    """Call Gemini Flash for binary YES/NO question. Returns 1.0 or 0.0."""
    try:
        answer = _gemini_call(f"{prompt}\n\nText to evaluate:\n{output}", max_tokens=5).upper()
        return 1.0 if answer.startswith("YES") else 0.0
    except Exception as e:
        print(f"[WARN] LLM judge error: {e}. Returning 0.5.", file=sys.stderr)
        return 0.5


def _llm_judge_score(output: str, prompt: str) -> float:
    """Call Gemini Flash for multi-dimension scoring. Returns normalised [0,1]."""
    try:
        text = _gemini_call(f"{prompt}\n\nText to evaluate:\n{output}", max_tokens=20)
        match = re.search(r'SCORES:\s*([\d,\s]+)', text)
        if match:
            scores = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]
            scores = [max(1, min(5, s)) for s in scores]
            if scores:
                mean = sum(scores) / len(scores)
                return (mean - 1) / 4.0
        return 0.5
    except Exception as e:
        print(f"[WARN] LLM judge error: {e}. Returning 0.5.", file=sys.stderr)
        return 0.5


# ─── Task Scorer ─────────────────────────────────────────────────────────────

def _parse_max_words(constraints: list[str]) -> int:
    for c in constraints:
        m = re.search(r'under (\d+) words?', c, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 500


CHECKER_REGISTRY = {
    "signal_grounding_fn": None,
    "tone_checker_fn": None,
    "banned_phrase_fn": None,
    "cta_checker_fn": None,
    "word_count_fn": None,
    "pricing_mention_fn": None,
    "objection_ack_fn": None,
}


def score_task(task: dict, candidate_output: str, mock_llm: bool = False) -> dict:
    """
    Apply all rubric dimensions to candidate_output.
    Returns per-dimension scores and weighted aggregate on [0, 5].
    """
    rubric = task["ground_truth"]["dimensions"]
    context = task["input"].get("context", "")
    constraints = task["input"].get("constraints", [])
    max_words = _parse_max_words(constraints)

    checker_map = {
        "signal_grounding_fn":  lambda: signal_grounding_fn(candidate_output, context),
        "tone_checker_fn":       lambda: tone_checker_fn(candidate_output, mock=mock_llm),
        "banned_phrase_fn":      lambda: banned_phrase_fn(candidate_output),
        "cta_checker_fn":        lambda: cta_checker_fn(candidate_output),
        "word_count_fn":         lambda: word_count_fn(candidate_output, max_words),
        "pricing_mention_fn":    lambda: pricing_mention_fn(candidate_output),
        "objection_ack_fn":      lambda: objection_ack_fn(candidate_output, mock=mock_llm),
    }

    dimension_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for dim_name, dim_config in rubric.items():
        checker_name = dim_config["checker"]
        weight = dim_config["weight"]
        fn = checker_map.get(checker_name)
        if fn is None:
            print(f"[WARN] Unknown checker '{checker_name}' — scoring 0.5", file=sys.stderr)
            raw = 0.5
        else:
            raw = fn()

        raw = max(0.0, min(1.0, raw))
        dimension_scores[dim_name] = {
            "raw_0_1": round(raw, 4),
            "scaled_0_5": round(raw * 5, 4),
            "weight": weight,
            "weighted_contribution": round(raw * weight * 5, 4),
        }
        weighted_sum += raw * weight
        total_weight += weight

    aggregate = (weighted_sum / total_weight) * 5 if total_weight > 0 else 0.0
    threshold = task["ground_truth"].get("passing_threshold", 3.5)

    return {
        "task_id": task["task_id"],
        "split": task.get("split", "unknown"),
        "aggregate_score": round(aggregate, 4),
        "passed": aggregate >= threshold,
        "passing_threshold": threshold,
        "dimensions": dimension_scores,
        "word_count": len(candidate_output.split()),
        "mock_llm": mock_llm,
    }


# ─── Aggregation & Statistics ─────────────────────────────────────────────────

def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> dict:
    """Paired bootstrap 95% CI over a list of aggregate scores."""
    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    arr = np.array(scores)
    boot_means = [
        float(np.mean(np.random.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "ci_lower": round(float(np.percentile(boot_means, 100 * alpha / 2)), 4),
        "ci_upper": round(float(np.percentile(boot_means, 100 * (1 - alpha / 2))), 4),
        "n": len(scores),
    }


def delta_significance(scores_a: list[float], scores_b: list[float],
                       n_bootstrap: int = 1000) -> dict:
    """
    Paired bootstrap test: is mean(b) - mean(a) significantly > 0?
    Returns delta, p-value estimate, and 95% CI of the delta.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must be the same length for paired bootstrap.")
    diffs = np.array(scores_b) - np.array(scores_a)
    observed_delta = float(np.mean(diffs))
    boot_deltas = [
        float(np.mean(np.random.choice(diffs, size=len(diffs), replace=True)))
        for _ in range(n_bootstrap)
    ]
    p_value = float(np.mean(np.array(boot_deltas) <= 0))
    return {
        "delta": round(observed_delta, 4),
        "p_value": round(p_value, 4),
        "significant_p05": p_value < 0.05,
        "ci_lower": round(float(np.percentile(boot_deltas, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_deltas, 97.5)), 4),
        "n_pairs": len(scores_a),
    }


# ─── Split Runner ─────────────────────────────────────────────────────────────

def run_on_split(split: str, mock_llm: bool = False) -> list[dict]:
    tasks_file = ROOT / "dataset" / "tenacious_bench_v0.1" / split / "tasks.jsonl"
    if not tasks_file.exists():
        print(f"[ERROR] {tasks_file} not found.", file=sys.stderr)
        return []

    results = []
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            candidate = task.get("candidate_output", "")
            result = score_task(task, candidate, mock_llm=mock_llm)
            results.append(result)
            print(f"  {result['task_id']}: {result['aggregate_score']:.2f}/5.0 "
                  f"({'PASS' if result['passed'] else 'FAIL'})")

    return results


# ─── Demo Mode ────────────────────────────────────────────────────────────────

DUMMY_TASKS = [
    {
        "task_id": "DEMO-001",
        "split": "demo",
        "authoring_mode": "trace_derived",
        "difficulty": "medium",
        "input": {
            "context": "Prospect: Sarah Chen, VP of Revenue at Lattice. Series C $45M 2023-Q2. LinkedIn post 3 days ago about sales rep ramp time.",
            "task_type": "email_outreach",
            "constraints": ["under 120 words", "include [CALENDLY_LINK]"],
        },
        "ground_truth": {
            "type": "rubric",
            "dimensions": {
                "signal_grounding": {"weight": 0.30, "checker": "signal_grounding_fn", "description": ""},
                "tone_compliance":   {"weight": 0.25, "checker": "tone_checker_fn",    "description": ""},
                "banned_phrase_absent": {"weight": 0.15, "checker": "banned_phrase_fn", "description": ""},
                "cta_present":       {"weight": 0.15, "checker": "cta_checker_fn",     "description": ""},
                "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn", "description": ""},
            },
            "aggregate": "weighted_sum",
            "passing_threshold": 3.5,
        },
        "candidate_output": (
            "Hi Sarah,\n\n"
            "Your LinkedIn post about sales rep ramp time caught my eye — it's a problem "
            "we've seen a lot since Lattice's Series C. Most teams at your stage find the "
            "bottleneck is onboarding content, not headcount.\n\n"
            "Tenacious cuts average ramp from 90 to 45 days. Worth a 20-min look?\n\n"
            "[CALENDLY_LINK]\n\n"
            "— Alex"
        ),
    },
    {
        "task_id": "DEMO-002",
        "split": "demo",
        "authoring_mode": "programmatic",
        "difficulty": "hard",
        "input": {
            "context": "Prospect: Mark Rivera, Head of Sales Ops at Meridian SaaS (500 employees). Objection: 'We already have a solution.'",
            "task_type": "objection_handling",
            "constraints": ["under 100 words", "acknowledge objection before pivoting"],
        },
        "ground_truth": {
            "type": "rubric",
            "dimensions": {
                "signal_grounding":    {"weight": 0.20, "checker": "signal_grounding_fn", "description": ""},
                "objection_acknowledged": {"weight": 0.30, "checker": "objection_ack_fn", "description": ""},
                "tone_compliance":     {"weight": 0.25, "checker": "tone_checker_fn",    "description": ""},
                "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn",  "description": ""},
                "banned_phrase_absent": {"weight": 0.10, "checker": "banned_phrase_fn",  "description": ""},
            },
            "aggregate": "weighted_sum",
            "passing_threshold": 3.0,
        },
        "candidate_output": (
            "That makes sense — most teams at Meridian's scale have something in place. "
            "The question we usually hear is whether it's handling the RevOps consolidation "
            "after your recent restructure, or just the original use case it was bought for.\n\n"
            "Happy to be proven wrong in 15 minutes. [CALENDLY_LINK]"
        ),
    },
    {
        "task_id": "DEMO-003",
        "split": "demo",
        "authoring_mode": "adversarial",
        "difficulty": "hard",
        "input": {
            "context": "Prospect: James Park, CEO of bootstrapped 12-person agency. LinkedIn: Head of Sales job posted 2 days ago.",
            "task_type": "email_outreach",
            "constraints": ["under 80 words", "reference Head of Sales job posting", "no pricing", "no 'leverage'"],
        },
        "ground_truth": {
            "type": "rubric",
            "dimensions": {
                "signal_grounding":  {"weight": 0.30, "checker": "signal_grounding_fn", "description": ""},
                "pricing_absent":    {"weight": 0.25, "checker": "pricing_mention_fn",  "description": ""},
                "banned_phrase_absent": {"weight": 0.20, "checker": "banned_phrase_fn", "description": ""},
                "cta_present":       {"weight": 0.10, "checker": "cta_checker_fn",     "description": ""},
                "word_count_within_limit": {"weight": 0.15, "checker": "word_count_fn", "description": ""},
            },
            "aggregate": "weighted_sum",
            "passing_threshold": 3.5,
        },
        "candidate_output": (
            "Hi James,\n\n"
            "Saw you're hiring a Head of Sales — that's a big hire for a 12-person team. "
            "A lot of founders at that stage wish they'd put some tooling in place before "
            "the first rep started.\n\n"
            "Happy to share what's worked. [CALENDLY_LINK]\n\n— Alex"
        ),
    },
]


def run_demo(mock_llm: bool = True) -> None:
    print("\n" + "=" * 60)
    print("Tenacious-Bench v0.1 — Demo Run (3 hand-built dummy tasks)")
    print(f"Mock LLM: {mock_llm}")
    print("=" * 60 + "\n")

    results = []
    for task in DUMMY_TASKS:
        result = score_task(task, task["candidate_output"], mock_llm=mock_llm)
        results.append(result)

        print(f"Task: {result['task_id']}")
        print(f"  Aggregate : {result['aggregate_score']:.2f} / 5.0  "
              f"({'PASS ✓' if result['passed'] else 'FAIL ✗'})  "
              f"(threshold {result['passing_threshold']})")
        for dim, data in result["dimensions"].items():
            bar = "█" * int(data["raw_0_1"] * 10) + "░" * (10 - int(data["raw_0_1"] * 10))
            print(f"  {dim:<30} {bar}  {data['raw_0_1']:.2f}  (w={data['weight']})")
        print()

    scores = [r["aggregate_score"] for r in results]
    ci = bootstrap_ci(scores)
    print("-" * 60)
    print(f"Summary: mean={ci['mean']:.2f}, CI=[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}], n={ci['n']}")
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    print(f"Pass rate: {pass_rate:.0%}  ({sum(1 for r in results if r['passed'])}/{len(results)})")
    print("=" * 60 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 scoring evaluator")
    parser.add_argument("--demo", action="store_true",
                        help="Run against 3 hand-built dummy tasks (no API key needed)")
    parser.add_argument("--split", choices=["train", "dev", "held_out"],
                        help="Score a dataset split")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use heuristic fallbacks instead of live LLM judge calls")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this path")
    parser.add_argument("--results-a", type=str, default=None,
                        help="Path to baseline results JSON (for Delta comparison)")
    parser.add_argument("--results-b", type=str, default=None,
                        help="Path to trained results JSON (for Delta comparison)")
    args = parser.parse_args()

    if args.demo:
        run_demo(mock_llm=True)
        return

    if args.results_a and args.results_b:
        with open(args.results_a) as f:
            res_a = json.load(f)
        with open(args.results_b) as f:
            res_b = json.load(f)
        scores_a = [r["aggregate_score"] for r in res_a]
        scores_b = [r["aggregate_score"] for r in res_b]
        delta = delta_significance(scores_a, scores_b)
        print(json.dumps(delta, indent=2))
        return

    if not args.split:
        parser.print_help()
        sys.exit(1)

    print(f"\nScoring split: {args.split}  (mock_llm={args.mock_llm})")
    results = run_on_split(args.split, mock_llm=args.mock_llm)

    if not results:
        print("No tasks scored.")
        return

    scores = [r["aggregate_score"] for r in results]
    ci = bootstrap_ci(scores)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)

    print(f"\nResults ({args.split}):")
    print(f"  Tasks scored : {ci['n']}")
    print(f"  Mean score   : {ci['mean']:.4f} / 5.0")
    print(f"  95% CI       : [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Std dev      : {ci['std']:.4f}")
    print(f"  Pass rate    : {pass_rate:.1%}")

    if args.output:
        out = {
            "split": args.split,
            "mock_llm": args.mock_llm,
            "summary": ci,
            "pass_rate": pass_rate,
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
