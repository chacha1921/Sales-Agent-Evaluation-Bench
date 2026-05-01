#!/usr/bin/env python3
"""
run_pipeline.py — Full dataset generation pipeline.

Steps:
  1. Re-generate multi_llm tasks (18 seeds × 5 variations = 90 tasks)
     - Bulk (adv=0.5): Gemini 2.0 Flash via GOOGLE_API_KEY
     - Hard (adv=1.0): Claude Sonnet via ANTHROPIC_API_KEY
  2. Merge all raw task files (trace_derived + programmatic + adversarial + multi_llm)
  3. Run LLM-as-a-judge filter (Claude Haiku) on every task
  4. Run three contamination checks; partition into train/dev/held_out
  5. Print cost summary and update training/cost_log.md

Usage:
  python generation/run_pipeline.py --live       # full live run
  python generation/run_pipeline.py --mock       # no API calls (regenerate with templates)
  python generation/run_pipeline.py --live --skip-gen   # skip generation, re-judge only
"""

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent


def _load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

_load_env()


def run(cmd: list, label: str) -> int:
    print(f"\n{'─'*60}")
    print(f"[pipeline] {label}")
    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  {status} — {elapsed:.1f}s")
    return result.returncode


def count_tasks() -> dict:
    counts = {}
    for split in ["train", "dev", "held_out"]:
        p = ROOT / "dataset" / "tenacious_bench_v0.1" / split / "tasks.jsonl"
        if p.exists():
            counts[split] = sum(1 for _ in p.open())
        else:
            counts[split] = 0
    return counts


def mode_breakdown() -> dict:
    by_mode = {}
    for split in ["train", "dev", "held_out"]:
        p = ROOT / "dataset" / "tenacious_bench_v0.1" / split / "tasks.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            t = json.loads(line)
            m = t.get("authoring_mode", "?")
            by_mode[m] = by_mode.get(m, 0) + 1
    return by_mode


def model_breakdown() -> dict:
    by_model = {}
    for split in ["train", "dev", "held_out"]:
        p = ROOT / "dataset" / "tenacious_bench_v0.1" / split / "tasks.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            t = json.loads(line)
            m = t.get("metadata", {}).get("generation_model", "?")
            by_model[m] = by_model.get(m, 0) + 1
    return by_model


def estimate_cost(n_gemini_gen: int, n_deepseek_gen: int, n_judge: int) -> dict:
    # Gemini 2.0 Flash gen: $0.075/M input, $0.30/M output (~300+250 tokens/call)
    g_cost = n_gemini_gen  * (300 * 0.075 + 250 * 0.30) / 1_000_000
    # DeepSeek Chat (OpenRouter): $0.14/M input, $0.28/M output (~300+250 tokens/call)
    d_cost = n_deepseek_gen * (300 * 0.14  + 250 * 0.28) / 1_000_000
    # Judge: mix of Gemini Flash + DeepSeek (~500+50 tokens/call) — estimate Gemini rate
    j_cost = n_judge * (500 * 0.075 + 50 * 0.30) / 1_000_000
    return {
        "gemini_generation":   round(g_cost, 4),
        "deepseek_generation": round(d_cost, 4),
        "judge_calls":         round(j_cost, 4),
        "total_estimated":     round(g_cost + d_cost + j_cost, 4),
    }


def append_cost_log(cost: dict, n_tasks: int, mode: str) -> None:
    log_path = ROOT / "training" / "cost_log.md"
    if not log_path.exists():
        return
    today = datetime.now().strftime("%Y-%m-%d")
    entry = (
        f"\n## Live Generation Run ({today})\n\n"
        f"| Item | Model | Calls (est.) | Cost (USD, est.) |\n"
        f"|---|---|---|---|\n"
        f"| multi_llm bulk generation | gemini/gemini-2.5-flash | 72 | ${cost['gemini_generation']:.4f} |\n"
        f"| multi_llm hard seed generation | deepseek/deepseek-chat (OpenRouter) | 18 | ${cost['deepseek_generation']:.4f} |\n"
        f"| judge_filter --live ({n_tasks} tasks) | gemini-2.5-flash + deepseek-chat | {n_tasks} | ${cost['judge_calls']:.4f} |\n"
        f"| **Run total** | | | **${cost['total_estimated']:.4f}** |\n"
    )
    with log_path.open("a") as f:
        f.write(entry)
    print(f"[pipeline] Cost entry appended to training/cost_log.md")


def main():
    parser = argparse.ArgumentParser(description="Full dataset generation pipeline")
    parser.add_argument("--live",       action="store_true", help="Use live LLM APIs")
    parser.add_argument("--mock",       action="store_true", help="Mock mode (no API calls)")
    parser.add_argument("--skip-gen",   action="store_true", help="Skip generation, re-judge only")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding similarity check")
    args = parser.parse_args()

    if args.live and args.mock:
        print("[ERROR] Cannot use --live and --mock together")
        sys.exit(1)
    if not args.live and not args.mock:
        print("[ERROR] Specify --live or --mock")
        sys.exit(1)

    mode_flag = "--live" if args.live else "--mock"
    mode_label = "live" if args.live else "mock"

    if args.live:
        missing = [k for k in ["GOOGLE_API_KEY", "OPENROUTER_API_KEY"] if not os.environ.get(k)]
        if missing:
            print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
            sys.exit(1)

    steps_failed = 0

    # ── Step 1: Generate multi_llm tasks ──────────────────────────────────────
    if not args.skip_gen:
        rc = run(
            [sys.executable, "generation/scripts/multi_llm_synthesis.py", mode_flag],
            f"Step 1 — multi_llm_synthesis ({mode_label}, 18 seeds × 5 = 90 tasks)",
        )
        steps_failed += rc != 0

    # ── Step 2: LLM-as-a-judge filter ─────────────────────────────────────────
    rc = run(
        [sys.executable, "generation/judge_filter.py", mode_flag],
        f"Step 2 — judge_filter ({mode_label})",
    )
    steps_failed += rc != 0

    # ── Step 3: Contamination checks + partition ───────────────────────────────
    contam_cmd = [sys.executable, "generation/contamination_check.py"]
    if args.skip_embed:
        contam_cmd.append("--skip-embedding")
    rc = run(contam_cmd, "Step 3 — contamination_check + partition")
    steps_failed += rc != 0

    # ── Step 4: Report ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("PIPELINE COMPLETE")
    print(f"  Steps failed: {steps_failed}")
    counts = count_tasks()
    total = sum(counts.values())
    print(f"  Tasks in dataset: {total} (train={counts['train']}, dev={counts['dev']}, held_out={counts['held_out']})")
    print(f"  By authoring mode: {mode_breakdown()}")
    print(f"  By generation model: {model_breakdown()}")

    # ── Step 5: Cost log ───────────────────────────────────────────────────────
    if args.live:
        cost = estimate_cost(n_gemini_gen=72, n_deepseek_gen=18, n_judge=total)
        print(f"\n  Estimated API cost this run:")
        print(f"    Gemini Flash generation (72 bulk):   ${cost['gemini_generation']:.4f}")
        print(f"    DeepSeek Chat generation (18 hard):  ${cost['deepseek_generation']:.4f}")
        print(f"    Judge calls ({total} tasks):          ${cost['judge_calls']:.4f}")
        print(f"    Total estimated:                      ${cost['total_estimated']:.4f}")
        append_cost_log(cost, total, mode_label)

    if steps_failed:
        print(f"\n  WARNING: {steps_failed} step(s) failed — check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
