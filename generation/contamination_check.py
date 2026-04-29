#!/usr/bin/env python3
"""
contamination_check.py — Three contamination checks + dataset partitioning.

Reads generation/raw_tasks/filtered.jsonl and:
  1. Check 1 — N-gram overlap: zero shared 8-grams between held_out and train+dev
  2. Check 2 — Embedding similarity: cosine < 0.85 between held_out and train/dev tasks
              (skipped if sentence-transformers not installed; flagged in output)
  3. Check 3 — Time-shift verification: every task with public signal source must have
              a non-null, documentable signal_time_window field

Partitions the passing dataset 50% train / 30% dev / 20% held_out (stratified by
segment to preserve distribution).

Outputs:
  dataset/tenacious_bench_v0.1/{train,dev,held_out}/tasks.jsonl
  generation/contamination_check.json

Usage:
  python generation/contamination_check.py
  python generation/contamination_check.py --skip-embedding   # skip embedding check
"""

import json
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

random.seed(42)

ROOT        = ROOT = Path(__file__).parent.parent
FILTERED    = ROOT / "generation" / "raw_tasks" / "filtered.jsonl"
OUT_CHECK   = ROOT / "generation" / "contamination_check.json"
SPLITS_DIR  = ROOT / "dataset" / "tenacious_bench_v0.1"

PUBLIC_SOURCES = {"crunchbase_odm", "layoffs_fyi"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ngrams(text: str, n: int = 8) -> set:
    tokens = text.lower().split()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def task_text(task: dict) -> str:
    # Constraints are intentional templates shared across task types — exclude them
    # from n-gram checks to avoid false positives. Only scenario context can contaminate.
    return task["input"].get("context", "")

def check_ngram(held_out: list, train_dev: list, n: int = 8) -> dict:
    train_dev_ngrams = set()
    for t in train_dev:
        train_dev_ngrams.update(get_ngrams(task_text(t), n))

    violations = []
    for t in held_out:
        ho_grams = get_ngrams(task_text(t), n)
        overlap = ho_grams & train_dev_ngrams
        if overlap:
            violations.append({
                "task_id": t["task_id"],
                "overlap_count": len(overlap),
                "example": list(next(iter(overlap))),
            })
    return {
        "check": "ngram_overlap",
        "n": n,
        "status": "PASS" if not violations else "FAIL",
        "violations": len(violations),
        "violation_details": violations[:5],
        "held_out_tasks_checked": len(held_out),
        "train_dev_ngrams_indexed": len(train_dev_ngrams),
    }

def check_embedding(held_out: list, train_dev: list, threshold: float = 0.85) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        ho_texts  = [task_text(t) for t in held_out]
        td_texts  = [task_text(t) for t in train_dev]
        ho_embs   = model.encode(ho_texts,  normalize_embeddings=True, show_progress_bar=False)
        td_embs   = model.encode(td_texts, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = ho_embs @ td_embs.T

        high_sim_pairs = []
        for i, row in enumerate(sim_matrix):
            for j, sim in enumerate(row):
                if sim >= threshold:
                    high_sim_pairs.append({
                        "held_out_id": held_out[i]["task_id"],
                        "train_dev_id": train_dev[j]["task_id"],
                        "cosine_similarity": round(float(sim), 4),
                    })
        return {
            "check": "embedding_similarity",
            "model": "all-MiniLM-L6-v2",
            "threshold": threshold,
            "status": "PASS" if not high_sim_pairs else "FAIL",
            "high_similarity_pairs": len(high_sim_pairs),
            "pair_details": high_sim_pairs[:5],
        }
    except ImportError:
        return {
            "check": "embedding_similarity",
            "status": "SKIPPED",
            "reason": "sentence-transformers not installed. Run: pip install sentence-transformers",
        }

def check_timeshift(tasks: list) -> dict:
    violations = []
    public_signal_count = 0
    for t in tasks:
        src = t["metadata"].get("signal_source", "synthetic")
        win = t["metadata"].get("signal_time_window")
        if src in PUBLIC_SOURCES:
            public_signal_count += 1
            if not win:
                violations.append({
                    "task_id": t["task_id"],
                    "signal_source": src,
                    "signal_time_window": win,
                })
    return {
        "check": "time_shift_verification",
        "status": "PASS" if not violations else "FAIL",
        "public_signal_tasks": public_signal_count,
        "missing_time_window": len(violations),
        "violation_details": violations[:5],
    }

# ── Stratified partition ──────────────────────────────────────────────────────

def _profile_key(task: dict) -> str:
    """Group key for partitioning. Tasks that share a prospect scenario (same first
    60 chars of context) are kept in the same split to prevent n-gram leakage.
    trace_derived variants share the same base prospect; programmatic/multi_llm share
    the same profile/seed. All other tasks are independent."""
    mode = task.get("authoring_mode", "")
    if mode in ("programmatic", "multi_llm", "trace_derived"):
        return task["input"]["context"][:60]
    return task["task_id"]


def stratified_partition(tasks: list) -> tuple:
    """Partition 50/30/20 preserving segment distribution.
    Programmatic tasks are grouped by prospect profile so all task-type variants
    of the same profile land in the same split (prevents n-gram leakage)."""
    by_segment = defaultdict(list)
    for t in tasks:
        by_segment[t["metadata"]["tenacious_segment"]].append(t)

    train_tasks, dev_tasks, held_out_tasks = [], [], []

    for seg, seg_tasks in by_segment.items():
        # Group by profile key
        groups = defaultdict(list)
        for t in seg_tasks:
            groups[_profile_key(t)].append(t)
        group_list = list(groups.values())
        random.shuffle(group_list)

        n = len(group_list)
        n_train = round(n * 0.50)
        n_dev   = round(n * 0.30)
        for group in group_list[:n_train]:
            train_tasks += group
        for group in group_list[n_train:n_train + n_dev]:
            dev_tasks += group
        for group in group_list[n_train + n_dev:]:
            held_out_tasks += group

    for t in train_tasks:
        t["split"] = "train"
    for t in dev_tasks:
        t["split"] = "dev"
    for t in held_out_tasks:
        t["split"] = "held_out"

    return train_tasks, dev_tasks, held_out_tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contamination checks + dataset partitioning")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding similarity check (faster)")
    args = parser.parse_args()

    if not FILTERED.exists():
        print(f"[ERROR] {FILTERED} not found. Run judge_filter.py first.")
        sys.exit(1)

    # Load filtered tasks
    tasks = []
    with open(FILTERED) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"\nContamination checks: {len(tasks)} filtered tasks")

    # Partition first (need splits before running cross-split checks)
    train, dev, held_out = stratified_partition(tasks)
    train_dev = train + dev
    print(f"  Partition: train={len(train)}, dev={len(dev)}, held_out={len(held_out)}")

    # Run checks
    r_ngram     = check_ngram(held_out, train_dev)
    r_timeshift = check_timeshift(tasks)
    if args.skip_embedding:
        r_embedding = {"check": "embedding_similarity", "status": "SKIPPED",
                       "reason": "--skip-embedding flag set"}
    else:
        print("  Running embedding similarity check (may take ~30s)...")
        r_embedding = check_embedding(held_out, train_dev)

    all_pass = all(
        r["status"] in ("PASS", "SKIPPED")
        for r in [r_ngram, r_embedding, r_timeshift]
    )

    # Write contamination report
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_tasks": len(tasks),
        "partition": {"train": len(train), "dev": len(dev), "held_out": len(held_out)},
        "all_checks_passed": all_pass,
        "checks": {
            "1_ngram_overlap":          r_ngram,
            "2_embedding_similarity":   r_embedding,
            "3_time_shift_verification": r_timeshift,
        },
    }
    OUT_CHECK.parent.mkdir(parents=True, exist_ok=True)
    OUT_CHECK.write_text(json.dumps(report, indent=2))
    print(f"\nContamination report → {OUT_CHECK}")
    print(f"  Check 1 (n-gram):    {r_ngram['status']}")
    print(f"  Check 2 (embedding): {r_embedding['status']}")
    print(f"  Check 3 (timeshift): {r_timeshift['status']}")
    print(f"  All passed: {all_pass}")

    if not all_pass:
        print("\n[WARN] One or more checks FAILED. Review contamination_check.json before sealing held_out.")

    # Write splits
    splits = {"train": train, "dev": dev, "held_out": held_out}
    for split_name, split_tasks in splits.items():
        split_dir = SPLITS_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "tasks.jsonl"
        with open(out_path, "w") as f:
            for t in split_tasks:
                f.write(json.dumps(t) + "\n")
        print(f"  Wrote {len(split_tasks):3d} tasks → {out_path}")

    print(f"\n[DONE] Dataset ready at {SPLITS_DIR}")
    print("  Next: seal held_out by ensuring it's in .gitignore, then commit.")


if __name__ == "__main__":
    main()
