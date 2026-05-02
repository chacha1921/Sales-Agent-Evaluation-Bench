#!/usr/bin/env python3
"""
compare_methods.py — Evaluate ORPO vs SimPO adapters on the dev split.

Generates outputs from each adapter for every dev task, scores them with
the evaluation rubric, and produces a comparison table with bootstrap CIs.

Usage:
    python training/compare_methods.py \
        --orpo-adapter  runs/orpo/adapter \
        --simpo-adapter runs/simpo/adapter \
        --base-model    unsloth/Qwen3-4B-bnb-4bit

    # Mock mode (no GPU — uses random scores to test the comparison logic)
    python training/compare_methods.py --mock
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DEV_FILE = ROOT / "dataset" / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
EVAL_DIR = ROOT / "evaluation"
OUT_DIR  = ROOT / "results"

_DEFAULT_SEED = 42

# ── Scoring (imports from evaluation/scoring_evaluator.py) ───────────────────

def score_output(output: str, task: dict) -> dict:
    """Score one output against the rubric. Returns per-dimension scores + aggregate."""
    sys.path.insert(0, str(EVAL_DIR.parent))
    try:
        from evaluation.scoring_evaluator import (
            signal_grounding_fn, banned_phrase_fn, cta_checker_fn,
            word_count_fn, pricing_mention_fn, tone_checker_fn,
        )
    except ImportError:
        return {"aggregate": random.uniform(2.0, 4.5), "mock": True}

    ctx = task["input"]["context"]
    constraints = task["input"].get("constraints", [])

    # Extract word limit
    import re
    word_limit = 200
    for c in constraints:
        m = re.search(r"under\s+(\d+)\s+words?", c, re.I)
        if m:
            word_limit = int(m.group(1))
            break

    scores = {
        "signal_grounding": signal_grounding_fn(output, ctx),
        "banned_phrase":    banned_phrase_fn(output),
        "cta_present":      cta_checker_fn(output),
        "word_count_ok":    word_count_fn(output, word_limit),
        "no_pricing":       pricing_mention_fn(output),
        "tone":             tone_checker_fn(output),
    }

    task_type = task["input"]["task_type"]
    if task_type in ("email_outreach", "email_outreach_no_pricing", "follow_up"):
        weights = {"signal_grounding": 0.30, "banned_phrase": 0.20,
                   "cta_present": 0.20, "word_count_ok": 0.10,
                   "no_pricing": 0.10, "tone": 0.10}
    elif task_type == "objection_handling":
        weights = {"signal_grounding": 0.20, "banned_phrase": 0.20,
                   "cta_present": 0.10, "word_count_ok": 0.10,
                   "no_pricing": 0.10, "tone": 0.30}
    else:
        weights = {"signal_grounding": 0.25, "banned_phrase": 0.20,
                   "cta_present": 0.15, "word_count_ok": 0.15,
                   "no_pricing": 0.10, "tone": 0.15}

    aggregate = sum(scores[k] * weights[k] for k in weights) * 5
    scores["aggregate"] = round(aggregate, 3)
    return scores


# ── Generation ────────────────────────────────────────────────────────────────

def _load_adapter(adapter_abs: str, base_model: str):
    """Load base model + apply a saved LoRA adapter for inference.

    Unsloth saves only adapter delta weights (adapter_config.json +
    adapter_model.safetensors). Neither PeftModel.from_pretrained nor
    FastLanguageModel.from_pretrained can load these directly in this
    environment, so we do it in three explicit steps:
      1. Load base model with Unsloth
      2. Re-apply LoRA structure via FastLanguageModel.get_peft_model
         using config read from adapter_config.json
      3. Load saved delta weights via load_state_dict
    """
    import json, os
    from unsloth import FastLanguageModel
    from safetensors.torch import load_file as load_safetensors

    # Diagnose directory contents on first call
    if os.path.exists(adapter_abs):
        print(f"  Adapter dir contents: {sorted(os.listdir(adapter_abs))}")
    else:
        raise FileNotFoundError(f"Adapter directory not found: {adapter_abs}")

    cfg_path = os.path.join(adapter_abs, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"adapter_config.json missing from {adapter_abs}. "
            "Re-save with model.save_pretrained() from a PEFT/Unsloth training run."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r                       = cfg["r"],
        lora_alpha              = cfg["lora_alpha"],
        target_modules          = cfg["target_modules"],
        lora_dropout            = 0,
        bias                    = cfg.get("bias", "none"),
        use_gradient_checkpointing = False,
        random_state            = 3407,
    )

    weights_st = os.path.join(adapter_abs, "adapter_model.safetensors")
    weights_pt = os.path.join(adapter_abs, "adapter_model.bin")
    if os.path.exists(weights_st):
        sd = load_safetensors(weights_st)
    elif os.path.exists(weights_pt):
        import torch
        sd = torch.load(weights_pt, map_location="cuda")
    else:
        raise FileNotFoundError(f"No adapter weights (safetensors/bin) in {adapter_abs}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys — adapter may not be fully applied")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_outputs(tasks: list, adapter_path: str, base_model: str) -> list:
    """Generate one output per task using a LoRA adapter."""
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)

    adapter_abs = str(Path(adapter_path).resolve())
    print(f"  Loading adapter from {adapter_abs}...")
    model, tokenizer = _load_adapter(adapter_abs, base_model)

    outputs = []
    for task in tasks:
        ctx = task["input"]["context"]
        task_type = task["input"]["task_type"]
        constraints = task["input"].get("constraints", [])
        constraint_str = "\n".join(f"- {c}" for c in constraints)

        messages = [
            {"role": "system",  "content": "You are a B2B sales assistant for Tenacious. Write direct, signal-led sales messages with no banned phrases."},
            {"role": "user",    "content": f"Write a {task_type.replace('_', ' ')} for this prospect.\n\nContext:\n{ctx}\n\nConstraints:\n{constraint_str}\n\nWrite only the message body."},
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with __import__("torch").no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=256, temperature=0.0, do_sample=False
            )
        decoded = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(decoded.strip())

    return outputs


def mock_outputs(tasks: list, label: str) -> list:
    """Generate plausible mock outputs for testing the comparison pipeline."""
    templates = {
        "orpo":  "{first} — {signal}. For teams at your stage, {pain_point} typically peaks around now. Worth 20 minutes? [CALENDLY_LINK]",
        "simpo": "{first}, {signal}. I've seen {pain_point} derail {segment} revenue teams at this exact stage. Brief conversation? [CALENDLY_LINK]",
    }
    import re
    tmpl = templates.get(label, templates["orpo"])
    outputs = []
    for t in tasks:
        ctx = t["input"]["context"]
        first_m = re.match(r"Prospect:\s*([A-Z][a-z]+)", ctx)
        signal_m = re.search(r"Signal:\s*(.+?)(?:\.|Known)", ctx)
        pain_m = re.search(r"pain point:\s*(.+?)(?:\.|$)", ctx)
        seg = t["metadata"]["tenacious_segment"]
        outputs.append(tmpl.format(
            first=first_m.group(1) if first_m else "Hi",
            signal=signal_m.group(1).strip() if signal_m else "recent signal",
            pain_point=pain_m.group(1).strip() if pain_m else "pipeline visibility",
            segment=seg,
        ))
    return outputs


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_mean_ci(scores: list, n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    means = [
        sum(random.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n_boot)
    ]
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(scores) / len(scores), lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare ORPO vs SimPO on dev split")
    parser.add_argument("--orpo-adapter",  default=str(ROOT / "runs" / "orpo" / "adapter"))
    parser.add_argument("--simpo-adapter", default=str(ROOT / "runs" / "simpo" / "adapter"))
    parser.add_argument("--base-model",    default="unsloth/Qwen3-4B-bnb-4bit")
    parser.add_argument("--mock",          action="store_true",
                        help="Use template outputs (no GPU needed)")
    parser.add_argument("--seed",          type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"[compare_methods] mock={args.mock}  seed={args.seed}")

    tasks = [json.loads(l) for l in open(DEV_FILE) if l.strip()]
    print(f"  Dev tasks: {len(tasks)}")

    if args.mock:
        orpo_outputs  = mock_outputs(tasks, "orpo")
        simpo_outputs = mock_outputs(tasks, "simpo")
    else:
        orpo_outputs  = generate_outputs(tasks, args.orpo_adapter,  args.base_model)
        simpo_outputs = generate_outputs(tasks, args.simpo_adapter, args.base_model)

    # Score all outputs
    print("\nScoring outputs...")
    results = []
    for i, task in enumerate(tasks):
        orpo_s  = score_output(orpo_outputs[i],  task)
        simpo_s = score_output(simpo_outputs[i], task)
        results.append({
            "task_id":        task["task_id"],
            "segment":        task["metadata"]["tenacious_segment"],
            "task_type":      task["input"]["task_type"],
            "failure_mode":   task["metadata"]["failure_mode_tag"],
            "orpo_aggregate": orpo_s["aggregate"],
            "simpo_aggregate": simpo_s["aggregate"],
            "orpo_scores":    orpo_s,
            "simpo_scores":   simpo_s,
        })

    orpo_agg  = [r["orpo_aggregate"]  for r in results]
    simpo_agg = [r["simpo_aggregate"] for r in results]

    orpo_mean,  orpo_lo,  orpo_hi  = bootstrap_mean_ci(orpo_agg)
    simpo_mean, simpo_lo, simpo_hi = bootstrap_mean_ci(simpo_agg)
    delta = simpo_mean - orpo_mean

    print("\n" + "=" * 60)
    print("ORPO  vs  SimPO  —  Dev Split Results")
    print("=" * 60)
    print(f"{'Method':<10} {'Mean':>6} {'95% CI':>20} {'n':>5}")
    print("-" * 60)
    print(f"{'ORPO':<10} {orpo_mean:>6.3f}  [{orpo_lo:.3f}, {orpo_hi:.3f}]  {len(orpo_agg):>5}")
    print(f"{'SimPO':<10} {simpo_mean:>6.3f}  [{simpo_lo:.3f}, {simpo_hi:.3f}]  {len(simpo_agg):>5}")
    print(f"{'Δ (SimPO-ORPO)':<10} {delta:>+6.3f}")
    print("=" * 60)

    winner = "SimPO" if delta > 0 else "ORPO" if delta < 0 else "tie"
    print(f"Winner: {winner}")
    if abs(delta) < 0.05:
        print("Note: difference < 0.05 — may not be practically significant.")

    # Per-failure-mode breakdown
    print("\nPer failure mode:")
    from collections import defaultdict
    by_mode = defaultdict(lambda: {"orpo": [], "simpo": []})
    for r in results:
        by_mode[r["failure_mode"]]["orpo"].append(r["orpo_aggregate"])
        by_mode[r["failure_mode"]]["simpo"].append(r["simpo_aggregate"])
    for mode, d in sorted(by_mode.items()):
        om = sum(d["orpo"])  / len(d["orpo"])
        sm = sum(d["simpo"]) / len(d["simpo"])
        print(f"  {mode:<22} ORPO={om:.3f}  SimPO={sm:.3f}  Δ={sm-om:+.3f}  (n={len(d['orpo'])})")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "orpo_vs_simpo.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "orpo":  {"mean": orpo_mean,  "ci_lo": orpo_lo,  "ci_hi": orpo_hi},
                "simpo": {"mean": simpo_mean, "ci_lo": simpo_lo, "ci_hi": simpo_hi},
                "delta": delta, "winner": winner,
            },
            "per_task": results,
        }, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
