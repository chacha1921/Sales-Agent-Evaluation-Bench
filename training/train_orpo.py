#!/usr/bin/env python3
"""
train_orpo.py — Fine-tune with ORPO (Odds Ratio Preference Optimization).

ORPO combines SFT loss + odds-ratio preference loss in a single stage.
No reference model needed — unlike DPO, there is no frozen model copy.
The odds-ratio term penalises the model for assigning high likelihood to
rejected responses relative to chosen responses.

Reference: Hong et al. (2024) "ORPO: Monolithic Preference Optimization
without Reference Model" — https://arxiv.org/abs/2403.07691

Usage (Google Colab T4 — 16GB):
    !pip install unsloth trl datasets peft bitsandbytes
    !python training/train_orpo.py

    # Custom model or output dir:
    !python training/train_orpo.py --model unsloth/Qwen3-0.6B-bnb-4bit --output-dir runs/orpo_small

    # Dry run (1 step, verifies setup):
    !python training/train_orpo.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
PAIRS_FILE = ROOT / "training" / "training_data" / "path_b_dpo" / "preference_pairs.jsonl"
DEV_FILE   = ROOT / "dataset" / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "unsloth/Qwen3-4B-bnb-4bit"    # T4 options: Qwen3-0.6B, 1.7B, 4B (bnb-4bit loads faster)
DEFAULT_OUT_DIR = str(ROOT / "runs" / "orpo")
DEFAULT_SEED    = 42

LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,                  # unsloth: alpha == r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,                 # unsloth: 0 preferred
    bias="none",
    use_gradient_checkpointing="unsloth",   # unsloth memory optimisation
    random_state=3407,
)

ORPO_ARGS = dict(
    beta=0.1,              # ORPO odds-ratio weight (λ in the paper)
    max_length=512,        # emails are <200 words; 2048 caused OOM on T4 4B
    max_prompt_length=256,
    per_device_train_batch_size=2,  # reduced from 4 to avoid OOM
    gradient_accumulation_steps=8,  # effective batch = 16 (same as before)
    num_train_epochs=3,
    learning_rate=5e-5,
    optim="adamw_8bit",             # unsloth: 8-bit Adam saves memory
    lr_scheduler_type="cosine",
    warmup_steps=10,
    fp16=True,             # T4 does not support bf16
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    seed=DEFAULT_SEED,
    report_to="none",
    remove_unused_columns=False,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_pairs(path: Path) -> list:
    if not path.exists():
        print(f"[ERROR] {path} not found. Run generate_preference_pairs.py first.")
        sys.exit(1)
    return [json.loads(l) for l in open(path) if l.strip()]


def to_hf_dataset(pairs: list, tokenizer):
    """Convert preference pairs to HuggingFace Dataset for ORPOTrainer."""
    from datasets import Dataset

    def format_messages(messages: list, add_gen: bool = False) -> str:
        # enable_thinking=False: disable Qwen3 <think> tokens for sales email generation
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_gen,
                enable_thinking=False,
            )
        except TypeError:
            # older tokenizer versions don't have enable_thinking
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen
            )

    rows = []
    for p in pairs:
        prompt_msgs   = p["prompt"]
        chosen_msgs   = prompt_msgs + p["chosen"]
        rejected_msgs = prompt_msgs + p["rejected"]
        rows.append({
            "prompt":   format_messages(prompt_msgs),
            "chosen":   format_messages(chosen_msgs),
            "rejected": format_messages(rejected_msgs),
        })

    ds = Dataset.from_list(rows)
    # 90/10 train/eval split within the preference pairs
    split = ds.train_test_split(test_size=0.1, seed=DEFAULT_SEED)
    return split["train"], split["test"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ORPO fine-tuning for Tenacious")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="T4 options: unsloth/Qwen3-{0.6B,1.7B,4B}-bnb-4bit")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--beta",       type=float, default=0.1,
                        help="ORPO odds-ratio weight λ (default: 0.1)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 1 training step to verify setup, then exit")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[train_orpo] model={args.model}  beta={args.beta}  "
          f"epochs={args.epochs}  seed={args.seed}")

    # ── Imports (deferred so --help works without GPU) ────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import ORPOConfig, ORPOTrainer
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install unsloth trl datasets peft bitsandbytes")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,         # auto-detect (bf16 on A100, fp16 on T4)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    print(f"  LoRA params: r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}, dropout={LORA_CONFIG['lora_dropout']}")

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs = load_pairs(PAIRS_FILE)
    print(f"  Preference pairs loaded: {len(pairs)}")
    train_ds, eval_ds = to_hf_dataset(pairs, tokenizer)
    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ── ORPO config ───────────────────────────────────────────────────────────
    orpo_cfg = ORPO_ARGS.copy()
    orpo_cfg.update({
        "beta":             args.beta,
        "num_train_epochs": args.epochs if not args.dry_run else 1,
        "learning_rate":    args.lr,
        "output_dir":       args.output_dir,
        "seed":             args.seed,
        "max_steps":        1 if args.dry_run else -1,
    })
    config = ORPOConfig(**orpo_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print(f"\nStarting ORPO training ({'DRY RUN — 1 step' if args.dry_run else f'{args.epochs} epochs'})...")
    import time
    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0

    # Write training_run.log
    log_path = ROOT / "training_run.log"
    with open(log_path, "w") as lf:
        lf.write("# ORPO Training Run Log\n")
        lf.write(f"method: ORPO\n")
        lf.write(f"model: {args.model}\n")
        lf.write(f"dry_run: {args.dry_run}\n")
        lf.write(f"wall_time_s: {wall_time:.1f}\n\n")
        lf.write("## Hyperparameters\n")
        lf.write(f"beta: {args.beta}\n")
        lf.write(f"epochs: {args.epochs}\n")
        lf.write(f"learning_rate: {args.lr}\n")
        lf.write(f"lora_r: {LORA_CONFIG['r']}\n")
        lf.write(f"lora_alpha: {LORA_CONFIG['lora_alpha']}\n")
        lf.write(f"lora_dropout: {LORA_CONFIG['lora_dropout']}\n")
        lf.write(f"batch_size: {ORPO_ARGS['per_device_train_batch_size']}\n")
        lf.write(f"grad_accum: {ORPO_ARGS['gradient_accumulation_steps']}\n")
        lf.write(f"effective_batch: {ORPO_ARGS['per_device_train_batch_size'] * ORPO_ARGS['gradient_accumulation_steps']}\n")
        lf.write(f"lr_scheduler: {ORPO_ARGS['lr_scheduler_type']}\n")
        lf.write(f"warmup_steps: {ORPO_ARGS['warmup_steps']}\n")
        lf.write(f"max_length: {ORPO_ARGS['max_length']}\n")
        lf.write(f"max_prompt_length: {ORPO_ARGS['max_prompt_length']}\n")
        lf.write(f"fp16: {ORPO_ARGS['fp16']}\n")
        lf.write(f"seed: {args.seed}\n")
        lf.write(f"train_pairs: {len(train_ds)}\n")
        lf.write(f"eval_pairs: {len(eval_ds)}\n\n")
        lf.write("## Loss Curve (step, train_loss, eval_loss)\n")
        for entry in trainer.state.log_history:
            step = entry.get("step", "")
            tloss = entry.get("loss", entry.get("train_loss", ""))
            eloss = entry.get("eval_loss", "")
            if tloss or eloss:
                lf.write(f"step={step}  train_loss={tloss}  eval_loss={eloss}\n")
    print(f"  Training log → {log_path}")

    if not args.dry_run:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
        print(f"\n[DONE] ORPO adapter saved → {out / 'adapter'}")
        print(f"  Next: python training/train_simpo.py")
        print(f"        python training/run_ablations.py --winner orpo")
    else:
        print("\n[DRY RUN DONE] Setup verified. Re-run without --dry-run to train.")


if __name__ == "__main__":
    main()
