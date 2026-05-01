#!/usr/bin/env python3
"""
train_simpo.py — Fine-tune with SimPO (Simple Preference Optimization).

SimPO replaces DPO's log-ratio reward with a length-normalised average
log-likelihood reward, plus a target-reward margin γ. No reference model.

Key differences from ORPO:
  - ORPO:  SFT loss + odds-ratio term (single joint loss)
  - SimPO: pure preference loss with length normalisation + margin γ
  - SimPO tends to produce more confident, decisive outputs; ORPO can be
    more conservative. Run both and compare on dev split.

Reference: Meng et al. (2024) "SimPO: Simple Preference Optimization with
a Reference-Free Reward" — https://arxiv.org/abs/2405.14734

Implemented via TRL's CPOTrainer with loss_type="simpo".

Usage (Google Colab T4 — 16GB):
    !pip install unsloth trl datasets peft bitsandbytes
    !python training/train_simpo.py

    # Custom config:
    !python training/train_simpo.py --model unsloth/Qwen3-0.6B-bnb-4bit --gamma 1.0

    # Dry run:
    !python training/train_simpo.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
PAIRS_FILE = ROOT / "training" / "training_data" / "path_b_dpo" / "preference_pairs.jsonl"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "unsloth/Qwen3-4B-bnb-4bit"    # T4 options: Qwen3-0.6B, 1.7B, 4B (bnb-4bit)
DEFAULT_OUT_DIR = str(ROOT / "runs" / "simpo")
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

SIMPO_ARGS = dict(
    loss_type="simpo",
    beta=2.0,              # SimPO temperature β (scales reward)
    simpo_gamma=1.0,       # target reward margin γ — key SimPO hyperparameter
    max_length=2048,
    max_prompt_length=1024,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    optim="adamw_8bit",             # unsloth: 8-bit Adam saves memory
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    seed=DEFAULT_SEED,
    report_to="none",
    remove_unused_columns=False,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_pairs(path):
    if not path.exists():
        print(f"[ERROR] {path} not found. Run generate_preference_pairs.py first.")
        sys.exit(1)
    return [json.loads(l) for l in open(path) if l.strip()]


def to_hf_dataset(pairs, tokenizer):
    from datasets import Dataset

    def fmt(messages, add_gen=False):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_gen,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen
            )

    rows = []
    for p in pairs:
        rows.append({
            "prompt":   fmt(p["prompt"]),
            "chosen":   fmt(p["prompt"] + p["chosen"]),
            "rejected": fmt(p["prompt"] + p["rejected"]),
        })

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.1, seed=DEFAULT_SEED)
    return split["train"], split["test"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimPO fine-tuning for Tenacious")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--beta",       type=float, default=2.0,
                        help="SimPO temperature β (default: 2.0)")
    parser.add_argument("--gamma",      type=float, default=1.0,
                        help="SimPO target reward margin γ (default: 1.0)")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[train_simpo] model={args.model}  beta={args.beta}  "
          f"gamma={args.gamma}  epochs={args.epochs}  seed={args.seed}")

    try:
        from unsloth import FastLanguageModel
        from trl import CPOConfig, CPOTrainer
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

    # ── Load data ─────────────────────────────────────────────────────────────
    pairs = load_pairs(PAIRS_FILE)
    print(f"  Preference pairs loaded: {len(pairs)}")
    train_ds, eval_ds = to_hf_dataset(pairs, tokenizer)
    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ── SimPO config ──────────────────────────────────────────────────────────
    cfg = SIMPO_ARGS.copy()
    cfg.update({
        "beta":             args.beta,
        "simpo_gamma":      args.gamma,
        "num_train_epochs": args.epochs if not args.dry_run else 1,
        "learning_rate":    args.lr,
        "output_dir":       args.output_dir,
        "seed":             args.seed,
        "max_steps":        1 if args.dry_run else -1,
    })
    config = CPOConfig(**cfg)

    trainer = CPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print(f"\nStarting SimPO training ({'DRY RUN' if args.dry_run else f'{args.epochs} epochs'})...")
    import time
    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0

    # Write training_run.log (appends SimPO section if ORPO already wrote it)
    log_path = ROOT / "training_run.log"
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode) as lf:
        lf.write("\n# SimPO Training Run Log\n")
        lf.write(f"method: SimPO\n")
        lf.write(f"model: {args.model}\n")
        lf.write(f"dry_run: {args.dry_run}\n")
        lf.write(f"wall_time_s: {wall_time:.1f}\n\n")
        lf.write("## Hyperparameters\n")
        lf.write(f"beta: {args.beta}\n")
        lf.write(f"simpo_gamma: {args.gamma}\n")
        lf.write(f"epochs: {args.epochs}\n")
        lf.write(f"learning_rate: {args.lr}\n")
        lf.write(f"lora_r: {LORA_CONFIG['r']}\n")
        lf.write(f"lora_alpha: {LORA_CONFIG['lora_alpha']}\n")
        lf.write(f"batch_size: {SIMPO_ARGS['per_device_train_batch_size']}\n")
        lf.write(f"grad_accum: {SIMPO_ARGS['gradient_accumulation_steps']}\n")
        lf.write(f"effective_batch: {SIMPO_ARGS['per_device_train_batch_size'] * SIMPO_ARGS['gradient_accumulation_steps']}\n")
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
        print(f"\n[DONE] SimPO adapter saved → {out / 'adapter'}")
        print(f"  Next: python training/run_ablations.py --winner simpo")
    else:
        print("\n[DRY RUN DONE] Setup verified.")


if __name__ == "__main__":
    main()
