"""Train one model (a specialist or the mixture) start-to-finish.

Six runs in total, all sharing the same SEED, optimizer, batch size, and epoch
count so any difference between them is attributable to the data they saw.

CLI:
    python src/train.py --run noir_detective       # specialist; trains on data/noir_detective/persona_train.jsonl
    python src/train.py --run mixture              # mixture; trains on all 5 mixture.jsonl files

Outputs:
    results/checkpoints/<run>/step_<N>/         saved model weights at each cadence step
    results/checkpoints/<run>/train_log.jsonl   per-step train loss + per-checkpoint eval loss
    results/checkpoints/<run>/run_config.json   hyperparameters + git sha + which split was used

Checkpoint cadence (task_plan.md):
    specialist: every 8 steps for first 80, every 16 after.        (~15 checkpoints over ~160 steps)
    mixture:    every 20 steps for first 200, every 50 after.       (~15 checkpoints over ~780 steps)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    PERSONAS,
    StoryDataset,
    collate_for_clm,
    load_inference_set,
    load_mixture_train,
    load_split,
)
from model import load_base_model_and_tokenizer, save_checkpoint

SEED = 42
BATCH_SIZE = 32
LR = 5e-4               # midpoint of the 3e-4..1e-3 range in task_plan.md
WEIGHT_DECAY = 0.0      # avoid conflating with the regularization story
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_ROOT = RESULTS_DIR / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Bit-identical reproducibility on GPU requires these flags (task_plan.md
    # §"Base Model and Fine-tuning Setup": seed/data-order/HPs held constant).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha() -> str:
    # Fail loudly per AGENT.md: silently labeling a run as "unknown" sha hides
    # provenance bugs (dirty env, missing git) that we want to see immediately.
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def specialist_cadence(step: int) -> bool:
    """Checkpoint after step `step` (1-indexed)."""
    if step <= 80:
        return step % 8 == 0
    return step % 16 == 0


def mixture_cadence(step: int) -> bool:
    if step <= 200:
        return step % 20 == 0
    return step % 50 == 0


def load_training_rows(run: str) -> tuple[list[dict], list[dict], callable]:
    """Returns (train_rows, eval_rows, cadence_fn) for the given run name."""
    if run == "mixture":
        train_rows = load_mixture_train()
        eval_rows = load_inference_set()
        return train_rows, eval_rows, mixture_cadence
    if run in PERSONAS:
        train_rows = load_split(run, "persona_train")
        eval_rows = load_split(run, "inference")
        return train_rows, eval_rows, specialist_cadence
    raise ValueError(f"Unknown run '{run}'. Must be 'mixture' or one of {PERSONAS}.")


@torch.no_grad()
def eval_loss(model, loader, device: str) -> float:
    """Mean cross-entropy per scored next-token position over the eval set.

    HF causal-LM `.loss` is the mean over *shifted* labels (i.e. positions
    1..T-1 are scored). We weight by the matching scored-position count to
    recover the true cross-set mean — not the un-shifted label count, which
    overcounts by one per sequence.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF shifts labels by 1 internally; mirror that for the token count.
        n_tokens = int((labels[..., 1:] != -100).sum().item())
        total_loss += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    model.train()
    if total_tokens == 0:
        raise RuntimeError("eval_loss: no scored tokens in eval set")
    return total_loss / total_tokens


def train_one_run(run: str) -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, eval_rows, cadence_fn = load_training_rows(run)
    model, tokenizer = load_base_model_and_tokenizer(device=device)
    pad_id = tokenizer.pad_token_id

    train_ds = StoryDataset(train_rows, tokenizer)
    eval_ds = StoryDataset(eval_rows, tokenizer)

    # Fixed data order: seed-shuffled at construction. Same seed -> same order across runs
    # for the persona_train sets; the mixture set has its own deterministic order. We do
    # *not* reshuffle every epoch — we want every model to see the same sequence ordering
    # so trajectory comparisons are clean.
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate_for_clm(b, pad_id),
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_for_clm(b, pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    out_dir = CHECKPOINT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_file = log_path.open("w")

    config = {
        "run": run,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "base_model": "SimpleStories/SimpleStories-V2-5M",
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "git_sha": git_sha(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    # Save the step-0 (base model, untrained on this data) checkpoint so trajectory plots
    # have a well-defined starting point.
    save_checkpoint(model, out_dir / "step_0")
    eval0 = eval_loss(model, eval_loader, device)
    log_file.write(json.dumps({"step": 0, "event": "checkpoint", "eval_loss": eval0}) + "\n")
    log_file.flush()

    step = 0
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_file.write(json.dumps({"step": step, "event": "train", "loss": float(loss.item()), "epoch": epoch}) + "\n")

            if cadence_fn(step):
                save_checkpoint(model, out_dir / f"step_{step}")
                ev = eval_loss(model, eval_loader, device)
                log_file.write(json.dumps({"step": step, "event": "checkpoint", "eval_loss": ev}) + "\n")
                log_file.flush()
                print(f"[{run}] step {step:>5}  train_loss={float(loss.item()):.4f}  eval_loss={ev:.4f}")

    # Always checkpoint the final step even if cadence didn't trigger.
    if not cadence_fn(step):
        save_checkpoint(model, out_dir / f"step_{step}")
        ev = eval_loss(model, eval_loader, device)
        log_file.write(json.dumps({"step": step, "event": "checkpoint", "eval_loss": ev, "note": "final"}) + "\n")

    log_file.close()
    print(f"[{run}] done. {step} steps. checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="'mixture' or one of the 5 personas")
    args = parser.parse_args()
    train_one_run(args.run)


if __name__ == "__main__":
    main()
