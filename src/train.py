"""Train one model (a specialist or the mixture) for a SINGLE epoch.

Single-epoch overfitting experiment (task_plan.md): every example is seen exactly
once, so any overfitting that shows up cannot be due to multi-epoch data reuse.
All six runs share SEED, optimizer, batch size, and epoch count so any difference
between them is attributable to the data they saw.

CLI:
    python src/train.py --run noir_detective   # specialist; trains on data/noir_detective/train.jsonl
    python src/train.py --run mixture          # mixture; trains on the union of all 5 train.jsonl

Outputs:
    results/checkpoints/<run>/step_<N>/         saved model weights at each cadence step
    results/checkpoints/<run>/train_log.jsonl   per-step train loss + per-checkpoint val losses
    results/checkpoints/<run>/run_config.json   hyperparameters + steps_per_epoch + git sha

Checkpointing: ~TARGET_CHECKPOINTS evenly spaced within the single epoch, plus
step 0 (base model) and the final step. The x-axis for all plots is fractional
epoch = step / steps_per_epoch (0 -> 1); on that axis the specialist and the
mixture have identical per-persona exposure at every point.

Per-checkpoint validation (the overfitting diagnostic):
    specialist -> own single-persona val (val_own) AND the full-mix val (val_mix)
    mixture    -> the full-mix val (val_mix; this is its own held-out)
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
    load_full_mix_val,
    load_mixture_train,
    load_mixture_train_nonuniform,
    load_mixture_train_uniform_matched,
    load_split,
    load_val,
)
from model import load_base_model_and_tokenizer, save_checkpoint

SEED = 42
BATCH_SIZE = 32
# Clustered base-model experiment (control/base_model_expr.md): the clustered data is
# IN-DISTRIBUTION for the base model (it already fits it at ~1.97 nats/tok), unlike
# the induce-prior personas which were out-of-distribution. lr=5e-4 (used there)
# destabilizes a converged base — step-2 loss spiked 2.0->4.6 and never recovered.
# We drop to 5e-5 with a linear warmup so the specialist gently sharpens on its
# cluster instead of being knocked off the base optimum.
LR = 5e-5
WARMUP_RATIO = 0.1      # linear warmup over the first 10% of steps, then constant
WEIGHT_DECAY = 0.0      # avoid conflating with the regularization story
NUM_EPOCHS = 1          # single-epoch experiment: every example seen exactly once
TARGET_CHECKPOINTS = 20  # evenly spaced within the single epoch
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


def load_train_rows(run: str) -> list[dict]:
    if run == "mixture":
        return load_mixture_train()
    if run == "mixture_exp":
        return load_mixture_train_nonuniform()
    if run == "mixture_unif_matched":
        return load_mixture_train_uniform_matched()
    if run in PERSONAS:
        return load_split(run, "train")
    raise ValueError(f"Unknown run '{run}'. Must be 'mixture', 'mixture_exp', or one of {PERSONAS}.")


def build_eval_loaders(run: str, tokenizer, pad_id: int) -> dict[str, DataLoader]:
    """Validation sets for the overfitting diagnostic.

    specialist -> {val_own: its persona val, val_mix: full-mix val}
    mixture    -> {val_mix: full-mix val}  (its own held-out)
    """
    if run.startswith("mixture"):
        sets = {"val_mix": load_full_mix_val()}
    else:
        sets = {"val_own": load_val(run), "val_mix": load_full_mix_val()}
    loaders = {}
    for name, rows in sets.items():
        ds = StoryDataset(rows, tokenizer)
        loaders[name] = DataLoader(
            ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=lambda b: collate_for_clm(b, pad_id),
        )
    return loaders


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


def eval_all(model, eval_loaders: dict[str, DataLoader], device: str) -> dict[str, float]:
    return {name: eval_loss(model, loader, device) for name, loader in eval_loaders.items()}


def train_one_run(run: str) -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_train_rows(run)
    model, tokenizer = load_base_model_and_tokenizer(device=device)
    pad_id = tokenizer.pad_token_id

    train_ds = StoryDataset(train_rows, tokenizer)

    # Fixed data order: seed-shuffled at construction. Same seed -> same order across
    # runs for the persona train sets; the mixture set has its own deterministic
    # order. We do NOT reshuffle (single epoch anyway), so every model sees a fixed
    # permutation of its data.
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
    eval_loaders = build_eval_loaders(run, tokenizer, pad_id)

    # ~TARGET_CHECKPOINTS evenly spaced within the single epoch.
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    stride = max(1, total_steps // TARGET_CHECKPOINTS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Linear warmup to LR over the first WARMUP_RATIO of steps, then hold constant.
    # Warmup kills the step-2 loss spike seen at the higher LR; constant-after keeps
    # the schedule simple and the per-step LR readable.
    from transformers import get_constant_schedule_with_warmup
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    out_dir = CHECKPOINT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_file = log_path.open("w")

    config = {
        "run": run,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "base_model": "SimpleStories/SimpleStories-V2-5M",
        "n_train": len(train_rows),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_stride": stride,
        "eval_sets": list(eval_loaders.keys()),
        "git_sha": git_sha(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    def checkpoint(step: int, note: str | None = None) -> None:
        save_checkpoint(model, out_dir / f"step_{step}")
        losses = eval_all(model, eval_loaders, device)
        entry = {"step": step, "event": "checkpoint", **losses}
        if note:
            entry["note"] = note
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()
        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"[{run}] step {step:>5}/{total_steps}  {loss_str}")

    # Step-0 (base model) checkpoint so trajectory plots have a defined start.
    checkpoint(0, note="base")

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
            scheduler.step()
            log_file.write(json.dumps({"step": step, "event": "train", "loss": float(loss.item()), "lr": scheduler.get_last_lr()[0], "epoch": epoch}) + "\n")

            # Evenly-spaced checkpoints, and always the final step.
            if step % stride == 0 or step == total_steps:
                checkpoint(step)

    log_file.close()
    print(f"[{run}] done. {step} steps ({steps_per_epoch} steps/epoch). checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="'mixture' or one of the 5 personas")
    args = parser.parse_args()
    train_one_run(args.run)


if __name__ == "__main__":
    main()
