"""For every checkpoint of one run, compute the measurements task_plan.md asks for:

    1. Σ log P(seq) over the 750-seq combined inference set, per sequence.
       (Sum of log-probs, NOT mean — LoTP is an identity over probabilities.)
    2. Per-persona aggregate Σ_{j ∈ persona_k} log P(seq_j).
    3. ‖θ‖₂ and ‖θ − θ_base‖₂ on the flattened parameter vector.
    4. Persona-own (or mixture) held-out eval loss is already in train_log.jsonl;
       we re-record it here for convenience.

Outputs:
    results/metrics/<run>/per_sequence_logp.npy   shape (n_checkpoints, 750)
    results/metrics/<run>/checkpoints.jsonl       one line per checkpoint:
        {step, persona_logp: {persona: float}, total_logp, param_norm, dev_norm}

The per-sequence matrix is needed for LoTP and for the convex-hull-escape check,
which we run separately in src/lotp.py over the 6 stacked runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    PERSONAS,
    StoryDataset,
    collate_for_clm,
    load_inference_set,
)
from model import (
    BASE_MODEL_ID,
    load_base_model_and_tokenizer,
    load_checkpoint,
    param_vector,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_ROOT = RESULTS_DIR / "checkpoints"
METRICS_ROOT = RESULTS_DIR / "metrics"

EVAL_BATCH_SIZE = 32


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, dir), ...] sorted by step. Detects step_<N> directories."""
    out = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            step = int(d.name.split("_")[1])
            out.append((step, d))
    out.sort(key=lambda x: x[0])
    return out


@torch.no_grad()
def per_sequence_log_prob(model, loader, device: str) -> np.ndarray:
    """Returns shape (n_seqs,) of Σ_t log p(x_t | x_<t) per sequence.

    The HF causal-LM head shifts internally for the .loss return, but to get
    per-sequence sums we re-do the shift here so we can mask exactly the
    positions we want to score (everything that has a non-pad target).
    """
    model.eval()
    log_probs: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits                  # (B, T, V)
        # Predict token t from logits at t-1, just like CLM.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_softmax = torch.log_softmax(shift_logits.float(), dim=-1)
        # Gather the log-prob of the true next-token at each position; mask -100.
        valid = shift_labels != -100
        safe_labels = shift_labels.clone()
        safe_labels[~valid] = 0
        token_logp = log_softmax.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid.float()
        seq_logp = token_logp.sum(dim=1)     # (B,)
        log_probs.extend(seq_logp.detach().cpu().tolist())
    return np.array(log_probs, dtype=np.float64)


def eval_run(run: str) -> None:
    run_dir = CHECKPOINT_ROOT / run
    if not run_dir.exists():
        raise FileNotFoundError(f"No checkpoints for run '{run}' at {run_dir}")

    out_dir = METRICS_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer + base parameter vector for ‖θ−θ_base‖.
    _, tokenizer = load_base_model_and_tokenizer(device="cpu")
    # We need a fresh base model to compute base params; reload to avoid mutation.
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    base_params = param_vector(base_model)
    del base_model

    eval_rows = load_inference_set()
    eval_ds = StoryDataset(eval_rows, tokenizer)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_for_clm(b, tokenizer.pad_token_id),
    )

    personas_per_seq = [r["persona"] for r in eval_rows]   # length 750

    checkpoints = list_checkpoints(run_dir)
    n_seqs = len(eval_rows)
    matrix = np.zeros((len(checkpoints), n_seqs), dtype=np.float64)
    rows_out: list[dict] = []

    for ci, (step, ckpt_dir) in enumerate(checkpoints):
        model = load_checkpoint(ckpt_dir, device=device)
        seq_logp = per_sequence_log_prob(model, eval_loader, device)
        matrix[ci] = seq_logp

        pv = param_vector(model)
        pn = float(pv.norm(p=2).item())
        dn = float((pv - base_params).norm(p=2).item())

        per_persona = {p: 0.0 for p in PERSONAS}
        for lp, persona in zip(seq_logp.tolist(), personas_per_seq):
            per_persona[persona] += lp
        rows_out.append({
            "step": step,
            "param_norm": pn,
            "dev_norm": dn,
            "total_logp": float(seq_logp.sum()),
            "persona_logp": per_persona,
        })
        print(f"[{run}] step {step:>5}  total_logp={float(seq_logp.sum()):>10.1f}  "
              f"‖θ‖={pn:.2f}  ‖θ−θ_base‖={dn:.4f}")
        del model

    np.save(out_dir / "per_sequence_logp.npy", matrix)
    np.save(out_dir / "steps.npy", np.array([s for s, _ in checkpoints], dtype=np.int64))
    with (out_dir / "checkpoints.jsonl").open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    print(f"[{run}] eval done -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    eval_run(args.run)


if __name__ == "__main__":
    main()
