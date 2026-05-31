"""Assemble every model's converged (min-val) per-sequence log-prob on the 2,500-seq
test set into one tidy CSV, for step-by-step analysis of the induced prior.

One row per test sequence. Columns:
    id, persona (true source), n_tokens
    logp_base                                  base model (step 0; same for every run)
    logp_spec_<persona>  x5                    each specialist at its min-val checkpoint
    logp_mix_uniform                           full uniform mixture (9k/persona, 45k total)
    logp_mix_uniform_small                     size-matched uniform (4.7k/persona, 23.5k)
    logp_mix_nonuniform                        geometric-1.5 non-uniform mixture (23.5k)

All log-probs are Σ_t log p(x_t | x_<t) over the sequence (sum, not mean); divide by
n_tokens for per-token. Writes results/logprob_dataset.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import PERSONAS, load_test_set, tokenize_story
from lotp import RESULTS_DIR, load_run_metrics, min_val_loss_step
from model import load_base_model_and_tokenizer

MIXTURES = {
    "logp_mix_uniform": "mixture",
    "logp_mix_uniform_small": "mixture_unif_matched",
    "logp_mix_nonuniform": "mixture_exp",
}


def _minval_row(run: str) -> tuple[np.ndarray, int]:
    steps, matrix, _ = load_run_metrics(run)
    step = min_val_loss_step(run)
    return matrix[int(np.where(steps == step)[0][0])], step


def main() -> None:
    rows = load_test_set()                       # ordered the same as the metrics matrices
    _, tok = load_base_model_and_tokenizer(device="cpu")
    n_tokens = [len(tokenize_story(r["story"], tok)) for r in rows]

    data = {
        "id": [r["id"] for r in rows],
        "persona": [r["persona"] for r in rows],
        "n_tokens": n_tokens,
        "logp_base": load_run_metrics(PERSONAS[0])[1][0],   # step 0, identical across runs
    }
    steps_used = {"base": 0}
    for p in PERSONAS:
        row, step = _minval_row(p)
        data[f"logp_spec_{p}"] = row
        steps_used[f"spec_{p}"] = step
    for col, run in MIXTURES.items():
        row, step = _minval_row(run)
        data[col] = row
        steps_used[col] = step

    df = pd.DataFrame(data)
    out = RESULTS_DIR / "logprob_dataset.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"min-val checkpoint steps used: {steps_used}")
    print(f"per-persona row counts: {df['persona'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
