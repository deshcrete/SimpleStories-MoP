"""Solve  ℓ_mixture(x) ≈ Σ_i α_i · ℓ_persona_i(x)  in per-token logprob space.

For each model (mixture + each persona LoRA), compute the per-token logprob
of the actual next token on every holdout example. Stack those into a design
matrix Θ ∈ R^{N×K} (N tokens, K personas) with target t ∈ R^N (mixture
logprobs) and solve for α under three constraint regimes:

  - linear span    : min ‖t - Θα‖²
  - affine span    : min ‖t - Θα‖²  s.t. Σα = 1
  - convex hull    : min ‖t - Θα‖²  s.t. α ≥ 0, Σα = 1

The "convex hull" form is the literal "mixture-probs" interpretation
(α_i ≥ 0, Σα_i = 1). Note that this corresponds to a *log-linear* pool of
distributions (product-of-experts, geometric mean) — not an *arithmetic*
mixture p_mix = Σ α_i p_i. The arithmetic mixture's logprob is
log Σ α_i exp(ℓ_i), which is bounded below by Σ α_i ℓ_i (Jensen), so the
linear-logprob fit is strictly the harder test.

Usage:
    python solve_logprob_mixture.py --runs_dir runs --holdout data/holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.optimize import minimize
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_holdout(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def per_token_logprobs(model, tokenizer, rows, max_length, batch_size, device):
    """Return per-token logprob of the actual next token over the holdout,
    flattened across (example, position), with padding positions removed."""
    model.eval()
    chunks: list[np.ndarray] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        enc = tokenizer(
            [r["text"] for r in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits  # (B, T, V)
        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        shift_mask = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(shift_logits.float(), dim=-1)
        tok_lp = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        chunks.append(tok_lp[shift_mask].cpu().numpy())
    return np.concatenate(chunks)


def solve_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def solve_affine(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = X.shape[1]
    G = X.T @ X
    b = X.T @ y
    M = np.zeros((K + 1, K + 1))
    M[:K, :K] = G
    M[:K, K] = 1.0
    M[K, :K] = 1.0
    rhs = np.concatenate([b, [1.0]])
    return np.linalg.solve(M, rhs)[:K]


def solve_convex(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = X.shape[1]
    G = X.T @ X
    b = X.T @ y

    def f(a: np.ndarray) -> float:
        return float(0.5 * a @ G @ a - b @ a)

    def jac(a: np.ndarray) -> np.ndarray:
        return G @ a - b

    cons = [{"type": "eq",
             "fun": lambda a: float(np.sum(a) - 1.0),
             "jac": lambda a: np.ones(K)}]
    bounds = [(0.0, 1.0)] * K
    res = minimize(f, np.full(K, 1.0 / K), jac=jac, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def report(name: str, alpha: np.ndarray, X: np.ndarray, y: np.ndarray,
           run_names: list[str]) -> None:
    pred = X @ alpha
    resid = y - pred
    rel = float(np.linalg.norm(resid) / (np.linalg.norm(y) + 1e-30))
    denom = np.linalg.norm(y) * np.linalg.norm(pred)
    cos = float(y @ pred / denom) if denom > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))  # systematic offset, e.g. log-Z mismatch
    print(f"\n[{name}]")
    print(f"  Σα = {alpha.sum():+.4f}")
    print(f"  ‖resid‖ / ‖y‖     = {rel:.4f}")
    print(f"  cos(y, pred)      = {cos:.4f}")
    print(f"  RMSE  (nats/tok)  = {rmse:.4f}")
    print(f"  mean resid (bias) = {bias:+.4f}")
    width = max(len(n) for n in run_names)
    for n, a in zip(run_names, alpha):
        print(f"    α[{n:<{width}}] = {a:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--holdout", default="data/holdout.jsonl")
    p.add_argument("--base_model", default=None,
                   help="If unset, read from mixture/adapter_config.json")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    mix_dir = runs_dir / "mixture"
    persona_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("theme_")
    )
    if not mix_dir.exists():
        raise SystemExit(f"no mixture run at {mix_dir}")
    if not persona_dirs:
        raise SystemExit(f"no theme_* runs under {runs_dir}")

    base_id = args.base_model or json.loads(
        (mix_dir / "adapter_config.json").read_text()
    )["base_model_name_or_path"]

    print(f"Base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_id).to(args.device)

    print(f"Wrapping with mixture adapter ({mix_dir.name}) ...")
    model = PeftModel.from_pretrained(base, mix_dir, adapter_name="mixture")
    persona_names = []
    for d in persona_dirs:
        model.load_adapter(str(d), adapter_name=d.name)
        persona_names.append(d.name)
    print(f"Loaded adapters: mixture + {len(persona_names)} personas")

    rows = load_holdout(Path(args.holdout))
    print(f"Holdout: {len(rows)} examples")

    print("\nScoring mixture ...")
    model.set_adapter("mixture")
    y = per_token_logprobs(model, tokenizer, rows, args.max_length,
                           args.batch_size, args.device)
    print(f"  N tokens = {y.size}, mean logprob = {y.mean():.4f}")

    cols: list[np.ndarray] = []
    for name in persona_names:
        print(f"Scoring {name} ...")
        model.set_adapter(name)
        col = per_token_logprobs(model, tokenizer, rows, args.max_length,
                                 args.batch_size, args.device)
        if col.size != y.size:
            raise RuntimeError(
                f"token-count mismatch for {name}: {col.size} vs {y.size}"
            )
        cols.append(col)
        print(f"  mean logprob = {col.mean():.4f}")

    X = np.stack(cols, axis=1)
    K = len(persona_names)

    print(f"\nDesign matrix: X ∈ R^({X.shape[0]} × {X.shape[1]}), "
          f"target y ∈ R^{y.size}")
    print(f"Per-persona vs mixture cosine (in logprob space):")
    y_n = y / (np.linalg.norm(y) + 1e-30)
    for name, col in zip(persona_names, cols):
        c = float(col @ y_n / (np.linalg.norm(col) + 1e-30))
        print(f"  {name:<24}  cos = {c:+.4f}")

    report("uniform avg α_i = 1/K", np.full(K, 1.0 / K), X, y, persona_names)
    report("linear span (unconstrained)", solve_linear(X, y), X, y, persona_names)
    report("affine span (Σα = 1)", solve_affine(X, y), X, y, persona_names)
    report("convex hull (α ≥ 0, Σα = 1)", solve_convex(X, y), X, y, persona_names)


if __name__ == "__main__":
    main()
