"""Convex-hull test: is the mixture LoRA adapter inside the convex hull of
the per-persona LoRA adapters?

For each run, the LoRA at every adapted layer defines an effective weight
delta  ΔW = (α/r) · B @ A. Flattened and concatenated across all adapted
layers, each run becomes one vector θ ∈ R^D, all living in the same space.

We project θ_mix onto {θ_persona_1, ..., θ_persona_K} under three nested
constraints:

  - linear span    : min ‖θ_mix - Θα‖²
  - affine span    : same, s.t. Σα = 1
  - convex hull    : same, s.t. α ≥ 0, Σα = 1   (the hypothesis)

and report normalized residual ‖resid‖ / ‖θ_mix‖, cosine similarity, and
the recovered weights. A uniform-average baseline (α_i = 1/K) is included
for reference.

Run:
    python analyze_hull.py            # uses ./runs
    python analyze_hull.py --runs_dir runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import minimize


def load_adapter_vector(run_dir: Path) -> np.ndarray:
    cfg = json.loads((run_dir / "adapter_config.json").read_text())
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])

    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(run_dir / "adapter_model.safetensors",
                   framework="pt", device="cpu") as f:
        for k in f.keys():
            if ".lora_A.weight" in k:
                base = k.replace(".lora_A.weight", "")
                pairs.setdefault(base, {})["A"] = (
                    f.get_tensor(k).to(torch.float32).numpy()
                )
            elif ".lora_B.weight" in k:
                base = k.replace(".lora_B.weight", "")
                pairs.setdefault(base, {})["B"] = (
                    f.get_tensor(k).to(torch.float32).numpy()
                )

    # Deterministic layer order across runs (same adapter_config → same keys).
    deltas: list[np.ndarray] = []
    for base in sorted(pairs):
        A = pairs[base]["A"]  # r × d_in   (peft convention)
        B = pairs[base]["B"]  # d_out × r
        if B.shape[1] != A.shape[0]:
            raise ValueError(
                f"shape mismatch at {base}: A {A.shape}, B {B.shape}"
            )
        deltas.append((scale * (B @ A)).ravel())
    return np.concatenate(deltas)


def solve_linear(Theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(Theta, t, rcond=None)[0]


def solve_affine(Theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    # min ‖t - Θα‖²  s.t. 1ᵀα = 1   (KKT block solve)
    K = Theta.shape[1]
    G = Theta.T @ Theta
    b = Theta.T @ t
    M = np.zeros((K + 1, K + 1))
    M[:K, :K] = G
    M[:K, K] = 1.0
    M[K, :K] = 1.0
    rhs = np.concatenate([b, [1.0]])
    return np.linalg.solve(M, rhs)[:K]


def solve_convex(Theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    K = Theta.shape[1]
    G = Theta.T @ Theta
    b = Theta.T @ t

    def f(a: np.ndarray) -> float:
        return float(0.5 * a @ G @ a - b @ a)

    def jac(a: np.ndarray) -> np.ndarray:
        return G @ a - b

    cons = [{"type": "eq",
             "fun": lambda a: float(np.sum(a) - 1.0),
             "jac": lambda a: np.ones(K)}]
    bounds = [(0.0, 1.0)] * K
    x0 = np.full(K, 1.0 / K)
    res = minimize(f, x0, jac=jac, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-14, "maxiter": 1000})
    return res.x


def report(name: str, alpha: np.ndarray,
           Theta: np.ndarray, t: np.ndarray,
           run_names: list[str]) -> None:
    proj = Theta @ alpha
    rel = np.linalg.norm(t - proj) / np.linalg.norm(t)
    denom = np.linalg.norm(t) * np.linalg.norm(proj)
    cos = float(t @ proj / denom) if denom > 0 else float("nan")
    print(f"\n[{name}]")
    print(f"  Σα = {alpha.sum():+.4f}")
    print(f"  ‖resid‖ / ‖θ_mix‖ = {rel:.4f}")
    print(f"  cos(θ_mix, proj)  = {cos:.4f}")
    width = max(len(n) for n in run_names)
    for n, a in zip(run_names, alpha):
        print(f"    α[{n:<{width}}] = {a:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    personas = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("theme_")
    )
    mix_dir = runs_dir / "mixture"
    if not mix_dir.exists():
        raise FileNotFoundError(f"no mixture run at {mix_dir}")
    if not personas:
        raise FileNotFoundError(f"no theme_* runs under {runs_dir}")

    print(f"Loading {len(personas)} persona adapters + mixture from {runs_dir}/")
    theta_personas = [load_adapter_vector(d) for d in personas]
    theta_mix = load_adapter_vector(mix_dir)
    Theta = np.stack(theta_personas, axis=1)
    K = Theta.shape[1]
    D = Theta.shape[0]
    print(f"D = {D} (flattened ΔW dim), K = {K} (personas)")

    print(f"\nNorms (‖ΔW‖_F per run):")
    print(f"  mixture                : {np.linalg.norm(theta_mix):.4f}")
    for d, th in zip(personas, theta_personas):
        print(f"  {d.name:<24}: {np.linalg.norm(th):.4f}")

    # Cosines between personas (how distinct are the experts?)
    print(f"\nCosine similarity between persona adapters:")
    norms = np.array([np.linalg.norm(th) for th in theta_personas])
    C = Theta.T @ Theta / (norms[:, None] * norms[None, :] + 1e-30)
    names_short = [d.name.replace("theme_", "t") for d in personas]
    header = "          " + "  ".join(f"{n:>6}" for n in names_short)
    print(header)
    for i, n in enumerate(names_short):
        row = "  ".join(f"{C[i, j]:>+6.3f}" for j in range(K))
        print(f"  {n:<6}  {row}")

    run_names = [d.name for d in personas]
    report("uniform avg α_i = 1/K", np.full(K, 1.0 / K),
           Theta, theta_mix, run_names)
    report("linear span (unconstrained)", solve_linear(Theta, theta_mix),
           Theta, theta_mix, run_names)
    report("affine span (Σα = 1)", solve_affine(Theta, theta_mix),
           Theta, theta_mix, run_names)
    report("convex hull (α ≥ 0, Σα = 1)", solve_convex(Theta, theta_mix),
           Theta, theta_mix, run_names)


if __name__ == "__main__":
    main()
