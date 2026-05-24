"""Probe whether the loss-aligned π fit is a robust optimum or a degeneracy.

Runs both the spec's residual minimization AND mixture-EM from many seeded
inits, then reports the L1 distance from each init's converged π to the
uniform-init reference. If all inits converge to the same π for the residual
fit, the optimum is well-defined; if not, π has a flat direction.

Outputs:
    results/lotp/loss_aligned_perturb.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import PERSONAS
from lotp import (
    LOTP_ROOT,
    RUNS,
    fit_pi_kl,
    fit_pi_residual,
    hull_escape_count,
    load_run_metrics,
    min_loss_step_own_held_out,
    persona_to_indices,
    stack_log_p_for_alignment,
)

N_RANDOM = 20
RNG_SEED = 0


def build_loss_aligned():
    idx_map = persona_to_indices()
    run_data = {run: load_run_metrics(run) for run in RUNS}
    loss_pair = {
        "mixture_step": min_loss_step_own_held_out(
            "mixture", run_data["mixture"][0], run_data["mixture"][1], idx_map),
        "specialist_steps": {
            p: min_loss_step_own_held_out(p, run_data[p][0], run_data[p][1], idx_map)
            for p in PERSONAS
        },
    }
    spec, mix = stack_log_p_for_alignment(run_data, loss_pair)
    return loss_pair, spec, mix, idx_map


def _inits(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    inits: list[tuple[str, np.ndarray]] = [("uniform", np.full(5, 0.2))]
    for i, persona in enumerate(PERSONAS):
        v = np.full(5, 0.05)
        v[i] = 0.8
        inits.append((f"corner_{persona}", v))
    for k in range(N_RANDOM):
        inits.append((f"dirichlet_{k}", rng.dirichlet(np.ones(5))))
    return inits


def _sweep(fit_fn, spec: np.ndarray, mix: np.ndarray) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    inits = _inits(rng)
    fits: list[dict] = []
    ref_pi = None
    for kind, init in inits:
        # both fit functions return (pi, residual_rms, avg_loglik)
        pi, res, ll = fit_fn(spec, mix, init_pi=init)
        if ref_pi is None and kind == "uniform":
            ref_pi = pi
        fits.append({
            "init_kind": kind,
            "init_pi": init.tolist(),
            "final_pi": pi.tolist(),
            "residual_rms": res,
            "avg_loglik": ll,
        })
    for f in fits:
        f["l1_to_ref"] = float(np.abs(np.array(f["final_pi"]) - ref_pi).sum())
    return fits, ref_pi


def _print_table(name: str, fits: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print(f"{'init':<22} {'residual':>10}  {'L1→ref':>8}  final π")
    print("-" * 95)
    for f in fits:
        pi_str = "[" + " ".join(f"{v:.4f}" for v in f["final_pi"]) + "]"
        print(f"{f['init_kind']:<22} {f['residual_rms']:>10.4f}  {f['l1_to_ref']:>8.5f}  {pi_str}")
    l1 = np.array([f["l1_to_ref"] for f in fits])
    print(f"L1 to uniform-init ref:  max={l1.max():.5f}  mean={l1.mean():.5f}  "
          f"({'robust' if l1.max() < 1e-3 else 'degenerate'})")


def main() -> None:
    loss_pair, spec, mix, _ = build_loss_aligned()
    print("loss-aligned alignment:", loss_pair)

    fits_kl, ref_kl = _sweep(fit_pi_kl, spec, mix)
    _print_table("KL fit (primary; KL(empirical || M))", fits_kl)

    fits_residual, ref_residual = _sweep(fit_pi_residual, spec, mix)
    _print_table("residual fit (diagnostic; spec's L(π))", fits_residual)

    out = {
        "alignment": loss_pair,
        "personas": PERSONAS,
        "kl_fit": {"reference_pi": ref_kl.tolist(), "fits": fits_kl},
        "residual_fit": {"reference_pi": ref_residual.tolist(), "fits": fits_residual},
        "hull_escapes": hull_escape_count(spec, mix, tau=0.0),
    }
    out_path = LOTP_ROOT / "loss_aligned_perturb.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
