"""Direct LoTP linear-system fit (probability space) with bootstrap CIs and
init-perturbation robustness, for the uniform and non-uniform mixtures.

Unlike π_KL (which recovers the eval-set composition) and the residual fit (which
is degenerate), the raw linear system `P_mix = Σ_i π_i P_i` recovers the prior's
RANKING (compressed toward uniform). Here we:
  * fit π_linear at each mixture's converged (min-val) checkpoint,
  * bootstrap-resample the test sequences to get 95% CIs,
  * re-solve from many seeded inits to confirm the optimum is unique (it should be:
    the objective is a convex least-squares on the simplex).

Writes results/lotp/lotp_linear.json and results/plots/lotp_linear.png.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from data import PERSONAS, mixture_exp_weights
from lotp import (
    LOTP_ROOT,
    RESULTS_DIR,
    fit_pi_linear,
    load_run_metrics,
    min_val_loss_step,
    persona_to_indices,
    stack_log_p_for_alignment,
)

PLOTS_ROOT = RESULTS_DIR / "plots"
N_BOOT = 500
N_DIRICHLET = 20
SEED = 0
PERSONA_COLORS = {
    "noir_detective": "#1f77b4", "fairy_tale": "#ff7f0e",
    "scientific_explainer": "#2ca02c", "absurdist": "#d62728", "epistolary": "#9467bd",
}


def _aligned(run: str) -> tuple[np.ndarray, np.ndarray]:
    run_data = {r: load_run_metrics(r) for r in PERSONAS + [run]}
    pair = {"mixture_step": min_val_loss_step(run),
            "specialist_steps": {p: min_val_loss_step(p) for p in PERSONAS}}
    return stack_log_p_for_alignment(run_data, pair, run)


def _data_props(run: str) -> dict[str, float]:
    return mixture_exp_weights() if run == "mixture_exp" else {p: 1.0 / len(PERSONAS) for p in PERSONAS}


def _bootstrap(spec: np.ndarray, mix: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    J = mix.size
    out = np.zeros((N_BOOT, len(PERSONAS)))
    for b in range(N_BOOT):
        idx = rng.integers(0, J, size=J)
        out[b], _ = fit_pi_linear(spec[:, idx], mix[idx])
    return out


def _perturb(spec: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, float]:
    """Re-solve from uniform + 5 corners + N Dirichlet inits; return (uniform-init
    reference π, max L1 distance any init lands from it)."""
    rng = np.random.default_rng(SEED)
    inits: list[np.ndarray] = [np.full(len(PERSONAS), 1.0 / len(PERSONAS))]
    for k in range(len(PERSONAS)):
        v = np.full(len(PERSONAS), 0.05); v[k] = 0.8
        inits.append(v)
    for _ in range(N_DIRICHLET):
        inits.append(rng.dirichlet(np.ones(len(PERSONAS))))
    ref, _ = fit_pi_linear(spec, mix, init_pi=inits[0])
    max_l1 = 0.0
    for init in inits:
        pi, _ = fit_pi_linear(spec, mix, init_pi=init)
        max_l1 = max(max_l1, float(np.abs(pi - ref).sum()))
    return ref, max_l1


def main() -> None:
    configs = [("mixture", "uniform mixture"), ("mixture_exp", "non-uniform mixture")]
    results = {}
    fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5), squeeze=False)
    for ax, (run, label) in zip(axes[0], configs):
        spec, mix = _aligned(run)
        props = _data_props(run)
        pi, rms = fit_pi_linear(spec, mix)
        boot = _bootstrap(spec, mix)
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        ref, max_l1 = _perturb(spec, mix)

        dp = np.array([props[p] for p in PERSONAS])
        r = float(np.corrcoef(dp, pi)[0, 1]) if np.std(dp) > 1e-9 else None
        results[run] = {
            "pi_linear": pi.tolist(), "prob_residual_rms": rms,
            "ci_low_2.5": lo.tolist(), "ci_high_97.5": hi.tolist(),
            "bootstrap_std": boot.std(axis=0).tolist(),
            "data_proportion": dp.tolist(), "corr_data_pi": r,
            "perturb_max_l1": max_l1,
        }

        x = np.arange(len(PERSONAS)); w = 0.38
        ax.bar(x - w / 2, dp, w, label="data proportion", color="0.65",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, pi, w, label=r"$\pi_{linear}$",
               color=[PERSONA_COLORS[p] for p in PERSONAS], edgecolor="black", linewidth=0.5)
        ax.errorbar(x + w / 2, pi, yerr=[pi - lo, hi - pi], fmt="none", ecolor="black",
                    capsize=4, linewidth=1.3, zorder=5)
        ax.axhline(0.2, ls="--", color="gray", linewidth=0.8, label="uniform 1/5")
        for i in range(len(PERSONAS)):
            ax.text(i + w / 2, hi[i] + 0.004, f"±{boot[:, i].std():.3f}", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(PERSONAS, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("proportion")
        rstr = "n/a" if r is None else f"r={r:+.2f}"
        ax.set_title(f"{label}\n$\\pi_{{linear}}$ vs data ({rstr}; 95% CI bars; "
                     f"init max L1={max_l1:.1e})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Direct LoTP linear system (probability space): recovers the ranking, "
                 "compressed toward uniform", y=1.03)
    fig.tight_layout()
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = PLOTS_ROOT / "lotp_linear.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    (LOTP_ROOT / "lotp_linear.json").write_text(json.dumps(results, indent=2))

    for run, res in results.items():
        robust = "robust (unique optimum)" if res["perturb_max_l1"] < 1e-3 else "NOT robust"
        print(f"{run:14} corr={res['corr_data_pi']}  perturb max L1={res['perturb_max_l1']:.2e}  -> {robust}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
