"""Diagnostic: π_KL recovers the EVAL-SET composition, not the mixture's prior.

Confirms the failure mode found in the non-uniform experiment. `fit_pi_kl` maximizes
the test-sample likelihood under Σ_i π_i P_i; because specialists have near-disjoint
support (each ~100 nats better on its own persona), the matched specialist dominates
the logsumexp and the MLE just recovers how often each persona appears in the eval
set. So re-fitting π_KL on a NON-uniform subsample of the test set should make π_KL
track that subsample's composition — and it never references any mixture.

We subsample the test set to several target compositions and show π_KL ≈ composition.
Writes results/lotp/eval_composition_probe.json and results/plots/eval_composition_probe.png.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from data import PERSONAS, mixture_exp_weights
from lotp import (
    LOTP_ROOT,
    fit_pi_kl,
    load_run_metrics,
    min_val_loss_step,
    persona_to_indices,
)

PLOTS_ROOT = LOTP_ROOT.parent / "plots"


def _specialist_rows() -> np.ndarray:
    rows = []
    for p in PERSONAS:
        steps, mat, _ = load_run_metrics(p)
        s = min_val_loss_step(p)
        rows.append(mat[int(np.where(steps == s)[0][0])])
    return np.stack(rows)   # (5, n_seqs)


def _subsample(idx: dict[str, np.ndarray], weights: dict[str, float]) -> tuple[np.ndarray, dict]:
    """Subsample the test set to the target persona composition, anchoring the
    largest-weight persona on all of its 500 test seqs."""
    anchor = max(PERSONAS, key=lambda p: weights[p])
    anchor_n = len(idx[anchor])
    sel, counts = [], {}
    for p in PERSONAS:
        n = min(round(anchor_n * weights[p] / weights[anchor]), len(idx[p]))
        sel.extend(idx[p][:n].tolist())
        counts[p] = n
    total = sum(counts.values())
    comp = {p: counts[p] / total for p in PERSONAS}
    return np.array(sel, dtype=np.int64), comp


def main() -> None:
    idx = persona_to_indices()
    spec = _specialist_rows()
    dummy_mix = np.zeros(spec.shape[1])   # fit_pi_kl ignores the mixture argument

    geom = mixture_exp_weights()
    geom_rev = {p: geom[q] for p, q in zip(PERSONAS, PERSONAS[::-1])}
    targets = {
        "uniform": {p: 0.2 for p in PERSONAS},
        "geom_1.5": geom,
        "geom_1.5_reversed": geom_rev,
    }

    out = {}
    for name, w in targets.items():
        sel, comp = _subsample(idx, w)
        pi, _, _ = fit_pi_kl(spec[:, sel], dummy_mix[sel])
        pi_d = dict(zip(PERSONAS, pi.tolist()))
        out[name] = {"eval_composition": comp, "pi_kl": pi_d, "n_seqs": int(len(sel))}
        max_dev = max(abs(pi_d[p] - comp[p]) for p in PERSONAS)
        print(f"\n[{name}]  N={len(sel)}   max|π_KL - eval_comp| = {max_dev:.4f}")
        print(f"{'persona':22}{'eval_comp':>10}{'π_KL':>9}")
        for p in PERSONAS:
            print(f"{p:22}{comp[p]:10.3f}{pi_d[p]:9.3f}")
    (LOTP_ROOT / "eval_composition_probe.json").write_text(json.dumps(out, indent=2))

    # plot: π_KL vs eval composition for each target
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 4.5), squeeze=False)
    for ax, (name, d) in zip(axes[0], out.items()):
        x = np.arange(len(PERSONAS)); wbar = 0.38
        comp = np.array([d["eval_composition"][p] for p in PERSONAS])
        pik = np.array([d["pi_kl"][p] for p in PERSONAS])
        ax.bar(x - wbar / 2, comp, wbar, label="eval composition", color="0.65",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + wbar / 2, pik, wbar, label="π_KL", color="#1f77b4",
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(PERSONAS, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("proportion")
        ax.set_title(f"eval = {name}\nπ_KL tracks the EVAL composition")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("π_KL recovers the eval-set composition, not the mixture's prior", y=1.02)
    fig.tight_layout()
    out_png = PLOTS_ROOT / "eval_composition_probe.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
