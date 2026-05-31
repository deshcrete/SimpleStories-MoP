"""Strongest evidence that the uniform and non-uniform mixtures learned different
per-persona compositions, controlling for total training.

The size-matched uniform mixture (4,702/persona) and the non-uniform mixture
(geometric, 9,026 ... 1,783) have the SAME total size (~23.5k examples, 735 steps),
so they differ ONLY in composition. For each persona we connect its competence in
the two mixtures: x = how many examples of that persona the mixture saw, y = the
gap to that persona's specialist (less negative = the mixture learned it better).

If every persona's line slopes the same way (more of its own data -> smaller gap),
competence is driven by per-persona example count -> the two mixtures genuinely
learned different compositions. Writes results/plots/composition_evidence.png.
"""

from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np

from data import (
    PERSONAS,
    load_mixture_train_nonuniform,
    load_mixture_train_uniform_matched,
)
from lotp import RESULTS_DIR, load_run_metrics, min_val_loss_step, persona_to_indices

PLOTS_ROOT = RESULTS_DIR / "plots"
PERSONA_COLORS = {
    "noir_detective": "#1f77b4", "fairy_tale": "#ff7f0e",
    "scientific_explainer": "#2ca02c", "absurdist": "#d62728", "epistolary": "#9467bd",
}


def _row(run: str) -> np.ndarray:
    steps, mat, _ = load_run_metrics(run)
    s = min_val_loss_step(run)
    return mat[int(np.where(steps == s)[0][0])]


def main() -> None:
    idx = persona_to_indices()
    spec = {p: _row(p) for p in PERSONAS}
    matched, nonu = _row("mixture_unif_matched"), _row("mixture_exp")

    n_matched = collections.Counter(r["persona"] for r in load_mixture_train_uniform_matched())
    n_nonu = collections.Counter(r["persona"] for r in load_mixture_train_nonuniform())

    fig, ax = plt.subplots(figsize=(9, 6))
    for p in PERSONAS:
        gm = float((matched[idx[p]] - spec[p][idx[p]]).mean())
        gn = float((nonu[idx[p]] - spec[p][idx[p]]).mean())
        xm, xn = n_matched[p], n_nonu[p]
        ax.plot([xm, xn], [gm, gn], color=PERSONA_COLORS[p], linewidth=1.8, zorder=2)
        ax.scatter([xm], [gm], facecolor="white", edgecolor=PERSONA_COLORS[p],
                   linewidth=2, s=90, zorder=3)
        ax.scatter([xn], [gn], color=PERSONA_COLORS[p], s=110, zorder=3,
                   edgecolor="black", linewidth=0.6)
        ax.annotate(p, (xn, gn), textcoords="offset points", xytext=(9, 0),
                    fontsize=9, va="center")

    ax.axvline(4702, color="gray", ls=":", linewidth=1, alpha=0.7)
    ax.text(4702, ax.get_ylim()[1], " matched-uniform\n (4,702 each)", fontsize=8,
            va="top", color="gray")
    # legend proxies
    ax.scatter([], [], facecolor="white", edgecolor="black", linewidth=2, s=90,
               label="matched-uniform mixture (4,702/persona)")
    ax.scatter([], [], color="gray", edgecolor="black", s=110,
               label="non-uniform mixture (its own count)")
    ax.set_xlabel("examples of this persona in the mixture")
    ax.set_ylabel(r"gap to specialist = mean(log $P_{mix}$ - log $P_{spec}$)  (nats/seq)")
    ax.set_title("Same total training (23.5k ex, 735 steps) — each persona's competence\n"
                 "tracks its own example count: the two mixtures learned different compositions")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = PLOTS_ROOT / "composition_evidence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
