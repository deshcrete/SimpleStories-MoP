"""Induced-prior measurement via the per-persona mixture-vs-specialist gap.

The LoTP π-fit (lotp.fit_pi_kl) does NOT recover the mixture's induced prior: it is
blind to P_mix and only reflects the eval-set composition (see analysis.md / the
non-uniform experiment). The induced prior is instead visible directly — on a
persona's held-out test seqs, compare the mixture's log-prob to the base model and
to that persona's dedicated specialist.

Per persona p, at each model's own min-val checkpoint (means over p's test seqs of
the per-sequence summed log-prob):

    gap_p     = mean(log P_mix) - mean(log P_spec_p)              (nats/seq; <0 = mixture worse)
    learn_p   = (mean log P_mix - mean log P_base)
                / (mean log P_spec_p - mean log P_base)
                fraction of the specialist's over-base improvement the mixture got.
                Complexity-normalized (length cancels in the ratio); ~1 = mixture
                learned p as well as the dedicated specialist, 0 = no better than base.
    induced_p = learn_p / Σ_q learn_q                             (normalized to a distribution)

We compare `induced` to the mixture's DATA proportion. Tracking the data proportion
means the mixture faithfully represents the data prior; systematic deviation is the
base-model prior signature (the original research question).

Writes results/lotp/induced_prior{suffix}.json and results/plots/induced_prior.png.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from data import PERSONAS, mixture_exp_weights
from lotp import (
    LOTP_ROOT,
    METRICS_ROOT,
    load_run_metrics,
    min_val_loss_step,
    persona_to_indices,
)

PLOTS_ROOT = LOTP_ROOT.parent / "plots"
PERSONA_COLORS = {
    "noir_detective": "#1f77b4", "fairy_tale": "#ff7f0e",
    "scientific_explainer": "#2ca02c", "absurdist": "#d62728", "epistolary": "#9467bd",
}


def _row_at_min_val(run: str) -> np.ndarray:
    steps, matrix, _ = load_run_metrics(run)
    s = min_val_loss_step(run)
    return matrix[int(np.where(steps == s)[0][0])]


def _base_row() -> np.ndarray:
    # step 0 is the untrained base model, identical across all runs.
    _, matrix, _ = load_run_metrics(PERSONAS[0])
    return matrix[0]


def compute_induced(mixture_run: str, data_props: dict[str, float],
                    idx: dict[str, np.ndarray]) -> dict[str, dict]:
    base = _base_row()
    mix = _row_at_min_val(mixture_run)
    res: dict[str, dict] = {}
    for p in PERSONAS:
        spec = _row_at_min_val(p)
        b = float(base[idx[p]].mean())
        s = float(spec[idx[p]].mean())
        m = float(mix[idx[p]].mean())
        res[p] = {"base": b, "spec": s, "mix": m, "gap": m - s,
                  "learn_frac": (m - b) / (s - b), "data_prop": data_props[p]}
    total = sum(res[p]["learn_frac"] for p in PERSONAS)
    for p in PERSONAS:
        res[p]["induced_weight"] = res[p]["learn_frac"] / total
    return res


def _data_props(run: str) -> dict[str, float]:
    """The training data proportions for a mixture run (uniform unless it's the
    geometric non-uniform mixture)."""
    if run == "mixture_exp":
        return mixture_exp_weights()
    return {p: 1.0 / len(PERSONAS) for p in PERSONAS}


def _configs() -> list[tuple[str, str, dict[str, float]]]:
    out = [("mixture", "", _data_props("mixture"))]
    for d in sorted(METRICS_ROOT.glob("mixture_*")):
        out.append((d.name, d.name.replace("mixture", ""), _data_props(d.name)))
    return out


def _corr_or_none(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r, or None when x has no variance (e.g. uniform data proportions)."""
    if np.std(x) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def plot_induced(all_res: list[tuple[str, dict, dict]]) -> None:
    """One panel per mixture: scatter of the raw per-persona gap (log P_mix -
    log P_spec) against the data proportion. The gap is the robust induced-prior
    signal (it works where π_KL fails); for the non-uniform mixture it tracks the
    data proportion. For the uniform mixture the data proportion is constant, so the
    spread of gaps at x=0.2 is the base-model prior / complexity signature."""
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(all_res), figsize=(6.5 * len(all_res), 5), squeeze=False)
    for ax, (label, res, props) in zip(axes[0], all_res):
        dp = np.array([props[p] for p in PERSONAS])
        gap = np.array([res[p]["gap"] for p in PERSONAS])
        for p in PERSONAS:
            ax.scatter(props[p], res[p]["gap"], s=130, color=PERSONA_COLORS[p],
                       edgecolor="black", linewidth=0.7, zorder=3)
            ax.annotate(p, (props[p], res[p]["gap"]), textcoords="offset points",
                        xytext=(8, 5), fontsize=8)
        r = _corr_or_none(dp, gap)
        if r is not None:                       # add a regression line when x varies
            b1, b0 = np.polyfit(dp, gap, 1)
            xs = np.linspace(dp.min(), dp.max(), 2)
            ax.plot(xs, b0 + b1 * xs, color="gray", ls="--", linewidth=1)
        ax.set_xlabel("data proportion in the mixture")
        ax.set_ylabel(r"gap = mean(log $P_{mix}$ - log $P_{spec}$)  (nats/seq)")
        rstr = "n/a (uniform data)" if r is None else f"Pearson r = {r:+.2f}"
        ax.set_title(f"{label}\ninduced-prior gap vs data proportion  ({rstr})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_ROOT / "induced_prior.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    idx = persona_to_indices()
    all_res = []
    for run, suffix, props in _configs():
        res = compute_induced(run, props, idx)
        (LOTP_ROOT / f"induced_prior{suffix}.json").write_text(json.dumps(res, indent=2))
        dp = np.array([props[p] for p in PERSONAS])
        gap = np.array([res[p]["gap"] for p in PERSONAS])
        print(f"\n=== {run} ===")
        print(f"{'persona':22}{'data':>8}{'gap':>9}{'learn_frac':>12}")
        for p in PERSONAS:
            print(f"{p:22}{props[p]:8.3f}{res[p]['gap']:9.1f}{res[p]['learn_frac']:12.3f}")
        r = _corr_or_none(dp, gap)
        print(f"  corr(data, gap) = {'n/a (uniform data)' if r is None else f'{r:+.3f}'}"
              "   [gap is the primary measure; learn_frac is a noisy complexity-normalized diagnostic]")
        all_res.append((run, res, props))
    plot_induced(all_res)


if __name__ == "__main__":
    main()
