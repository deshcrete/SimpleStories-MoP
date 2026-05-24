"""Trajectory plots for the persona experiment.

Reads:
    results/lotp/step_aligned.jsonl         — π fits, residual, hull-escapes per pair
    results/lotp/loss_aligned.json          — endpoint annotation
    results/metrics/<run>/checkpoints.jsonl — per-checkpoint param-norms + total log-P
    results/metrics/<run>/per_sequence_logp.npy + steps.npy
                                            — for own-held-out log-P trajectories

Writes (to results/plots/):
    pi_kl.png              — π_KL(persona) vs mixture step  (PRIMARY fit)
    pi_residual.png        — π_residual(persona) vs mixture step  (diagnostic)
    residual_rms.png       — residual_rms (both fits) vs mixture step
    hull_escapes.png       — hull-escape count vs mixture step
    param_norms.png        — ‖θ‖ and ‖θ - θ_base‖ for all 6 runs vs their own step
    own_held_out_logp.png  — own-held-out Σ log P trajectory per run
    logprob_overfit.png    — per-persona: own specialist vs mixture (+ other
                              specialists in gray) on this persona's 150 seqs.
                              Direct visualization of overfitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import PERSONAS
from lotp import (
    RUNS,
    fit_pi_kl,
    load_run_metrics,
    min_loss_step_own_held_out,
    persona_to_indices,
    stack_log_p_for_alignment,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOTP_ROOT = RESULTS_DIR / "lotp"
METRICS_ROOT = RESULTS_DIR / "metrics"
PLOTS_ROOT = RESULTS_DIR / "plots"

# Colormap keyed by persona; mixture gets a distinct dark line.
PERSONA_COLORS = {
    "noir_detective":       "#1f77b4",
    "fairy_tale":           "#ff7f0e",
    "scientific_explainer": "#2ca02c",
    "absurdist":            "#d62728",
    "epistolary":           "#9467bd",
}
MIXTURE_COLOR = "#222222"


def _load_step_aligned() -> list[dict]:
    rows: list[dict] = []
    with (LOTP_ROOT / "step_aligned.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_loss_aligned() -> dict:
    return json.loads((LOTP_ROOT / "loss_aligned.json").read_text())


def _load_checkpoint_rows(run: str) -> list[dict]:
    path = METRICS_ROOT / run / "checkpoints.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f]


def _save(fig, name: str) -> Path:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = PLOTS_ROOT / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------- π trajectory ----------

def plot_pi_trajectory(rows: list[dict], which_fit: str, loss_aligned: dict, fname: str) -> Path:
    """which_fit ∈ {'kl_fit', 'residual_fit'}."""
    steps = np.array([r["alignment"]["mixture_step"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    for persona in PERSONAS:
        ys = np.array([r[which_fit]["pi"][persona] for r in rows])
        ax.plot(steps, ys, label=persona, color=PERSONA_COLORS[persona], linewidth=1.8)
    ax.axhline(0.2, ls="--", color="gray", linewidth=0.8, label="uniform 1/5")

    # mark the loss-aligned mixture step (the endpoint reported separately).
    la_step = loss_aligned["alignment"]["mixture_step"]
    ax.axvline(la_step, ls=":", color="black", linewidth=0.8, alpha=0.6)
    ax.text(la_step, 1.02, f"loss-aligned step={la_step}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("mixture training step")
    ax.set_ylabel(r"$\pi_i$")
    ax.set_ylim(-0.02, 1.02)
    title = {
        "kl_fit":       "KL fit  (primary, KL(empirical || M))",
        "residual_fit": "Residual fit  (diagnostic, spec L(π))",
    }[which_fit]
    ax.set_title(f"Fitted π trajectory — {title}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, fname)


# ---------- residual_rms ----------

def plot_residual_rms(rows: list[dict], loss_aligned: dict) -> Path:
    steps = np.array([r["alignment"]["mixture_step"] for r in rows])
    res_residual = np.array([r["residual_fit"]["residual_rms"] for r in rows])
    res_kl = np.array([r["kl_fit"]["residual_rms"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, res_kl, label="KL fit  (primary)", color="#1f77b4")
    ax.plot(steps, res_residual, label="residual fit  (diagnostic)", color="#d62728", linestyle="--")

    i_best = int(np.argmin(res_kl))
    ax.scatter([steps[i_best]], [res_kl[i_best]], color="#1f77b4", s=40, zorder=5)
    ax.annotate(f"min={res_kl[i_best]:.1f}\n@ step {steps[i_best]}",
                (steps[i_best], res_kl[i_best]),
                textcoords="offset points", xytext=(10, 10), fontsize=8)

    la_step = loss_aligned["alignment"]["mixture_step"]
    ax.axvline(la_step, ls=":", color="black", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("mixture training step")
    ax.set_ylabel("residual RMS (nats)")
    ax.set_title("LoTP residual — how well the identity holds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "residual_rms.png")


# ---------- hull escapes ----------

def plot_hull_escapes(rows: list[dict], loss_aligned: dict) -> Path:
    steps = np.array([r["alignment"]["mixture_step"] for r in rows])
    n_esc = np.array([r["hull_escapes"]["n_escapes"] for r in rows])
    max_gap = np.array([r["hull_escapes"]["max_gap"] for r in rows])
    n_seqs = rows[0]["hull_escapes"]["n_seqs"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, n_esc, color="#d62728", linewidth=1.8, label="escapes")
    ax1.set_xlabel("mixture training step")
    ax1.set_ylabel(f"# escapes (out of {n_seqs})", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, max_gap, color="#1f77b4", linewidth=1.4, linestyle="--", label="max gap")
    ax2.set_ylabel("max gap = max_i log P_i - log P_mix  (nats)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    la_step = loss_aligned["alignment"]["mixture_step"]
    ax1.axvline(la_step, ls=":", color="black", linewidth=0.8, alpha=0.6)

    ax1.set_title("Convex-hull escapes (τ=0) over training")
    return _save(fig, "hull_escapes.png")


# ---------- parameter norms ----------

def plot_param_norms() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    for run in RUNS:
        rows = _load_checkpoint_rows(run)
        steps = np.array([r["step"] for r in rows])
        pn = np.array([r["param_norm"] for r in rows])
        dn = np.array([r["dev_norm"] for r in rows])
        color = MIXTURE_COLOR if run == "mixture" else PERSONA_COLORS[run]
        lw = 2.2 if run == "mixture" else 1.5
        axes[0].plot(steps, pn, label=run, color=color, linewidth=lw)
        axes[1].plot(steps, dn, label=run, color=color, linewidth=lw)
    axes[0].set_title(r"$\|\theta\|_2$ per checkpoint")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$\|\theta\|_2$")
    axes[0].grid(alpha=0.3)
    axes[1].set_title(r"$\|\theta - \theta_{\mathrm{base}}\|_2$ per checkpoint")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"$\|\theta - \theta_{\mathrm{base}}\|_2$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")
    return _save(fig, "param_norms.png")


# ---------- own held-out log-P ----------

def plot_own_held_out_logp() -> Path:
    idx_map = persona_to_indices()
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in RUNS:
        steps, matrix, _ = load_run_metrics(run)
        if run == "mixture":
            own_logp = matrix.sum(axis=1)
        else:
            own_logp = matrix[:, idx_map[run]].sum(axis=1)
        color = MIXTURE_COLOR if run == "mixture" else PERSONA_COLORS[run]
        lw = 2.2 if run == "mixture" else 1.5
        ax.plot(steps, own_logp, label=run, color=color, linewidth=lw)
        # mark each run's own min-loss step.
        i_best = int(np.argmax(own_logp))
        ax.scatter([steps[i_best]], [own_logp[i_best]], color=color, s=30, zorder=5)

    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\sum_j \log P_\theta(x_j)$  (own held-out)")
    ax.set_title("Own held-out total log-P per checkpoint  (dot = min-loss step)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "own_held_out_logp.png")


# ---------- per-persona log-P (overfitting view) ----------

def plot_logprob_overfit() -> Path:
    """For each persona, plot Σ log P over that persona's 150 inference seqs vs
    training step, for: (a) the own-persona specialist (heavy, persona color),
    (b) the mixture model (heavy black dashed), (c) the other 4 specialists
    (light gray) for context. Dot marks each highlighted line's max.

    Reads: results/metrics/<run>/{per_sequence_logp.npy, steps.npy}.
    """
    idx_map = persona_to_indices()
    run_data = {run: load_run_metrics(run) for run in RUNS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for ax, persona in zip(axes_flat, PERSONAS):
        own_idx = idx_map[persona]

        # other 4 specialists (context)
        for other in PERSONAS:
            if other == persona:
                continue
            steps, mat, _ = run_data[other]
            logp = mat[:, own_idx].sum(axis=1)
            ax.plot(steps, logp, color="lightgray", linewidth=0.9, alpha=0.8, zorder=1,
                    label=f"{other} specialist" if other == PERSONAS[0] or
                          (other == PERSONAS[1] and persona == PERSONAS[0]) else None)

        # mixture
        steps, mat, _ = run_data["mixture"]
        mix_logp = mat[:, own_idx].sum(axis=1)
        ax.plot(steps, mix_logp, color=MIXTURE_COLOR, linewidth=2.2, linestyle="--",
                label="mixture", zorder=2)
        i_mix = int(np.argmax(mix_logp))
        ax.scatter([steps[i_mix]], [mix_logp[i_mix]], color=MIXTURE_COLOR, s=35, zorder=5)

        # own specialist
        steps, mat, _ = run_data[persona]
        own_logp = mat[:, own_idx].sum(axis=1)
        ax.plot(steps, own_logp, color=PERSONA_COLORS[persona], linewidth=2.4,
                label=f"{persona} (own)", zorder=3)
        i_own = int(np.argmax(own_logp))
        ax.scatter([steps[i_own]], [own_logp[i_own]], color=PERSONA_COLORS[persona],
                   s=45, zorder=5)
        ax.axvline(steps[i_own], color=PERSONA_COLORS[persona], linewidth=0.6,
                   linestyle=":", alpha=0.5)

        ax.set_title(persona, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel(r"$\sum_j \log P_\theta(x_j)$  (own 150 seqs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    # Hide the unused 6th subplot
    axes_flat[5].axis("off")
    fig.suptitle("Specialist (own) vs mixture log-P on each persona's 150 held-out seqs  "
                 "(dots = max; vertical = specialist's peak step)",
                 y=1.00)
    fig.tight_layout()
    return _save(fig, "logprob_overfit.png")


# ---------- persona complexity analysis ----------

def plot_complexity_analysis() -> Path:
    """Per-persona complexity under the base model and its relationship to
    π_KL and to the mix-vs-spec gap.

    Top row:
      - base log P / token per persona (bars; sorted easiest→hardest)
      - dev_norm at the specialist's peak step (bars)
    Bottom row:
      - scatter: base/token vs π_KL  (does the prior signature track complexity?)
      - scatter: base/token vs (log P_mix − log P_spec) per seq at loss-aligned
        (the U-shape: specialists win at extremes of complexity)
    """
    from data import load_inference_set, tokenize_story
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-5M")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rows = load_inference_set()
    lens = np.array([len(tokenize_story(r["story"], tok)) for r in rows])
    n_scored = np.maximum(lens - 1, 1)
    idx_map = persona_to_indices()

    # Base-model per-sequence log P (step 0 is identical across all six runs).
    _, mat_any, _ = load_run_metrics(PERSONAS[0])
    base_per_seq = mat_any[0]

    # Loss-aligned spec/mix log P per sequence + π_KL.
    la = json.loads((LOTP_ROOT / "loss_aligned.json").read_text())
    pi_kl_dict = la["kl_fit"]["pi"]
    spec_logp_per_persona: dict[str, np.ndarray] = {}
    for p in PERSONAS:
        steps_p, mat_p, _ = load_run_metrics(p)
        s = la["alignment"]["specialist_steps"][p]
        spec_logp_per_persona[p] = mat_p[int(np.where(steps_p == s)[0][0])]
    steps_m, mat_m, _ = load_run_metrics("mixture")
    mix_logp = mat_m[int(np.where(steps_m == la["alignment"]["mixture_step"])[0][0])]

    # Per-persona dev_norm at peak (own held-out)
    dev_norm_peak: dict[str, float] = {}
    spec_peak_logp: dict[str, float] = {}
    for p in PERSONAS:
        crows = [json.loads(line) for line in (METRICS_ROOT / p / "checkpoints.jsonl").open()]
        steps_p, mat_p, _ = load_run_metrics(p)
        own = mat_p[:, idx_map[p]].sum(axis=1)
        i_peak = int(np.argmax(own))
        dev_norm_peak[p] = float(crows[i_peak]["dev_norm"])
        spec_peak_logp[p] = float(own[i_peak])

    # Per-persona aggregates
    base_per_tok = {
        p: float((base_per_seq[idx_map[p]] / n_scored[idx_map[p]]).mean())
        for p in PERSONAS
    }
    # Per-seq mean mix and own-specialist log P on each persona's slice
    mix_per_seq = {p: float(mix_logp[idx_map[p]].mean()) for p in PERSONAS}
    spec_per_seq = {p: float(spec_logp_per_persona[p][idx_map[p]].mean()) for p in PERSONAS}
    gap = {p: mix_per_seq[p] - spec_per_seq[p] for p in PERSONAS}   # >0 mix wins

    # Sort easiest -> hardest by base/token
    order = sorted(PERSONAS, key=lambda p: -base_per_tok[p])  # least negative first

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ----- (0,0) base log P / token, sorted easiest->hardest -----
    ax = axes[0, 0]
    xs = np.arange(len(order))
    vals = [base_per_tok[p] for p in order]
    bars = ax.bar(xs, vals, color=[PERSONA_COLORS[p] for p in order],
                  alpha=0.8, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v - 0.1, f"{v:.2f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("base log P / token  (nats)")
    ax.set_title("A.  Persona complexity under the base model  (easier → harder)")
    ax.grid(alpha=0.3, axis="y")

    # ----- (0,1) dev_norm at peak -----
    ax = axes[0, 1]
    vals = [dev_norm_peak[p] for p in order]
    ax.bar(xs, vals, color=[PERSONA_COLORS[p] for p in order],
           alpha=0.8, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\|\theta_{\mathrm{spec}}^{*} - \theta_{\mathrm{base}}\|_2$")
    ax.set_title("B.  Specialist parameter move at peak — tracks complexity")
    ax.grid(alpha=0.3, axis="y")

    # ----- (1,0) scatter: base/token vs π_KL -----
    ax = axes[1, 0]
    xs = [base_per_tok[p] for p in PERSONAS]
    ys = [pi_kl_dict[p] for p in PERSONAS]
    for p, x, y in zip(PERSONAS, xs, ys):
        ax.scatter(x, y, s=110, color=PERSONA_COLORS[p], edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(0.2, ls="--", color="gray", linewidth=0.9, label="uniform 1/5")
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel("base log P / token  (nats)")
    ax.set_ylabel(r"$\pi_{\mathrm{KL}}$  (loss-aligned)")
    ax.set_title(f"C.  Does the recovered prior track complexity?  (Pearson r = {r:+.2f}, n=5)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # ----- (1,1) U-shape: base/token vs mix-spec gap -----
    ax = axes[1, 1]
    ys = [gap[p] for p in PERSONAS]
    for p, x, y in zip(PERSONAS, xs, ys):
        ax.scatter(x, y, s=110, color=PERSONA_COLORS[p], edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(0, ls="--", color="gray", linewidth=0.9, label="tie")
    ax.set_xlabel("base log P / token  (nats)")
    ax.set_ylabel(r"$\log P_{\mathrm{mix}} - \log P_{\mathrm{spec}}$  per seq  (mix wins if > 0)")
    ax.set_title("D.  Mix-vs-spec gap at loss-aligned — specialists win at extremes")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "complexity_analysis.png")


# ---------- hull escapes vs each fit's loss ----------

def plot_escapes_vs_fits() -> Path:
    """3-panel view of why the KL fit is insensitive to hull escapes:

      A) hull-escape count + max gap over training
      B) residual fit's residual_rms (spec L(π)) — tracks max_gap closely
      C) KL fit's avg log M (its actual objective) — locks once specialists
         pin at step 160; insensitive to the mixture's late-stage overfit.
    """
    rows = _load_step_aligned()
    steps = np.array([r["alignment"]["mixture_step"] for r in rows])
    escapes = np.array([r["hull_escapes"]["n_escapes"] for r in rows])
    max_gap = np.array([r["hull_escapes"]["max_gap"] for r in rows])
    res_rms = np.array([r["residual_fit"]["residual_rms"] for r in rows])
    kl_ll = np.array([r["kl_fit"]["avg_loglik"] for r in rows])

    # Correlations for the panel subtitles.
    r_escape_res = float(np.corrcoef(escapes, res_rms)[0, 1])
    r_gap_res = float(np.corrcoef(max_gap, res_rms)[0, 1])
    r_escape_kl = float(np.corrcoef(escapes, kl_ll)[0, 1])

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)

    # A — escapes + max_gap
    ax = axes[0]
    l1, = ax.plot(steps, escapes, color="#d62728", linewidth=2, label="# escapes")
    ax.set_ylabel("# escapes (out of 750)", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax2 = ax.twinx()
    l2, = ax2.plot(steps, max_gap, color="#1f77b4", linewidth=1.4, linestyle="--",
                   label="max gap (nats)")
    ax2.set_ylabel(r"max gap = $\max_i \log P_i - \log P_{\mathrm{mix}}$  (nats)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("A.  Hull escapes — count (red) and worst-case gap (blue, dashed)")
    ax.grid(alpha=0.3)

    # B — residual fit's residual_rms (the spec L(π))
    ax = axes[1]
    ax.plot(steps, res_rms, color="#d62728", linewidth=2)
    ax.set_ylabel(r"residual fit  residual_rms  (nats)")
    ax.set_title(f"B.  Residual fit's loss tracks the worst-case gap  "
                 f"(corr(max_gap, residual_rms) = {r_gap_res:+.2f})")
    ax.grid(alpha=0.3)

    # C — KL fit's avg log M (objective)
    ax = axes[2]
    ax.plot(steps, kl_ll, color="#1f77b4", linewidth=2)
    # Annotate the lock point.
    lock_idx = int(np.argmin(np.abs(steps - 180)))
    ax.axvline(steps[lock_idx], color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(steps[lock_idx], kl_ll[lock_idx], "  specialists pin at step 160\n  → KL objective locks",
            fontsize=8, va="center")
    ax.set_ylabel(r"KL fit  $\mathrm{avg}_j \log M(x_j)$  (objective)")
    ax.set_xlabel("mixture training step")
    ax.set_title(f"C.  KL fit's objective is insensitive to escapes  "
                 f"(corr(escapes, avg_log_M) = {r_escape_kl:+.2f})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "escapes_vs_fits.png")


# ---------- bootstrap error bars on π_KL ----------

def plot_pi_kl_errorbars(n_bootstrap: int = 500, seed: int = 0) -> Path:
    """Bootstrap-resample the 750-seq inference set (with replacement), refit
    π_KL on each resample, and plot the loss-aligned π with 95% CI error bars.

    Saves the raw bootstrap stats next to the plot for reproducibility.
    """
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

    pi_ref, _, _ = fit_pi_kl(spec, mix)

    rng = np.random.default_rng(seed)
    n_seqs = mix.size
    boot_pi = np.zeros((n_bootstrap, len(PERSONAS)), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seqs, size=n_seqs)
        pi_b, _, _ = fit_pi_kl(spec[:, idx], mix[idx])
        boot_pi[b] = pi_b

    p_low = np.percentile(boot_pi, 2.5, axis=0)
    p_high = np.percentile(boot_pi, 97.5, axis=0)
    err_low = pi_ref - p_low
    err_high = p_high - pi_ref

    # Persist raw stats next to the plot.
    stats = {
        "alignment": loss_pair,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "personas": PERSONAS,
        "pi_full_data": pi_ref.tolist(),
        "bootstrap_mean": boot_pi.mean(axis=0).tolist(),
        "bootstrap_std": boot_pi.std(axis=0).tolist(),
        "ci_low_2.5": p_low.tolist(),
        "ci_high_97.5": p_high.tolist(),
    }
    (LOTP_ROOT / "pi_kl_bootstrap.json").write_text(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERSONAS))
    bars = ax.bar(x, pi_ref,
                  color=[PERSONA_COLORS[p] for p in PERSONAS],
                  alpha=0.75, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, pi_ref, yerr=[err_low, err_high],
                fmt="none", ecolor="black", capsize=5, linewidth=1.4, zorder=5)
    ax.axhline(0.2, ls="--", color="gray", linewidth=1.0, label="uniform 1/5")

    # Per-bar text: π ± std and CI hit/miss
    for i, (b_rect, p) in enumerate(zip(bars, PERSONAS)):
        std = float(boot_pi[:, i].std())
        contains_uniform = p_low[i] <= 0.2 <= p_high[i]
        sig = "" if contains_uniform else " *"
        ax.text(i, pi_ref[i] + err_high[i] + 0.002,
                f"{pi_ref[i]:.4f}\n±{std:.4f}{sig}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(PERSONAS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\pi_i$")
    ax.set_ylim(0, max(p_high.max() + 0.03, 0.25))
    ax.set_title(f"KL-fit π with bootstrap 95% CI  ({n_bootstrap} resamples of 750 seqs)\n"
                 r"$*$ = 95% CI excludes the uniform $1/5$ reference")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "pi_kl_errorbars.png")


# ---------- main ----------

def main() -> None:
    rows = _load_step_aligned()
    la = _load_loss_aligned()

    outs = [
        plot_pi_trajectory(rows, "kl_fit", la, "pi_kl.png"),
        plot_pi_trajectory(rows, "residual_fit", la, "pi_residual.png"),
        plot_residual_rms(rows, la),
        plot_hull_escapes(rows, la),
        plot_param_norms(),
        plot_own_held_out_logp(),
        plot_logprob_overfit(),
        plot_escapes_vs_fits(),
        plot_pi_kl_errorbars(),
        plot_complexity_analysis(),
    ]
    for p in outs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
