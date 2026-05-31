"""Trajectory plots for the single-epoch overfitting experiment.

All x-axes are FRACTIONAL EPOCH (= step / steps_per_epoch, 0 -> 1). On the epoch
axis the specialist and the mixture have identical per-persona exposure at every
point (each persona's stories seen `epoch` times in both models), so it is the
fair axis for comparing them — see task_plan.md.

Reads:
    results/checkpoints/<run>/train_log.jsonl   — per-step train loss + per-checkpoint
                                                   val losses (val_own / val_mix)
    results/checkpoints/<run>/run_config.json   — steps_per_epoch (step -> epoch)
    results/lotp/step_aligned.jsonl             — π fits, residual, hull-escapes per pair
    results/lotp/loss_aligned.json              — endpoint (min-val) annotation
    results/metrics/<run>/checkpoints.jsonl     — per-checkpoint param-norms + total log-P
    results/metrics/<run>/{per_sequence_logp.npy, steps.npy}

Writes (to results/plots/):
    overfit_curves.png     — HEADLINE: train vs val (own + full-mix) per model vs epoch.
                             Single-epoch -> val should keep falling (no U-shape).
    own_held_out_logp.png  — own held-out Σ log P trajectory per run vs epoch.
    logprob_overfit.png    — per-persona: own specialist vs mixture (+ other
                             specialists in gray) on this persona's test seqs vs epoch.
    hull_escapes.png       — hull-escape count + max gap vs mixture epoch.
    pi_kl.png / pi_residual.png — fitted π(persona) vs mixture epoch.
    residual_rms.png       — LoTP residual_rms (both fits) vs mixture epoch.
    param_norms.png        — ‖θ‖ and ‖θ-θ_base‖ for all 6 runs vs their own epoch.
    pi_kl_errorbars.png    — loss-aligned π_KL with bootstrap 95% CI (2,500-seq test set).
    complexity_analysis.png— persona complexity under the base model vs π_KL / mix-spec gap.
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
    min_val_loss_step,
    persona_to_indices,
    stack_log_p_for_alignment,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOTP_ROOT = RESULTS_DIR / "lotp"
METRICS_ROOT = RESULTS_DIR / "metrics"
CKPT_ROOT = RESULTS_DIR / "checkpoints"
PLOTS_ROOT = RESULTS_DIR / "plots"

PERSONA_COLORS = {
    "noir_detective":       "#1f77b4",
    "fairy_tale":           "#ff7f0e",
    "scientific_explainer": "#2ca02c",
    "absurdist":            "#d62728",
    "epistolary":           "#9467bd",
}
MIXTURE_COLOR = "#222222"


# ---------- loaders ----------

def _load_step_aligned() -> list[dict]:
    with (LOTP_ROOT / "step_aligned.jsonl").open() as f:
        return [json.loads(line) for line in f]


def _load_loss_aligned() -> dict:
    return json.loads((LOTP_ROOT / "loss_aligned.json").read_text())


def _load_checkpoint_rows(run: str) -> list[dict]:
    with (METRICS_ROOT / run / "checkpoints.jsonl").open() as f:
        return [json.loads(line) for line in f]


def _load_train_log(run: str) -> list[dict]:
    with (CKPT_ROOT / run / "train_log.jsonl").open() as f:
        return [json.loads(line) for line in f]


def _run_config(run: str) -> dict:
    return json.loads((CKPT_ROOT / run / "run_config.json").read_text())


def _steps_per_epoch(run: str) -> int:
    return int(_run_config(run)["steps_per_epoch"])


def step_to_epoch(run: str, steps) -> np.ndarray:
    """Convert raw training steps to fractional epoch for `run`."""
    return np.asarray(steps, dtype=float) / _steps_per_epoch(run)


def _save(fig, name: str) -> Path:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = PLOTS_ROOT / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _rolling(y: np.ndarray, k: int = 9) -> np.ndarray:
    """Simple centered moving average to de-noise the per-step train loss."""
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


# ---------- HEADLINE: train vs val overfitting curves ----------

def plot_overfit_curves() -> Path:
    """The core diagnostic: per model, train loss (per step) vs validation loss
    (own single-persona val and full-mix val) on the fractional-epoch axis.

    Single-epoch hypothesis: val keeps falling to the end (no overfitting U-shape),
    in contrast to the prior multi-epoch run where specialist val turned up early.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for ax, run in zip(axes_flat, RUNS):
        log = _load_train_log(run)
        spe = _steps_per_epoch(run)
        color = MIXTURE_COLOR if run == "mixture" else PERSONA_COLORS[run]

        train = [(e["step"], e["loss"]) for e in log if e["event"] == "train"]
        tr_ep = np.array([s for s, _ in train]) / spe
        tr_loss = np.array([l for _, l in train])
        ax.plot(tr_ep, tr_loss, color="0.8", linewidth=0.7, zorder=1, label="train (per step)")
        ax.plot(tr_ep, _rolling(tr_loss), color=color, linewidth=1.3, alpha=0.7, zorder=2,
                label="train (smoothed)")

        ck = [e for e in log if e["event"] == "checkpoint"]
        ck_ep = np.array([e["step"] for e in ck]) / spe
        if "val_own" in ck[-1]:
            vo = np.array([e["val_own"] for e in ck])
            ax.plot(ck_ep, vo, color=color, linewidth=2.2, marker="o", ms=3,
                    zorder=4, label="val (own persona)")
            i = int(np.argmin(vo))
            ax.scatter([ck_ep[i]], [vo[i]], color=color, s=55, zorder=5,
                       edgecolor="black", linewidth=0.6)
            ax.annotate(f"min val @ epoch {ck_ep[i]:.2f}", (ck_ep[i], vo[i]),
                        textcoords="offset points", xytext=(6, 8), fontsize=7)
        vm = np.array([e["val_mix"] for e in ck])
        ax.plot(ck_ep, vm, color="black", linewidth=1.5, linestyle="--", marker="s", ms=3,
                zorder=3, label="val (full mix)")

        ax.set_title(run, fontsize=11)
        ax.set_xlabel("epoch (fraction of the single pass)")
        ax.set_ylabel("cross-entropy (nats/token)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes_flat[5].axis("off") if len(RUNS) < 6 else None
    fig.suptitle("Single-epoch train vs validation loss — val keeps falling = no overfitting",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    return _save(fig, "overfit_curves.png")


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
        epochs = step_to_epoch(run, steps)
        color = MIXTURE_COLOR if run == "mixture" else PERSONA_COLORS[run]
        lw = 2.2 if run == "mixture" else 1.5
        ax.plot(epochs, own_logp, label=run, color=color, linewidth=lw)
        i_best = int(np.argmax(own_logp))
        ax.scatter([epochs[i_best]], [own_logp[i_best]], color=color, s=30, zorder=5)

    ax.set_xlabel("epoch (fraction of the single pass)")
    ax.set_ylabel(r"$\sum_j \log P_\theta(x_j)$  (own held-out)")
    ax.set_title("Own held-out total log-P per checkpoint  (dot = max)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "own_held_out_logp.png")


# ---------- per-persona log-P (spec vs mix) ----------

def plot_logprob_overfit() -> Path:
    """For each persona, Σ log P over that persona's test seqs vs epoch, for the
    own specialist (heavy), the mixture (black dashed), and the other specialists
    (gray). Each line is on its own run's epoch axis, so specialist (~283 steps)
    and mixture (~1,400 steps) are compared at equal per-persona exposure."""
    idx_map = persona_to_indices()
    run_data = {run: load_run_metrics(run) for run in RUNS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for ax, persona in zip(axes_flat, PERSONAS):
        own_idx = idx_map[persona]
        shown_gray = False
        for other in PERSONAS:
            if other == persona:
                continue
            steps, mat, _ = run_data[other]
            ax.plot(step_to_epoch(other, steps), mat[:, own_idx].sum(axis=1),
                    color="lightgray", linewidth=0.9, alpha=0.8, zorder=1,
                    label="other specialists" if not shown_gray else None)
            shown_gray = True

        steps, mat, _ = run_data["mixture"]
        mix_logp = mat[:, own_idx].sum(axis=1)
        ep_mix = step_to_epoch("mixture", steps)
        ax.plot(ep_mix, mix_logp, color=MIXTURE_COLOR, linewidth=2.2, linestyle="--",
                label="mixture", zorder=2)
        i_mix = int(np.argmax(mix_logp))
        ax.scatter([ep_mix[i_mix]], [mix_logp[i_mix]], color=MIXTURE_COLOR, s=35, zorder=5)

        steps, mat, _ = run_data[persona]
        own_logp = mat[:, own_idx].sum(axis=1)
        ep_own = step_to_epoch(persona, steps)
        ax.plot(ep_own, own_logp, color=PERSONA_COLORS[persona], linewidth=2.4,
                label=f"{persona} (own)", zorder=3)
        i_own = int(np.argmax(own_logp))
        ax.scatter([ep_own[i_own]], [own_logp[i_own]], color=PERSONA_COLORS[persona],
                   s=45, zorder=5)

        ax.set_title(persona, fontsize=11)
        ax.set_xlabel("epoch (fraction of the single pass)")
        ax.set_ylabel(r"$\sum_j \log P_\theta(x_j)$  (own test seqs)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    axes_flat[5].axis("off")
    fig.suptitle("Specialist (own) vs mixture log-P on each persona's test seqs, vs epoch  (dots = max)",
                 y=1.00)
    fig.tight_layout()
    return _save(fig, "logprob_overfit.png")


# ---------- hull escapes ----------

def plot_hull_escapes(rows: list[dict], loss_aligned: dict) -> Path:
    epochs = step_to_epoch("mixture", [r["alignment"]["mixture_step"] for r in rows])
    n_esc = np.array([r["hull_escapes"]["n_escapes"] for r in rows])
    max_excess = np.array([r["hull_escapes"]["max_excess"] for r in rows])
    n_seqs = rows[0]["hull_escapes"]["n_seqs"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, n_esc, color="#d62728", linewidth=1.8, label="escapes")
    ax1.set_xlabel("mixture epoch (fraction of the single pass)")
    ax1.set_ylabel(f"# escapes: mixture beats all specialists (out of {n_seqs})", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, max_excess, color="#1f77b4", linewidth=1.4, linestyle="--", label="max excess")
    ax2.set_ylabel("max excess = log P_mix - max_i log P_i  (nats)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    la_ep = step_to_epoch("mixture", loss_aligned["alignment"]["mixture_step"])
    ax1.axvline(float(la_ep), ls=":", color="black", linewidth=0.8, alpha=0.6)
    ax1.set_title("Convex-hull escapes (τ=0): mixture exceeds the best specialist")
    return _save(fig, "hull_escapes.png")


# ---------- π trajectory ----------

def plot_pi_trajectory(rows: list[dict], which_fit: str, loss_aligned: dict, fname: str) -> Path:
    epochs = step_to_epoch("mixture", [r["alignment"]["mixture_step"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    for persona in PERSONAS:
        ys = np.array([r[which_fit]["pi"][persona] for r in rows])
        ax.plot(epochs, ys, label=persona, color=PERSONA_COLORS[persona], linewidth=1.8)
    ax.axhline(0.2, ls="--", color="gray", linewidth=0.8, label="uniform 1/5")

    la_ep = float(step_to_epoch("mixture", loss_aligned["alignment"]["mixture_step"]))
    ax.axvline(la_ep, ls=":", color="black", linewidth=0.8, alpha=0.6)
    ax.text(la_ep, 1.02, f"loss-aligned epoch={la_ep:.2f}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("mixture epoch (fraction of the single pass)")
    ax.set_ylabel(r"$\pi_i$")
    ax.set_ylim(-0.02, 1.02)
    title = {"kl_fit": "KL fit  (primary, KL(empirical || M))",
             "residual_fit": "Residual fit  (diagnostic, spec L(π))"}[which_fit]
    ax.set_title(f"Fitted π trajectory — {title}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, fname)


# ---------- residual_rms ----------

def plot_residual_rms(rows: list[dict], loss_aligned: dict) -> Path:
    epochs = step_to_epoch("mixture", [r["alignment"]["mixture_step"] for r in rows])
    res_residual = np.array([r["residual_fit"]["residual_rms"] for r in rows])
    res_kl = np.array([r["kl_fit"]["residual_rms"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, res_kl, label="KL fit  (primary)", color="#1f77b4")
    ax.plot(epochs, res_residual, label="residual fit  (diagnostic)", color="#d62728", linestyle="--")

    i_best = int(np.argmin(res_kl))
    ax.scatter([epochs[i_best]], [res_kl[i_best]], color="#1f77b4", s=40, zorder=5)
    ax.annotate(f"min={res_kl[i_best]:.1f}\n@ epoch {epochs[i_best]:.2f}",
                (epochs[i_best], res_kl[i_best]),
                textcoords="offset points", xytext=(10, 10), fontsize=8)

    la_ep = float(step_to_epoch("mixture", loss_aligned["alignment"]["mixture_step"]))
    ax.axvline(la_ep, ls=":", color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("mixture epoch (fraction of the single pass)")
    ax.set_ylabel("residual RMS (nats)")
    ax.set_title("LoTP residual — how well the identity holds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "residual_rms.png")


# ---------- parameter norms ----------

def plot_param_norms() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in RUNS:
        rows = _load_checkpoint_rows(run)
        epochs = step_to_epoch(run, [r["step"] for r in rows])
        pn = np.array([r["param_norm"] for r in rows])
        dn = np.array([r["dev_norm"] for r in rows])
        color = MIXTURE_COLOR if run == "mixture" else PERSONA_COLORS[run]
        lw = 2.2 if run == "mixture" else 1.5
        axes[0].plot(epochs, pn, label=run, color=color, linewidth=lw)
        axes[1].plot(epochs, dn, label=run, color=color, linewidth=lw)
    for ax in axes:
        ax.set_xlabel("epoch (fraction of the single pass)")
        ax.grid(alpha=0.3)
    axes[0].set_title(r"$\|\theta\|_2$ per checkpoint")
    axes[0].set_ylabel(r"$\|\theta\|_2$")
    axes[1].set_title(r"$\|\theta - \theta_{\mathrm{base}}\|_2$ per checkpoint")
    axes[1].set_ylabel(r"$\|\theta - \theta_{\mathrm{base}}\|_2$")
    axes[1].legend(fontsize=8, loc="upper left")
    return _save(fig, "param_norms.png")


# ---------- bootstrap error bars on π_KL ----------

def plot_pi_kl_errorbars(n_bootstrap: int = 500, seed: int = 0) -> Path:
    """Bootstrap-resample the test set, refit π_KL on each resample, plot the
    loss-aligned (min-val) π_KL with 95% CI error bars."""
    run_data = {run: load_run_metrics(run) for run in RUNS}
    loss_pair = {
        "mixture_step": min_val_loss_step("mixture"),
        "specialist_steps": {p: min_val_loss_step(p) for p in PERSONAS},
    }
    spec, mix = stack_log_p_for_alignment(run_data, loss_pair)
    n_seqs = mix.size

    pi_ref, _, _ = fit_pi_kl(spec, mix)
    rng = np.random.default_rng(seed)
    boot_pi = np.zeros((n_bootstrap, len(PERSONAS)), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seqs, size=n_seqs)
        pi_b, _, _ = fit_pi_kl(spec[:, idx], mix[idx])
        boot_pi[b] = pi_b

    p_low = np.percentile(boot_pi, 2.5, axis=0)
    p_high = np.percentile(boot_pi, 97.5, axis=0)
    err_low = pi_ref - p_low
    err_high = p_high - pi_ref

    (LOTP_ROOT / "pi_kl_bootstrap.json").write_text(json.dumps({
        "alignment": loss_pair, "n_bootstrap": n_bootstrap, "bootstrap_seed": seed,
        "n_seqs": int(n_seqs), "personas": PERSONAS,
        "pi_full_data": pi_ref.tolist(),
        "bootstrap_mean": boot_pi.mean(axis=0).tolist(),
        "bootstrap_std": boot_pi.std(axis=0).tolist(),
        "ci_low_2.5": p_low.tolist(), "ci_high_97.5": p_high.tolist(),
    }, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERSONAS))
    ax.bar(x, pi_ref, color=[PERSONA_COLORS[p] for p in PERSONAS],
           alpha=0.75, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, pi_ref, yerr=[err_low, err_high], fmt="none", ecolor="black",
                capsize=5, linewidth=1.4, zorder=5)
    ax.axhline(0.2, ls="--", color="gray", linewidth=1.0, label="uniform 1/5")
    for i, p in enumerate(PERSONAS):
        std = float(boot_pi[:, i].std())
        sig = "" if p_low[i] <= 0.2 <= p_high[i] else " *"
        ax.text(i, pi_ref[i] + err_high[i] + 0.002, f"{pi_ref[i]:.4f}\n±{std:.4f}{sig}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(PERSONAS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\pi_i$")
    ax.set_ylim(0, max(p_high.max() + 0.03, 0.25))
    ax.set_title(f"KL-fit π with bootstrap 95% CI  ({n_bootstrap} resamples of {n_seqs} seqs)\n"
                 r"$*$ = 95% CI excludes uniform $1/5$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "pi_kl_errorbars.png")


# ---------- persona complexity analysis ----------

def plot_complexity_analysis() -> Path:
    """Persona complexity under the base model vs the recovered prior and the
    mix-vs-spec gap (at the loss-aligned / min-val checkpoint)."""
    from data import load_test_set, tokenize_story
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-5M")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rows = load_test_set()
    lens = np.array([len(tokenize_story(r["story"], tok)) for r in rows])
    n_scored = np.maximum(lens - 1, 1)
    idx_map = persona_to_indices()

    _, mat_any, _ = load_run_metrics(PERSONAS[0])
    base_per_seq = mat_any[0]   # step 0 identical across runs

    la = _load_loss_aligned()
    pi_kl_dict = la["kl_fit"]["pi"]
    spec_logp_per_persona = {}
    for p in PERSONAS:
        steps_p, mat_p, _ = load_run_metrics(p)
        s = la["alignment"]["specialist_steps"][p]
        spec_logp_per_persona[p] = mat_p[int(np.where(steps_p == s)[0][0])]
    steps_m, mat_m, _ = load_run_metrics("mixture")
    mix_logp = mat_m[int(np.where(steps_m == la["alignment"]["mixture_step"])[0][0])]

    dev_norm_peak = {}
    for p in PERSONAS:
        crows = _load_checkpoint_rows(p)
        steps_p, mat_p, _ = load_run_metrics(p)
        own = mat_p[:, idx_map[p]].sum(axis=1)
        dev_norm_peak[p] = float(crows[int(np.argmax(own))]["dev_norm"])

    base_per_tok = {p: float((base_per_seq[idx_map[p]] / n_scored[idx_map[p]]).mean()) for p in PERSONAS}
    mix_per_seq = {p: float(mix_logp[idx_map[p]].mean()) for p in PERSONAS}
    spec_per_seq = {p: float(spec_logp_per_persona[p][idx_map[p]].mean()) for p in PERSONAS}
    gap = {p: mix_per_seq[p] - spec_per_seq[p] for p in PERSONAS}
    order = sorted(PERSONAS, key=lambda p: -base_per_tok[p])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    xs = np.arange(len(order))

    ax = axes[0, 0]
    vals = [base_per_tok[p] for p in order]
    ax.bar(xs, vals, color=[PERSONA_COLORS[p] for p in order], alpha=0.8, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v - 0.1, f"{v:.2f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("base log P / token  (nats)")
    ax.set_title("A.  Persona complexity under the base model  (easier → harder)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    vals = [dev_norm_peak[p] for p in order]
    ax.bar(xs, vals, color=[PERSONA_COLORS[p] for p in order], alpha=0.8, edgecolor="black", linewidth=0.6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$\|\theta_{\mathrm{spec}}^{*} - \theta_{\mathrm{base}}\|_2$")
    ax.set_title("B.  Specialist parameter move at peak")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    xv = [base_per_tok[p] for p in PERSONAS]
    yv = [pi_kl_dict[p] for p in PERSONAS]
    for p, x, y in zip(PERSONAS, xv, yv):
        ax.scatter(x, y, s=110, color=PERSONA_COLORS[p], edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(0.2, ls="--", color="gray", linewidth=0.9, label="uniform 1/5")
    r = float(np.corrcoef(xv, yv)[0, 1])
    ax.set_xlabel("base log P / token  (nats)")
    ax.set_ylabel(r"$\pi_{\mathrm{KL}}$  (loss-aligned)")
    ax.set_title(f"C.  Does the recovered prior track complexity?  (r = {r:+.2f}, n=5)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    yv = [gap[p] for p in PERSONAS]
    for p, x, y in zip(PERSONAS, xv, yv):
        ax.scatter(x, y, s=110, color=PERSONA_COLORS[p], edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(0, ls="--", color="gray", linewidth=0.9, label="tie")
    ax.set_xlabel("base log P / token  (nats)")
    ax.set_ylabel(r"$\log P_{\mathrm{mix}} - \log P_{\mathrm{spec}}$  per seq  (mix wins if > 0)")
    ax.set_title("D.  Mix-vs-spec gap at loss-aligned")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "complexity_analysis.png")


# ---------- recovered π vs data proportions (uniform vs non-uniform mixture) ----------

def plot_pi_vs_proportions() -> Path:
    """Headline for the non-uniform experiment: for each mixture, the recovered
    loss-aligned π_KL next to the true DATA proportions. If π tracks the data
    proportions, recovery is faithful; systematic deviation is the base-model
    prior signature."""
    from data import mixture_exp_weights

    configs = [("loss_aligned.json", "uniform mixture", {p: 1.0 / len(PERSONAS) for p in PERSONAS})]
    if (LOTP_ROOT / "loss_aligned_exp.json").exists():
        configs.append(("loss_aligned_exp.json", "non-uniform mixture (geom 1.5)", mixture_exp_weights()))

    fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5), squeeze=False)
    for ax, (fname, title, props) in zip(axes[0], configs):
        pi = json.loads((LOTP_ROOT / fname).read_text())["kl_fit"]["pi"]
        x = np.arange(len(PERSONAS)); w = 0.38
        data_p = np.array([props[p] for p in PERSONAS])
        rec_p = np.array([pi[p] for p in PERSONAS])
        ax.bar(x - w / 2, data_p, w, label="data proportion", color="0.65",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, rec_p, w, label="recovered π_KL",
               color=[PERSONA_COLORS[p] for p in PERSONAS], edgecolor="black", linewidth=0.5)
        for i in range(len(PERSONAS)):
            ax.text(i + w / 2, rec_p[i] + 0.005, f"{rec_p[i] - data_p[i]:+.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(PERSONAS, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("proportion")
        ax.set_title(f"{title}\nrecovered π vs data proportion (Δ above bars)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, "pi_vs_proportions.png")


# ---------- main ----------

def main() -> None:
    rows = _load_step_aligned()
    la = _load_loss_aligned()
    outs = [
        plot_pi_vs_proportions(),
        plot_overfit_curves(),
        plot_own_held_out_logp(),
        plot_logprob_overfit(),
        plot_hull_escapes(rows, la),
        plot_pi_trajectory(rows, "kl_fit", la, "pi_kl.png"),
        plot_pi_trajectory(rows, "residual_fit", la, "pi_residual.png"),
        plot_residual_rms(rows, la),
        plot_param_norms(),
        plot_pi_kl_errorbars(),
        plot_complexity_analysis(),
    ]
    for p in outs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
