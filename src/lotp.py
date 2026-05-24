"""Fit π on the simplex for the LoTP identity:

    log P_mixture(x_j) ≈ logsumexp_i ( log P_persona_i(x_j) + log π_i )

Two fits are reported at every alignment point:

  * `fit_pi_kl` — PRIMARY. Minimizes the forward KL between the empirical
    inference set (treated as samples from P_mix) and the implied mixture
    `M(x) = Σ_i π_i P_i(x)`:

        KL(empirical || M) = const - mean_j logsumexp_i(log P_i(x_j) + log π_i)

    So minimizing KL is equivalent (up to a constant) to MAXIMIZING the average
    log-likelihood of the inference set under the implied mixture — this is the
    same optimum that mixture-model EM finds, but better motivated for our
    problem. Insensitive to per-sequence log-P offsets (the "magnitude issue":
    log-P magnitudes scale with sequence length, ~hundreds of nats per seq,
    so any squared-residual loss is dominated by per-sequence noise variance,
    not by the actual signal about π).

  * `fit_pi_residual` — DIAGNOSTIC. Minimizes the spec's
    `L(π) = mean_j (log P_mix - logsumexp(log P_i + log π_i))^2`.
    Kept for transparency; flagged as noise-sensitive in practice — see
    notes.md §"LoTP fit".

Outputs:
    results/lotp/loss_aligned.json
    results/lotp/step_aligned.jsonl

Alignment modes (task_plan.md §"Alignment Strategy"):
    step    every mixture checkpoint paired with the closest specialist step
    loss    each model at its own minimum-eval-loss checkpoint
            (own held-out = persona's 150 seqs for specialists, full 750 for mixture)

Convex-hull-escape count at margin τ:
    #{ j : max_i log P_persona_i(x_j) > log P_mixture(x_j) + τ }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from data import PERSONAS, load_inference_set

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS_ROOT = RESULTS_DIR / "metrics"
LOTP_ROOT = RESULTS_DIR / "lotp"

RUNS = PERSONAS + ["mixture"]   # specialists first, mixture last


# ---------- inference-set indexing ----------

def persona_to_indices() -> dict[str, np.ndarray]:
    """Map persona name -> the sequence indices in the 750-seq inference matrix
    whose source persona equals that name. This is the canonical own-held-out
    slice for each specialist."""
    rows = load_inference_set()
    out: dict[str, list[int]] = {p: [] for p in PERSONAS}
    for j, r in enumerate(rows):
        out[r["persona"]].append(j)
    return {p: np.asarray(idx, dtype=np.int64) for p, idx in out.items()}


def own_indices_for_run(run: str, idx_map: dict[str, np.ndarray]) -> np.ndarray:
    """Per task_plan.md: specialist's own held-out is its 150 seqs; mixture's is
    the full 750."""
    if run == "mixture":
        n = sum(len(v) for v in idx_map.values())
        return np.arange(n, dtype=np.int64)
    return idx_map[run]


# ---------- I/O ----------

def load_run_metrics(run: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (steps, per_seq_logp[checkpoints, seqs], checkpoint_rows)."""
    base = METRICS_ROOT / run
    steps = np.load(base / "steps.npy")
    matrix = np.load(base / "per_sequence_logp.npy")
    with (base / "checkpoints.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    if not (len(rows) == len(steps) == matrix.shape[0]):
        raise RuntimeError(f"{run}: metrics row counts disagree ({len(rows)}, {len(steps)}, {matrix.shape[0]})")
    return steps, matrix, rows


# ---------- π fits ----------

def fit_pi_residual(log_p_specialists: np.ndarray, log_p_mixture: np.ndarray,
                    init_pi: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """DIAGNOSTIC: minimize the spec's L(π) = mean_j (log P_mix - logsumexp_i(...))^2.

    Sensitive to per-sequence log-P magnitude (the "magnitude issue" — see
    module docstring). Kept for transparency, but prefer `fit_pi_kl` as primary.
    """
    n_personas = log_p_specialists.shape[0]
    if init_pi is None:
        x0 = np.zeros(n_personas, dtype=np.float64)   # softmax(0) = uniform
    else:
        pi0 = np.maximum(np.asarray(init_pi, dtype=np.float64), 1e-12)
        pi0 /= pi0.sum()
        x0 = np.log(pi0)

    def loss(logits: np.ndarray) -> float:
        m = logits.max()
        log_pi = logits - (m + np.log(np.exp(logits - m).sum()))
        joint = log_p_specialists + log_pi[:, None]    # (P, J)
        pred = logsumexp(joint, axis=0)                # (J,)
        return float(((log_p_mixture - pred) ** 2).mean())

    res = minimize(loss, x0, method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000})
    logits = res.x
    log_pi = logits - logsumexp(logits)
    pi = np.exp(log_pi)
    pi /= pi.sum()
    pred = logsumexp(log_p_specialists + log_pi[:, None], axis=0)
    avg_loglik = float(pred.mean())
    return pi, float(np.sqrt(res.fun)), avg_loglik


def fit_pi_kl(log_p_specialists: np.ndarray, log_p_mixture: np.ndarray,
              init_pi: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Minimize KL(empirical || M) ≡ maximize the avg log-likelihood of the
    inference set under the implied mixture M = Σ_i π_i P_i.

    The objective is

        loss(π) = -mean_j logsumexp_i(log P_i(x_j) + log π_i)

    `log_p_mixture` is NOT used for fitting (KL of P_mix with M does not depend
    on the constant log P_mix part once we hold the samples fixed). It IS used
    to compute the reported `residual_rms` for cross-fit comparability.
    """
    n_personas = log_p_specialists.shape[0]
    if init_pi is None:
        x0 = np.zeros(n_personas, dtype=np.float64)
    else:
        pi0 = np.maximum(np.asarray(init_pi, dtype=np.float64), 1e-12)
        pi0 /= pi0.sum()
        x0 = np.log(pi0)

    def loss(logits: np.ndarray) -> float:
        m = logits.max()
        log_pi = logits - (m + np.log(np.exp(logits - m).sum()))
        joint = log_p_specialists + log_pi[:, None]
        return -float(logsumexp(joint, axis=0).mean())

    res = minimize(loss, x0, method="L-BFGS-B",
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000})
    logits = res.x
    log_pi = logits - logsumexp(logits)
    pi = np.exp(log_pi)
    pi /= pi.sum()

    pred = logsumexp(log_p_specialists + log_pi[:, None], axis=0)
    residual_rms = float(np.sqrt(np.mean((log_p_mixture - pred) ** 2)))
    avg_loglik = float(pred.mean())   # mean_j log M(x_j) — KL up to const
    return pi, residual_rms, avg_loglik


def per_persona_residual(log_p_specialists: np.ndarray, log_p_mixture: np.ndarray,
                          pi: np.ndarray, idx_map: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-persona breakdown of the squared residual under fitted π, to detect
    length-domination by any single persona."""
    log_pi = np.log(np.maximum(pi, 1e-300))
    pred = logsumexp(log_p_specialists + log_pi[:, None], axis=0)
    sq = (log_p_mixture - pred) ** 2
    return {p: float(sq[idx_map[p]].mean()) for p in PERSONAS}


# ---------- hull escape ----------

def hull_escape_count(log_p_specialists: np.ndarray, log_p_mixture: np.ndarray,
                      tau: float = 0.0) -> dict:
    """Sequences where the LoTP inequality is broken at margin τ.

    Inequality P_mix(x) ≤ max_i P_persona_i(x). In log-space:
    log P_mix ≤ max_i log P_persona_i. Escape iff max_i log P_persona_i >
    log P_mix + τ.
    """
    max_specialist = log_p_specialists.max(axis=0)
    gap = max_specialist - log_p_mixture
    escapes = gap > tau
    return {
        "tau": tau,
        "n_escapes": int(escapes.sum()),
        "n_seqs": int(log_p_mixture.size),
        "mean_gap": float(gap.mean()),
        "max_gap": float(gap.max()),
    }


# ---------- alignment ----------

def pair_steps_step_aligned(per_run_steps: dict[str, np.ndarray]) -> list[dict]:
    """For each mixture checkpoint, pair each specialist with its closest
    available step. Edge case: for late mixture checkpoints (past the
    specialist's last step), all specialists pin to their final checkpoint."""
    mixture_steps = per_run_steps["mixture"]
    pairs: list[dict] = []
    for ms in mixture_steps.tolist():
        chosen = {"mixture_step": int(ms), "specialist_steps": {}, "specialist_gap": {}}
        for persona in PERSONAS:
            ss = per_run_steps[persona]
            idx = int(np.argmin(np.abs(ss - ms)))
            chosen["specialist_steps"][persona] = int(ss[idx])
            chosen["specialist_gap"][persona] = int(ss[idx] - ms)  # signed gap
        pairs.append(chosen)
    return pairs


def stack_log_p_for_alignment(run_data: dict, mode_choice: dict) -> tuple[np.ndarray, np.ndarray]:
    """Given a per-run dict {run: (steps, matrix, rows)} and an alignment
    {'mixture_step': int, 'specialist_steps': {persona: step}}, return
    (log_p_specialists[5, n_seqs], log_p_mixture[n_seqs]) at those checkpoints."""
    n_seqs = run_data["mixture"][1].shape[1]
    spec = np.zeros((len(PERSONAS), n_seqs), dtype=np.float64)
    for i, persona in enumerate(PERSONAS):
        steps, matrix, _ = run_data[persona]
        step = mode_choice["specialist_steps"][persona]
        idx = int(np.where(steps == step)[0][0])
        spec[i] = matrix[idx]
    m_steps, m_matrix, _ = run_data["mixture"]
    m_idx = int(np.where(m_steps == mode_choice["mixture_step"])[0][0])
    mix = m_matrix[m_idx]
    return spec, mix


def min_loss_step_own_held_out(run: str, steps: np.ndarray, matrix: np.ndarray,
                                idx_map: dict[str, np.ndarray]) -> int:
    """Pick the checkpoint with the highest Σ log P over the run's *own*
    held-out: specialist -> its 150 seqs; mixture -> all 750.

    Fixes the prior bug of using total_logp over the full 750 for everyone."""
    own = own_indices_for_run(run, idx_map)
    own_logp = matrix[:, own].sum(axis=1)                # (n_checkpoints,)
    best = int(np.argmax(own_logp))
    return int(steps[best])


# ---------- pipeline ----------

def _fit_both(spec: np.ndarray, mix: np.ndarray, idx_map: dict[str, np.ndarray]) -> dict:
    pi_kl, rms_kl, ll_kl = fit_pi_kl(spec, mix)
    pi_res, rms_res, ll_res = fit_pi_residual(spec, mix)
    return {
        "kl_fit": {
            "pi": dict(zip(PERSONAS, pi_kl.tolist())),
            "residual_rms": rms_kl,
            "avg_loglik": ll_kl,
            "per_persona_residual": per_persona_residual(spec, mix, pi_kl, idx_map),
        },
        "residual_fit": {
            "pi": dict(zip(PERSONAS, pi_res.tolist())),
            "residual_rms": rms_res,
            "avg_loglik": ll_res,
            "per_persona_residual": per_persona_residual(spec, mix, pi_res, idx_map),
        },
        "hull_escapes": hull_escape_count(spec, mix, tau=0.0),
    }


def run_all_modes() -> None:
    LOTP_ROOT.mkdir(parents=True, exist_ok=True)
    idx_map = persona_to_indices()
    run_data = {run: load_run_metrics(run) for run in RUNS}
    per_run_steps = {run: run_data[run][0] for run in RUNS}

    # --- step-aligned ---
    pairs = pair_steps_step_aligned(per_run_steps)
    step_results = []
    for pair in pairs:
        spec, mix = stack_log_p_for_alignment(run_data, pair)
        result = {"alignment": pair, **_fit_both(spec, mix, idx_map)}
        step_results.append(result)
    (LOTP_ROOT / "step_aligned.jsonl").write_text(
        "\n".join(json.dumps(r) for r in step_results) + "\n"
    )

    # --- loss-aligned (per-model own held-out) ---
    loss_pair = {
        "mixture_step": min_loss_step_own_held_out(
            "mixture", run_data["mixture"][0], run_data["mixture"][1], idx_map),
        "specialist_steps": {
            p: min_loss_step_own_held_out(p, run_data[p][0], run_data[p][1], idx_map)
            for p in PERSONAS
        },
    }
    spec, mix = stack_log_p_for_alignment(run_data, loss_pair)
    loss_result = {"alignment": loss_pair, **_fit_both(spec, mix, idx_map)}
    (LOTP_ROOT / "loss_aligned.json").write_text(json.dumps(loss_result, indent=2))

    print("step-aligned: wrote", LOTP_ROOT / "step_aligned.jsonl")
    print("\nloss-aligned alignment:", loss_pair)
    print("\n  KL fit  (primary; minimize KL(empirical || M)):")
    print("    π =", loss_result["kl_fit"]["pi"])
    print(f"    avg log M(x) = {loss_result['kl_fit']['avg_loglik']:.3f}  "
          f"residual_rms = {loss_result['kl_fit']['residual_rms']:.3f}")
    print("\n  residual fit  (diagnostic; minimize spec L(π)):")
    print("    π =", loss_result["residual_fit"]["pi"])
    print(f"    avg log M(x) = {loss_result['residual_fit']['avg_loglik']:.3f}  "
          f"residual_rms = {loss_result['residual_fit']['residual_rms']:.3f}")
    print("\n  hull escapes:", loss_result["hull_escapes"])


if __name__ == "__main__":
    run_all_modes()
