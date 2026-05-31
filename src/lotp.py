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

from data import PERSONAS, load_test_set

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS_ROOT = RESULTS_DIR / "metrics"
LOTP_ROOT = RESULTS_DIR / "lotp"
CHECKPOINT_ROOT = RESULTS_DIR / "checkpoints"

RUNS = PERSONAS + ["mixture"]   # specialists first, mixture last


# ---------- inference-set indexing ----------

def persona_to_indices() -> dict[str, np.ndarray]:
    """Map persona name -> the sequence indices in the ~2,500-seq test matrix
    whose source persona equals that name. This is the canonical own-held-out
    slice for each specialist."""
    rows = load_test_set()
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


def fit_pi_linear(log_p_specialists: np.ndarray, log_p_mixture: np.ndarray,
                  init_pi: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Solve the LoTP linear system P_mix(x) = Σ_i π_i P_i(x) DIRECTLY in probability
    space (not log/KL), on the simplex. Each sequence's equation is rescaled by its
    own max (a per-row scaling leaves the linear solution unchanged) so the tiny
    probabilities don't underflow, then least-squares over a softmax-parameterized π.

    Unlike `fit_pi_kl` this genuinely USES P_mix (each row becomes π_persona ≈
    exp(log P_mix − log P_max-specialist)), so it is sensitive to the mixture and
    recovers the prior's RANKING — though compressed toward uniform because the
    exp(gap) magnitudes are tiny. The objective is convex in π (a non-negative
    least-squares on the simplex), so the optimum is unique. Returns
    (pi, prob_residual_rms)."""
    n = log_p_specialists.shape[0]
    m = np.maximum(log_p_specialists.max(axis=0), log_p_mixture)
    A = np.exp(log_p_specialists - m)      # (P, J), in (0, 1]
    b = np.exp(log_p_mixture - m)          # (J,)
    if init_pi is None:
        x0 = np.zeros(n, dtype=np.float64)
    else:
        pi0 = np.maximum(np.asarray(init_pi, dtype=np.float64), 1e-12)
        x0 = np.log(pi0 / pi0.sum())

    def loss(logits: np.ndarray) -> float:
        pi = np.exp(logits - logsumexp(logits))
        return float(np.mean((pi @ A - b) ** 2))

    res = minimize(loss, x0, method="L-BFGS-B",
                   options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 5000})
    pi = np.exp(res.x - logsumexp(res.x))
    pi /= pi.sum()
    return pi, float(np.sqrt(res.fun))


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
    """Convex-hull ESCAPE = the mixture beats the BEST specialist, i.e. the LoTP
    inequality P_mix(x) ≤ max_i P_persona_i(x) is VIOLATED.

    In log-space: escape iff log P_mix > max_i log P_persona_i + τ. We report
    `excess = log P_mix - max_i log P_specialist_i` (>0 on an escape). Escapes are
    expected to rise with overfitting (the design doc) — the mixture rescues
    specialists that have overfit their own data.

    NOTE: earlier code had this sign INVERTED (it counted `max_i P_i > P_mix`, i.e.
    the inequality HOLDING, and called that an escape). Fixed here to match the
    design-doc definition; downstream n_escapes therefore changed meaning.
    """
    max_specialist = log_p_specialists.max(axis=0)
    excess = log_p_mixture - max_specialist   # >0: mixture beats every specialist
    escapes = excess > tau
    return {
        "tau": tau,
        "n_escapes": int(escapes.sum()),
        "n_seqs": int(log_p_mixture.size),
        "mean_excess": float(excess.mean()),
        "max_excess": float(excess.max()),
    }


# ---------- alignment ----------

def pair_steps_step_aligned(per_run_steps: dict[str, np.ndarray],
                            mixture_run: str = "mixture") -> list[dict]:
    """For each mixture checkpoint, pair each specialist with its closest
    available step. Edge case: for late mixture checkpoints (past the
    specialist's last step), all specialists pin to their final checkpoint."""
    mixture_steps = per_run_steps[mixture_run]
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


def stack_log_p_for_alignment(run_data: dict, mode_choice: dict,
                              mixture_run: str = "mixture") -> tuple[np.ndarray, np.ndarray]:
    """Given a per-run dict {run: (steps, matrix, rows)} and an alignment
    {'mixture_step': int, 'specialist_steps': {persona: step}}, return
    (log_p_specialists[5, n_seqs], log_p_mixture[n_seqs]) at those checkpoints."""
    n_seqs = run_data[mixture_run][1].shape[1]
    spec = np.zeros((len(PERSONAS), n_seqs), dtype=np.float64)
    for i, persona in enumerate(PERSONAS):
        steps, matrix, _ = run_data[persona]
        step = mode_choice["specialist_steps"][persona]
        idx = int(np.where(steps == step)[0][0])
        spec[i] = matrix[idx]
    m_steps, m_matrix, _ = run_data[mixture_run]
    m_idx = int(np.where(m_steps == mode_choice["mixture_step"])[0][0])
    mix = m_matrix[m_idx]
    return spec, mix


def min_loss_step_own_held_out(run: str, steps: np.ndarray, matrix: np.ndarray,
                                idx_map: dict[str, np.ndarray]) -> int:
    """DIAGNOSTIC: checkpoint with the highest Σ log P over the run's *own* test
    slice (specialist -> its persona's test seqs; mixture -> all). Selects on the
    LoTP test set itself — kept only as a diagnostic; primary selection is
    `min_val_loss_step`, which uses the held-out val split instead."""
    own = own_indices_for_run(run, idx_map)
    own_logp = matrix[:, own].sum(axis=1)                # (n_checkpoints,)
    best = int(np.argmax(own_logp))
    return int(steps[best])


def min_val_loss_step(run: str) -> int:
    """PRIMARY loss-aligned selection: the checkpoint with the lowest validation
    loss, read from the run's train_log.jsonl. Specialists select on their own
    single-persona val (`val_own`); the mixture on the full-mix val (`val_mix`).
    Using val (not the test set) keeps model selection out of the LoTP fit."""
    log_path = CHECKPOINT_ROOT / run / "train_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}; run train.py for '{run}' first.")
    key = "val_mix" if run.startswith("mixture") else "val_own"
    best_step, best_loss = None, float("inf")
    with log_path.open() as f:
        for line in f:
            e = json.loads(line)
            if e.get("event") != "checkpoint":
                continue
            if key not in e:
                raise RuntimeError(f"{log_path}: checkpoint step {e.get('step')} missing '{key}'")
            if e[key] < best_loss:
                best_loss, best_step = e[key], int(e["step"])
    if best_step is None:
        raise RuntimeError(f"{log_path}: no checkpoint entries with '{key}'")
    return best_step


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


def run_for_mixture(mixture_run: str, suffix: str, idx_map: dict[str, np.ndarray]) -> dict:
    """Step- and loss-aligned LoTP fit for one mixture run against the 5 specialists.
    Writes step_aligned{suffix}.jsonl and loss_aligned{suffix}.json."""
    run_data = {r: load_run_metrics(r) for r in PERSONAS + [mixture_run]}
    per_run_steps = {r: run_data[r][0] for r in PERSONAS + [mixture_run]}

    # --- step-aligned ---
    pairs = pair_steps_step_aligned(per_run_steps, mixture_run)
    step_results = []
    for pair in pairs:
        spec, mix = stack_log_p_for_alignment(run_data, pair, mixture_run)
        step_results.append({"alignment": pair, **_fit_both(spec, mix, idx_map)})
    (LOTP_ROOT / f"step_aligned{suffix}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in step_results) + "\n"
    )

    # --- loss-aligned (each model at its own min-val-loss checkpoint) ---
    loss_pair = {
        "mixture_step": min_val_loss_step(mixture_run),
        "specialist_steps": {p: min_val_loss_step(p) for p in PERSONAS},
    }
    spec, mix = stack_log_p_for_alignment(run_data, loss_pair, mixture_run)
    loss_result = {"mixture_run": mixture_run, "alignment": loss_pair,
                   **_fit_both(spec, mix, idx_map)}
    (LOTP_ROOT / f"loss_aligned{suffix}.json").write_text(json.dumps(loss_result, indent=2))

    print(f"\n=== {mixture_run} (loss-aligned: {loss_pair['mixture_step']}) ===")
    print("  KL π   =", {k: round(v, 4) for k, v in loss_result["kl_fit"]["pi"].items()})
    print("  hull escapes (mixture beats ALL specialists):", loss_result["hull_escapes"])
    return loss_result


def run_all_modes() -> None:
    """Fit LoTP π for every available mixture run (uniform 'mixture' and, if it has
    been trained/evaluated, the non-uniform 'mixture_exp') against the specialists."""
    LOTP_ROOT.mkdir(parents=True, exist_ok=True)
    idx_map = persona_to_indices()

    # uniform 'mixture' plus any 'mixture_*' variant that has been evaluated.
    mixtures = [("mixture", "")]
    for d in sorted(METRICS_ROOT.glob("mixture_*")):
        mixtures.append((d.name, d.name.replace("mixture", "")))

    for mixture_run, suffix in mixtures:
        run_for_mixture(mixture_run, suffix, idx_map)


if __name__ == "__main__":
    run_all_modes()
