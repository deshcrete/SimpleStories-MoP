"""Measure the induced prior by SAMPLING from the mixture and classifying samples.

The LoTP π-fit recovers the persona composition of whatever samples it is fed. Fed
the eval set it returns the eval composition (uniform), blind to the mixture
(see analysis.md). Fed samples DRAWN FROM the mixture model, it returns the
mixture's own persona composition = P_mix(persona) = the induced prior.

For each trained mixture we:
  1. generate N stories, seeded with EOS (the SimpleStories model's natural
     unconditional start — it has no BOS and treats EOS as "begin a new story");
  2. score each generated story under the 5 specialists (min-val checkpoints);
  3. report the induced prior two ways — argmax_i log P_i classification, and the
     KL fit — and compare to the mixture's TRAINING proportions.

Deviation of the generated prior from the training proportions is the base-model
prior signature (the original research question).

Writes results/lotp/generated_prior{suffix}.json and
results/plots/generated_prior.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PERSONAS, StoryDataset, collate_for_clm, mixture_exp_weights
from eval_checkpoints import per_sequence_log_prob
from lotp import METRICS_ROOT, fit_pi_kl, min_val_loss_step
from model import load_base_model_and_tokenizer, load_checkpoint

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CKPT_ROOT = RESULTS_DIR / "checkpoints"
LOTP_ROOT = RESULTS_DIR / "lotp"
PLOTS_ROOT = RESULTS_DIR / "plots"
SEED = 42

PERSONA_COLORS = {
    "noir_detective": "#1f77b4", "fairy_tale": "#ff7f0e",
    "scientific_explainer": "#2ca02c", "absurdist": "#d62728", "epistolary": "#9467bd",
}


@torch.no_grad()
def generate_samples(mixture_run: str, tokenizer, n: int, device: str,
                     batch: int = 64, temperature: float = 1.0, top_p: float = 0.95,
                     max_new_tokens: int = 150) -> list[str]:
    """Generate n stories from the mixture's min-val checkpoint, seeded with EOS."""
    step = min_val_loss_step(mixture_run)
    model = load_checkpoint(CKPT_ROOT / mixture_run / f"step_{step}", device=device)
    model.eval()
    torch.manual_seed(SEED)
    stories: list[str] = []
    eos = tokenizer.eos_token_id
    while len(stories) < n:
        bs = min(batch, n - len(stories) + batch)  # slight over-generate; trimmed below
        seed_ids = torch.full((bs, 1), eos, dtype=torch.long, device=device)
        attn = torch.ones_like(seed_ids)
        gen = model.generate(seed_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature, top_p=top_p,
                             eos_token_id=eos, pad_token_id=eos)
        for row in gen.tolist():
            txt = tokenizer.decode(row[1:], skip_special_tokens=True).strip()  # drop seed EOS
            if len(txt) > 5:
                stories.append(txt)
    del model
    torch.cuda.empty_cache()
    return stories[:n]


def score_under_specialists(stories: list[str], tokenizer, device: str, batch: int = 64) -> np.ndarray:
    """Return log_p_specialists of shape (5, N): Σ log P_spec_i over each story."""
    rows = [{"story": s, "persona": "gen", "id": i} for i, s in enumerate(stories)]
    loader = DataLoader(
        StoryDataset(rows, tokenizer), batch_size=batch, shuffle=False,
        collate_fn=lambda b: collate_for_clm(b, tokenizer.pad_token_id),
    )
    mat = np.zeros((len(PERSONAS), len(rows)), dtype=np.float64)
    for i, p in enumerate(PERSONAS):
        step = min_val_loss_step(p)
        model = load_checkpoint(CKPT_ROOT / p / f"step_{step}", device=device)
        mat[i] = per_sequence_log_prob(model, loader, device)
        del model
        torch.cuda.empty_cache()
    return mat


def analyze(mat: np.ndarray) -> tuple[dict, dict, float]:
    """Induced prior via argmax classification and via the KL fit; plus their agreement."""
    labels = mat.argmax(axis=0)
    argmax_pi = {PERSONAS[i]: float((labels == i).mean()) for i in range(len(PERSONAS))}
    pi_kl, _, _ = fit_pi_kl(mat, np.zeros(mat.shape[1]))
    kl_pi = dict(zip(PERSONAS, pi_kl.tolist()))
    # agreement: fraction of samples whose argmax persona is the KL-fit's argmax... use
    # a simpler proxy — L1 distance between the two distributions.
    l1 = float(sum(abs(argmax_pi[p] - kl_pi[p]) for p in PERSONAS))
    return argmax_pi, kl_pi, l1


def _configs() -> list[tuple[str, str, dict]]:
    out = [("mixture", "", {p: 1.0 / len(PERSONAS) for p in PERSONAS})]
    if (METRICS_ROOT / "mixture_exp").exists():
        out.append(("mixture_exp", "_exp", mixture_exp_weights()))
    return out


def plot(all_res: list[tuple[str, dict, dict]]) -> None:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(all_res), figsize=(7 * len(all_res), 5), squeeze=False)
    for ax, (label, gen_pi, props) in zip(axes[0], all_res):
        x = np.arange(len(PERSONAS)); w = 0.38
        tp = np.array([props[p] for p in PERSONAS])
        gp = np.array([gen_pi[p] for p in PERSONAS])
        ax.bar(x - w / 2, tp, w, label="training proportion", color="0.65",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, gp, w, label="generated prior (sampled from mixture)",
               color=[PERSONA_COLORS[p] for p in PERSONAS], edgecolor="black", linewidth=0.5)
        for i in range(len(PERSONAS)):
            ax.text(i + w / 2, gp[i] + 0.005, f"{gp[i] - tp[i]:+.3f}", ha="center",
                    va="bottom", fontsize=7)
        r = np.corrcoef(tp, gp)[0, 1] if np.std(tp) > 1e-9 else None
        rstr = "" if r is None else f"  (r = {r:+.2f})"
        ax.set_xticks(x); ax.set_xticklabels(PERSONAS, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("proportion")
        ax.set_title(f"{label}\ninduced prior from sampling vs training proportion{rstr}\n"
                     "(Δ above bars = base-model prior signature)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = PLOTS_ROOT / "generated_prior.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="samples per mixture")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, tokenizer = load_base_model_and_tokenizer(device="cpu")

    all_res = []
    for run, suffix, props in _configs():
        print(f"\n=== {run}: generating {args.n} samples (temp {args.temperature}) ===")
        stories = generate_samples(run, tokenizer, args.n, device, temperature=args.temperature)
        mat = score_under_specialists(stories, tokenizer, device)
        argmax_pi, kl_pi, l1 = analyze(mat)

        # eyeball: one example per persona-mode
        labels = mat.argmax(axis=0)
        print(f"{'persona':22}{'train':>8}{'argmax':>9}{'KL':>8}{'Δ(KL-train)':>13}")
        for i, p in enumerate(PERSONAS):
            print(f"{p:22}{props[p]:8.3f}{argmax_pi[p]:9.3f}{kl_pi[p]:8.3f}{kl_pi[p]-props[p]:+13.3f}")
        print(f"  argmax/KL L1 distance = {l1:.4f}  (low = they agree)")
        r = np.corrcoef([props[p] for p in PERSONAS], [kl_pi[p] for p in PERSONAS])[0, 1] \
            if np.std([props[p] for p in PERSONAS]) > 1e-9 else float("nan")
        print(f"  corr(training proportion, generated prior) = {r:+.3f}")
        for i, p in enumerate(PERSONAS):
            ex = next((s for s, lab in zip(stories, labels) if lab == i), None)
            if ex:
                print(f"    [{p}] {ex[:110]}")

        out = {"mixture_run": run, "n": args.n, "training_proportion": props,
               "argmax_prior": argmax_pi, "kl_prior": kl_pi, "argmax_kl_l1": l1}
        (LOTP_ROOT / f"generated_prior{suffix}.json").write_text(json.dumps(out, indent=2))
        all_res.append((run, kl_pi, props))

    plot(all_res)


if __name__ == "__main__":
    main()
