# Inducing Mixture Distribution: Concrete Methodology

## Overview

This experiment induces a known prior on a base LLM by fine-tuning on a mixture of persona-labeled story data, then measures how that prior is recovered through a Law of Total Probability (LoTP) decomposition. The goal is to expose the *base model's underlying prior signature* — i.e., which personas the base model finds easier to represent — as deviation of the recovered mixture weights from the data-uniform 1/5.

The experiment also produces the empirical trajectories needed for the broader Bayesian-model-of-SGD program: per-checkpoint sequence likelihoods, parameter norms, and convex-hull-escape indicators across six co-trained models.

---

## Personas and Dataset

Five stylistically distinct personas, sampled using the SimpleStories procedure:

- **noir_detective** — first-person hardboiled detective narrator; short, clipped sentences; dark atmosphere.
- **fairy_tale** — classic "Once upon a time" opener; third-person omniscient; ends with an explicit moral.
- **scientific_explainer** — narrator pauses to explain how/why things happen with cause-and-effect framing.
- **absurdist** — non-sequiturs and broken narrative logic; no plot, no resolution.
- **epistolary** — story told entirely as letters between two characters.

Dataset: 2,150 stories per persona (HuggingFace: desh2806/simplestories-personas).

## Data Splits (apples-to-apples)

Per persona (2,150 stories):

- **500** persona-train split
- **500** mixture-contribution split
- **150** inference split (held out from all training)
- **~1,000** reserve / extra eval (untouched during main experiment)

Resulting training sets:

- **Mixture model:** trained on 5 × 500 = 2,500 stories (union of all mixture-contribution splits).
- **Five persona models:** each trained on 500 stories of their own persona.

Every model has seen exactly 500 stories of any given persona's distribution. The mixture has additionally seen 4 other personas; the specialists have not. Any difference between P_mixture and P_persona_i on persona_i's inference data is therefore attributable to *what else the mixture saw*, not to data quantity for that persona.

**Combined inference set:** 5 × 150 = 750 stories, labeled by source persona. All six models are evaluated on this set.

---

## Base Model and Fine-tuning Setup

- **Base model:** SimpleStories 5M parameters.
- **Optimizer:** AdamW.
- **Learning rate:** 3e-4 to 1e-3 (sweep if needed; SimpleStories tolerates higher LRs).
- **Batch size:** 32–64.
- **Weight decay:** 0 or very light (1e-4) — avoid conflating with the regularization story.
- **Epochs:** ~10.
- **Seed, data order, and hyperparameters held constant across all six runs.** Non-negotiable: any optimizer-trajectory difference between mixture and persona models otherwise becomes a confound for "the prior selected for this persona."

### Step counts and checkpointing

With batch size 32:

- Specialists: 500 examples → ~16 steps/epoch → ~160 steps for 10 epochs.
- Mixture: 2,500 examples → ~78 steps/epoch → ~780 steps for 10 epochs.

Checkpoint cadence:

- **Specialists:** every 8 steps for the first 80 steps, then every 16 steps. ~15 checkpoints/model.
- **Mixture:** every 20 steps for the first 200 steps, then every 50 steps. ~15 checkpoints.

At each checkpoint: full inference-set eval (cheap at 5M params), parameter norm, fitted π from LoTP, convex-hull-escape count.

---

## Defining P_persona_i(sequence)

Token-level autoregressive likelihood under model θ_i:

    log P_{θ_i}(x) = Σ_{t=1}^{T} log p_{θ_i}(x_t | x_{<t})

Implementation rules:

1. **Sum of log-probs, not mean.** The LoTP identity P_mixture(seq) ≈ Σ_i P_persona_i(seq) · π_i is a statement about probabilities, not per-token averages. Mean log-prob breaks the additive structure across sequences of different lengths. Keep Σ log p for LoTP; separately report mean log-prob / perplexity for human-readable plots.

2. **Length-match across personas.** Epistolary stories are likely longer than noir ones. Uneven length distributions cause aggregate log-likelihoods to be dominated by the longest persona. Either (a) truncate/pad to fixed T per sequence, or (b) report per-persona log-likelihoods separately and only aggregate when fitting the LoTP residual.

3. **Cross-evaluate on the full inference set.** Compute P_persona_i(seq_j) for *all* j, including sequences from other personas. Restricting to same-persona evaluation makes the LoTP linear system rank-deficient.

4. **Consistent BOS/conditioning handling.** Decide once whether log-prob starts at t=1 (with BOS) or t=2. Doesn't affect the LoTP fit if consistent, but matters for cross-model comparisons.

---

## Recovering the Prior: LoTP Fit

Given log P_{θ_i}(seq_j) and log P_mixture(seq_j) for all (i, j), solve:

    P_mixture(x_j) = Σ_i P_{θ_i}(x_j) · π_i

subject to π_i ≥ 0, Σ_i π_i = 1.

### Numerically stable fit

Work in log-space using log-sum-exp:

    log P_mixture(x_j) ≈ logsumexp_i ( log P_{θ_i}(x_j) + log π_i )

Minimize the squared residual

    L(π) = Σ_j ( log P_mixture(x_j) − logsumexp_i ( log P_{θ_i}(x_j) + log π_i ) )^2

on the probability simplex via projected gradient or EM. EM gives closed-form updates with fixed θ_i and θ_mixture and is the natural choice — also a useful pointer to the broader EM-as-framing-for-SGD question.

### Interpreting π

- If π ≈ (1/5, 1/5, 1/5, 1/5, 1/5): the mixture's posterior matches the data-uniform prior — no base-model prior signature.
- If π deviates from uniform: the base model finds some personas easier to represent, and the mixture training allocated more posterior mass to those than the data alone justifies. **Deviation of π from 1/5 is the induced prior signature.**

---

## Measurements Logged Per Checkpoint

For each of the 6 models, at each checkpoint:

1. **Total inference log-likelihood:** Σ_j log P_θ(x_j) over the full 750-sequence inference set.
2. **Per-persona cross-evaluation matrix:** Σ_{j ∈ persona_k} log P_θ(x_j) for each k. Produces a 6 × 5 matrix per checkpoint.
3. **Parameter norms:** ‖θ‖_2 and ‖θ − θ_base‖_2 (deviation from base).
4. **Eval losses:** persona-own held-out and mixture held-out.
5. **Fitted π_i** from the LoTP linear system, plus residual ‖log P_mixture − logsumexp(…)‖.
6. **Convex-hull-escape indicator:** count and magnitude of sequences j where log P_{θ_i}(x_j) > log P_mixture(x_j) by more than some margin τ. (As overfitting progresses, expected count rises.)

---

## Alignment Strategy for LoTP Comparisons

With 500 training examples, specialists hit minimum eval loss quickly and then overfit. The mixture, with 2,500 examples, overfits later. Minimum-eval-loss checkpoints therefore occur at different training steps across models.

Two alignment modes; **report both**:

- **Step-aligned:** all models compared at the same step count. Cleanest for studying SGD trajectory; mixture will be undertrained relative to specialists at any given step. **Use for trajectory plots and the Bayesian-evolution analysis.**
- **Loss-aligned:** each model at its own min-eval-loss checkpoint. Cleanest for studying the converged endpoint, at the cost of comparing models at different training points. **Use for the LoTP π-fit and the "does the identity hold" check.**

Report whether fitted π changes meaningfully between the two alignment modes — that itself is an informative diagnostic.

---

## Predicted LoTP Behavior

Two LoTP claims become cleanly testable under the apples-to-apples split:

1. **P_mixture(seq) ≤ max_i P_persona_i(seq)** — fair comparison at convergence. If the mixture beats the relevant specialist on persona_i's own sequences, that's a strong claim about cross-persona transfer / shared structure in the base model. If the specialist wins (expected at convergence), the gap measures how much the other personas diluted learning.
2. **P_mixture(seq) ≈ Σ_i P_persona_i(seq) · π_i** — the fitted π should ideally be ≈ uniform. Deviation from 1/5 is the base-model prior signature.

The formulation is sensitive to overfitting — escapes from the convex hull break the inequality. The convex-hull-escape count per checkpoint tracks this directly.

---

## Open Questions This Setup Targets

- Is SFT here doing hypothesis selection? Is EM the right framing for what SGD itself is doing?
- "Data quantity vs. prior weight" — once the apples-to-apples version is run, repeat with a 1,500/persona specialist variant to study this directly.
- How does distance of the induced prior from uniformity affect downstream SFT behavior?
- Relating fitted π deviation to parameter-norm trajectories (Betley et al.: complex personas penalized by SFT, observable in parameter norm at matched training performance).
- Best first-class measurement of deviation from base-model representation, correlated with P(data | hypothesis).

---

## Current Findings (carried forward)

Persona models overfit on 500 examples and end up with lower aggregate probabilities than the mixture; at minimum loss, the persona models win out as predicted by (1). This is consistent with the mixture acting as a regularizer due to dataset diversity. The apples-to-apples version above is designed to make this finding quantitative and to separate it from data-quantity confounds.

---

# Single-Epoch Overfitting Experiment (2026-05-31)

## Hypothesis
Persona overfitting in the prior run is caused by **multi-epoch training (data
reuse)**, not by gradient-step count or data quantity. Test: run a single
**single-epoch, lots-of-data** regime. If it does NOT overfit (val keeps
improving, hull-escapes stay low, specialists don't collapse below the mixture)
despite >= the prior run's gradient steps, multi-epoch reuse is the cause. We run
no multi-epoch regime; we contrast against the already-documented prior findings.

## Data
~10k/persona regenerated via the vendored SimpleStories pipeline (see notes.md),
local at `simple_stories_generate/data_raw/<persona>.jsonl`.

## Splits (per persona, from ~10k)
- `test.jsonl`  500   -> combined **2,500-seq** LoTP/hull-escape test set (was 750)
- `val.jsonl`   500   -> overfitting diagnostic: single-persona val (specialist)
                         and union = 2,500-seq full-mix val (mixture)
- `train.jsonl` ~9,000 -> specialist trains on own; mixture on the union (~45,000)

**DECISION (disjoint -> identical train).** The specialist's train split and this
persona's mixture-contribution are now the SAME ~9,000 examples, not disjoint as
in the prior run. Forced by the ~10k budget (disjoint -> ~4,500 each -> ~140 steps
< prior 160). Identical-train is also a STRONGER control: any mixture advantage on
a persona's held-out is then purely from cross-persona data, since the persona data
is identical. Apples-to-apples preserved (spec and mixture see the same 9,000/persona).

## Training (all 6 models; shared seed / LR / batch / AdamW / fp32)
- `NUM_EPOCHS = 1`.
- Specialist ~9,000/32 = ~281 steps; mixture ~45,000/32 = ~1,406 steps.
- Frequent within-epoch checkpointing (~20/model) + step 0 + final.
- Log `steps_per_epoch` so plots convert step -> fractional epoch.

## Diagnostics (per checkpoint)
- Train loss (running) + val loss on BOTH own single-persona val AND full-mix val.
  Overfitting = train down while val up.
- Per-seq Sum log P over the 2,500-seq test set (all 6 models cross-evaluated).
- Param norms ||theta||, ||theta - theta_base||.
- LoTP pi fit (KL primary) + hull-escape count on the test set.

## Plots (x-axis = fractional epoch 0->1)
- train-vs-val loss per model on both val sets (the overfitting story).
- hull-escapes / pi / own-held-out logP / param-norms vs epoch.
- On the epoch axis specialist & mixture have equal per-persona exposure at every x.

## Open
- Consume local data (build_splits reads `data_raw`) vs push regenerated data to
  HF. Default: local.
- LoTP pi endpoint alignment = each model's min-val-loss checkpoint; expect single-
  epoch min-val at/near the final checkpoint (no overfit) — informative if true.

## OUTCOME (2026-05-31) — see results/analysis.md and design_doc.md findings

- CONFIRMED: single-epoch does not overfit (min-val = final for all 6; specialists
  beat the mixture 5/5). Mixture-as-regularizer was a multi-epoch artifact.
- Min-val DID land at/near the final checkpoint for every model, as predicted above.
- NON-UNIFORM mixture (geom 1.5) revealed the LoTP π_KL estimator measures the
  EVAL-SET composition, not the induced prior. Replaced by the per-persona gap
  estimator (`induced_prior.py`): gap tracks the data proportion at r=+0.92.

## Next experiment (proposed): 2-D data-share × complexity

The gap entangles data share with persona complexity (absurdist learns worst for its
data share). To separate the data prior from the base-model prior signature, vary a
persona's mixture data share AND choose personas spanning known complexity, in a
grid, and regress the gap on (share, complexity). Reuse the same pipeline; only the
mixture compositions / persona set change.