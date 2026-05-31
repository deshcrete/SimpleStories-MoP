# Post-Run Analysis Log

Chronological record of the analysis discussion after the first end-to-end run
of the pipeline (commit on the `induce-prior` branch).

Pipeline: `src/run_experiment.py` →
  `build_splits.py` → 6× `train.py` → 6× `eval_checkpoints.py` → `lotp.py`.

All plots referenced live in `results/plots/`.

---

## 1. Initial run output

After fixing two environment issues (a) the HF dataset config naming
(`desh2806/simplestories-personas` has a single `default` config with persona
as a column; the earlier code tried to load configs by persona name) and
(b) the `transformers` ↔ `torch` version mismatch (pinned to
`transformers==4.46.3`), the full pipeline completed:

- Data: 19,000 total rows → **3,800 stories per persona** (much larger than
  the docs' ~2,150). Split sizes per persona: 500 / 500 / 150 / ~2,650.
- Training: 6 models (5 specialists + mixture), ~15 checkpoints each.
  Specialists: 160 steps. Mixture: 790 steps.
- Eval: per-sequence log-P matrices over the 750-seq combined inference set
  at every checkpoint, plus `‖θ‖` and `‖θ − θ_base‖`.
- LoTP: step-aligned + loss-aligned fits.

**Initial loss-aligned LoTP result (pre-fix):**

```
π = {
  noir_detective:        0.147,
  fairy_tale:            0.177,
  scientific_explainer:  0.380,
  absurdist:             0.147,
  epistolary:            0.147,
}
residual_rms = 225.77
hull escapes = 191 / 750  (tau=0, mean_gap=-146.7, max_gap=281.2)
```

Three personas landed at *identical* π = 0.14747… to 13 decimal places. The
remaining two were normal-looking. This was immediately suspicious.

---

## 2. Perturbed-init EM analysis

To check whether the EM fit was hitting a real fixed point or a degeneracy,
ran 26 initializations (uniform + 5 corners + 20 Dirichlet) of EM:

- All 26 inits converged to **the same residual_rms = 225.7652** to 4 decimals.
- The π values differed substantially across inits in the
  `{noir_detective, absurdist, epistolary}` dimensions — L1 distance up to
  0.49 between fits.
- `fairy_tale` (0.1774) and `scientific_explainer` (0.3801) were *pinned*
  across every init.

**Diagnosis:** the EM landscape has a flat direction. The mass allocated to
`{noir_detective + absurdist + epistolary}` sums to a constant (≈0.442) but is
freely redistributable among them.

Looking at the loss-aligned alignment selection:

```
noir_detective:        step 0    ← base model (untrained)
fairy_tale:            step 32
scientific_explainer:  step 48
absurdist:             step 0    ← base model
epistolary:            step 0    ← base model
```

Three "loss-aligned" checkpoints were step 0 — i.e., the untrained base model,
identical weights for all three runs. Their per-sequence log-P columns in the
LoTP matrix were bit-identical, so EM couldn't tell them apart → π was free
to move mass between them.

**Root cause:** `min_loss_step` picked the checkpoint with the highest
`total_logp` over the *full 750-seq* inference set. For three specialists,
training improved log-P on their own 150 sequences but degraded the other
600 *faster*, so the *total* log-P on 750 peaked at step 0 (the base model).
`task_plan.md` actually specifies "each model at its own min-eval-loss" — the
specialist's own held-out, not the full 750.

---

## 3. Audit pass (two parallel agents)

Two general-purpose audit agents reviewed the codebase against `task_plan.md`,
one on the training side and one on the eval/LoTP side. Most consequential
findings:

1. **`min_loss_step` (known)** — use each model's own held-out, not the
   full 750.
2. **EM was optimizing the wrong objective.** The implementation was standard
   mixture-model EM that maximizes `Σ_j log Σ_i π_i P_i(x_j)`. The spec
   defines `L(π) = mean_j (log P_mix − logsumexp(log P_i + log π_i))^2`.
   These are different optima in general.
3. **CUDA determinism flags not set** in `train.py` — reproducibility claim
   not enforced.
4. **`eval_loss` token-count off-by-shift** — HF's `.loss` is mean over n−1
   shifted positions, but the code weighted it by un-shifted n. Logged
   `eval_loss` values were biased by a small constant; checkpoint ranking
   unchanged.
5. **Silent fallbacks** — `git_sha → "unknown"` on failure; `max(total_tokens, 1)`
   hides an empty eval batch.
6. **EOS token id hardcoded** — should assert against the tokenizer.

Fixes were applied for all of these.

---

## 4. Re-run with fixed loss alignment + spec L(π) fit

After fixing `min_loss_step` to use each model's own held-out (specialist's 150
seqs / mixture's 750), and replacing EM with a direct L-BFGS-B minimization
of the spec's L(π):

**New loss-aligned alignment:**

```
noir_detective:        step 80     (was 0)
fairy_tale:            step 64
scientific_explainer:  step 64
absurdist:             step 80     (was 0)
epistolary:            step 64
mixture:               step 200
```

**Residual fit (spec L(π)):**

```
π = {
  noir_detective:        0.404,
  fairy_tale:            0.000,    ← driven to zero
  scientific_explainer:  0.478,
  absurdist:             0.006,
  epistolary:            0.112,
}
residual_rms = 92.77
```

**Mixture-EM fit (likelihood) at the same alignment:**

```
π = {
  noir_detective:        0.200,
  fairy_tale:            0.197,
  scientific_explainer:  0.203,
  absurdist:             0.200,
  epistolary:            0.200,
}
residual_rms = 92.96
```

Both fits perturbation-robust (L1 distance < 0.0001 across 26 inits).

**Hull escapes: 327 / 750** (was 191). Specialists are now actually trained,
so they more often beat the mixture on their own data. `max_gap = 403` —
huge worst-case escape.

The two fits disagree dramatically. EM says "uniform with tiny sci tilt";
residual says "no mass on fairy or absurdist, 40% noir + 48% sci". Per the
spec the residual fit is primary, but its answer is bizarre.

---

## 5. Magnitude diagnosis

To pin down the disagreement, computed the per-sequence log-P matrix at the
loss-aligned alignment and the residual against a uniform π directly:

```
=== Mean log P per sequence (rows = scoring model, cols = source persona) ===
                       noir_det  fairy_ta  scientif  absurdis  epistola
  noir_detective         -441.8    -671.8    -866.8   -1007.5    -795.0
  fairy_tale             -784.2    -363.5    -676.1    -871.1    -702.0
  scientific_explainer   -781.9    -462.2    -494.9    -848.6    -626.4
  absurdist              -892.4    -676.7    -854.0    -489.0    -847.4
  epistolary             -866.3    -716.0    -922.1   -1112.6    -333.5
  mixture                -433.4    -375.0    -487.6    -493.9    -333.1
```

The mixture row is within ~10 nats of each specialist diagonal — at this
alignment the mixture is *competitive* with every specialist on the
specialist's own persona, and beats them on noir and sci.

**Residual under uniform π = 1/5:**

```
log P_mix range:   -1306 .. -112      span 1194 nats
mean(log P_mix):   -424.6

lse(log P_i + log(1/5)) − log P_mix:   mean = -1.53   std = 92.95
```

**The magnitude issue:**

- Per-sequence log-P magnitudes are ~hundreds of nats and scale roughly with
  sequence length (T ≈ 100..400 → log P ≈ −100..−1300, span 1200 nats).
- The mean residual under uniform π is essentially zero (−1.5 nats). The
  LoTP identity *already holds on average*.
- But the std of the residual is ~93 nats — this is per-sequence noise.
- The spec's L(π) = `Σ_j residual_j²` is dominated by `var(residual_j)`, not
  `mean(residual_j)`. Minimizing it shrinks the variance by ~0.2 nat
  (92.95 → 92.77) by trading off mass across personas in essentially
  arbitrary ways.
- The "weird" residual π = [0.40, 0.00, 0.48, 0.01, 0.11] is the optimizer
  exploiting per-sequence noise; it isn't a real prior signature.

The squared-residual loss in log-space is the wrong objective when the per-
sequence log-P magnitudes are this large.

---

## 6. Switch to KL (forward) as the primary fit

KL of `empirical || M` is equivalent (up to a constant) to maximizing
`mean_j log M(x_j)`:

    KL(P_mix || M) = E[log P_mix - log M] = (const) - E[log M]

→ argmin over π = argmax mean log M = standard mixture-model MLE = what EM
finds. Mathematically the same as our old EM fit; better motivated and
insensitive to per-sequence log-P magnitude offsets.

Made `fit_pi_kl` the primary fit (L-BFGS-B on softmax-parameterized logits
maximizing `mean_j logsumexp_i(log P_i + log π_i)`). Demoted `fit_pi_residual`
to a diagnostic. Both reported side-by-side.

**Loss-aligned KL result (primary, current):**

```
π = {
  noir_detective:        0.20000   (uniform)
  fairy_tale:            0.19712
  scientific_explainer:  0.20287   ← real but small positive tilt
  absurdist:             0.20000
  epistolary:            0.20000
}
avg log M(x) = -426.13    (vs -429.21 for the residual fit's weird π)
```

KL fit assigns higher avg log-likelihood than the residual fit's π by ~3 nats
per sequence — the right diagnostic that it's the better fit even by the
residual fit's own intended quality criterion.

The signal is tiny: scientific_explainer at 0.203 vs fairy_tale at 0.197, a
3% deviation from uniform 0.20.

---

## 7. Bootstrap error bars on π_KL

500 bootstrap resamples of the 750-seq inference set (with replacement), each
refitting π_KL:

| persona | π | bootstrap std | 95% CI | excludes 1/5? |
|---|---|---|---|---|
| noir_detective | 0.2000 | ±0.0140 | [0.172, 0.227] | no |
| fairy_tale | 0.1971 | ±0.0142 | [0.170, 0.226] | no |
| scientific_explainer | 0.2029 | ±0.0152 | [0.174, 0.236] | no |
| absurdist | 0.2000 | ±0.0136 | [0.173, 0.228] | no |
| epistolary | 0.2000 | ±0.0147 | [0.171, 0.233] | no |

**The induced-prior signal we recover (~0.003) is ~20× smaller than the
sampling uncertainty (~0.014).** At this run's scale and overfitting regime
we *cannot* statistically distinguish π from uniform. The point estimate
shows a sci tilt; the CIs all span 1/5.

Raw stats: `results/lotp/pi_kl_bootstrap.json`.
Plot: `results/plots/pi_kl_errorbars.png`.

---

## 8. Overfitting trajectory (per-persona log-P)

Per-persona log-P over training steps for each specialist on its own 150
held-out seqs, plus the mixture model on the same slice:

| persona | spec peak (@ step) | spec at end (step 160) | mix peak (@ step) | mix at end (step 790) |
|---|---|---|---|---|
| noir_detective | −66,266 (80) | −76,743 | **−65,005** (200) | −76,527 |
| fairy_tale | **−54,532** (64) | −66,344 | −55,647 (140) | −68,379 |
| scientific_explainer | −74,236 (64) | −90,053 | **−73,146** (200) | −86,278 |
| absurdist | **−73,344** (80) | −82,472 | −74,086 (200) | −85,988 |
| epistolary | −50,026 (64) | −56,971 | **−49,968** (200) | −58,956 |

Bold = best of the two. **The mixture beats the specialist peak on 3 of 5
personas** (noir, sci, epistolary). Specialists win on fairy_tale and
absurdist — the two with the most distinctive surface markers (which
specialists can memorize).

Every specialist overfits hard: 10,000–17,000 nat drop from peak (step 64–80)
to step 160. Mixture overfits more gently — peaks later (140–200) and drops
~10–13k nats by step 790.

Cross-persona transfer is essentially zero — other specialists score this
persona's data far below both own-specialist and mixture lines. The mixture's
advantage comes from *diverse data of all 5 personas*, not from generalist
representations.

Plot: `results/plots/logprob_overfit.png`.

---

## 9. Hull escapes — KL bypass mechanism and overfitting trajectory

A hull escape on sequence j is `max_i log P_i(x_j) > log P_mix(x_j)`. On such
a sequence, `log P_mix = logsumexp(log P_i + log π_i)` cannot hold for any
non-negative π (the RHS is bounded below by `max_i log P_i`).

**Residual fit** measures the *equality residual* and pays a per-escape
quadratic penalty proportional to `gap²`. Empirically `corr(max_gap,
residual_rms) = +0.97` — residual_rms tracks the worst-case escape almost
perfectly. The residual fit's π is being yanked around by the handful of
catastrophically-overfit sequences.

**KL fit** maximizes `mean log M`, never references `log P_mix`. On an
escape sequence, some `log P_i` is high → logsumexp is high → log M is high →
KL is *happy*. Escapes don't enter the KL gradient at all.

**Concrete demonstration:** from step 180 onward (after specialists pin at
their final step), the KL objective **locks at −498 and stays flat through
step 790**, even though escape count climbs from 100 → 384 and max_gap climbs
from 720 → 1000 nats. Specialists are fixed → log P_i are fixed → π_KL is
essentially uniform → mean log M is constant. The mixture's late-stage
overfitting (log P_mix dropping ~10k nats from step 200 to 790) is completely
invisible to the KL fit.

**Hull-escape trajectory:** U-shaped in step.

- step 40: **640 escapes** (peak). Specialists differentiate faster than the
  mixture catches up.
- steps 140–300: **~100 escapes** (minimum). Loss-aligned region; both models
  converged.
- step 790: **384 escapes**. Specialists pinned at step 160; mixture is the
  one overfitting now.

Max gap rises monotonically from 0 → 1000 nats — the *worst* escape gets
worse the whole time, even when the *count* dips in the middle.

Plot: `results/plots/escapes_vs_fits.png`.

---

## 10. Training-step asymmetry (specialist vs mixture)

The spec sets 10 epochs for every model. Specialists train on 500 stories
(16 steps/epoch → 160 total). Mixture trains on 2,500 (79 steps/epoch → 790
total). The mixture therefore does ~5× more optimizer updates by
construction.

Two perspectives on "how much has the model learned persona p?" at the
loss-aligned alignment (spec step 80, mix step 200):

| | spec (step 80) | mix (step 200) | mix (step 790, end) |
|---|---|---|---|
| Total gradient updates | 80 | 200 | 790 |
| Epochs over own training set | 5.00 | 2.53 | 10.00 |
| Updates touching persona p's data | 80 | 40 | 158 |
| Per-persona epoch count | 5.00 | 2.53 | 10.00 |

**At loss-aligned, the specialist has seen persona p's 500 stories ~2× more
than the mixture has.** Mixture-equivalent of "5 specialist epochs" is mix
step ~400, not step 200. The mixture's own min-eval-loss arrives at 2.5
per-persona-epochs — half the specialist's 5. Cross-persona gradient updates
are doing implicit regularization: fewer own-persona passes needed before the
mixture overfits.

**Consequence for results.** This *strengthens* the mixture-as-regularizer
finding: the mixture beats specialists on 3 of 5 personas with strictly less
per-persona learning. Conversely, it means step-aligned trajectory plots past
step 160 (where specialists pin at their final step) compare a fully-overfit
specialist against a mixture still progressing through its per-persona
epochs — apples-to-oranges past step 160.

Three alignment options:

| alignment | spec step | mix step | what it measures |
|---|---|---|---|
| step-aligned | S | S | raw SGD trajectory, mixture undertrained early |
| loss-aligned | own-min | own-min | each model at its own best, unequal exposure |
| per-persona-epoch | S | 5·S | matched per-persona exposure |

Per-persona-epoch alignment would pair `spec 80 ↔ mix 400` (both at 5 epochs
of any given persona's 500 stories) and `spec 160 ↔ mix 790` (both at 10
epochs). Not implemented in `lotp.py` yet — worth adding as a third mode for
the next run.

---

## 11. Persona complexity under the base model

Per-token base log P on each persona's own 150 sequences (computed at step 0,
which is identical across all six runs):

| persona | base log P / token | base log P (sum) | improvement (spec peak − base) | dev_norm at spec peak | π_KL |
|---|---|---|---|---|---|
| fairy_tale (easiest) | **−2.12** | −63,802 | +9,270 | 9.04 | 0.1971 |
| sci_explainer | −2.95 | −101,242 | +27,005 | 8.93 | 0.2029 |
| epistolary | −3.62 | −85,897 | +35,871 | 9.95 | 0.2000 |
| noir_detective | −3.74 | −112,681 | +46,415 | 10.11 | 0.2000 |
| absurdist (hardest) | **−4.49** | −158,903 | **+85,559** | **10.19** | 0.2000 |

**Complexity span is large**: per-token base log P ranges from −2.12 (fairy)
to −4.49 (absurdist), 2× variation in difficulty. Fairy_tale text sits deep
inside the base distribution ("Once upon a time" is everywhere in pretraining);
absurdist text (non-sequiturs, broken logic) is far outside it. Both the
improvement-from-base and the final parameter-deviation norm rank-order
correlate with this — absurdist takes the most log-likelihood gain and the
largest weight move.

**Correlations with π_KL (n=5, directional only):**

| measure | Pearson r with π_KL |
|---|---|
| base log P (sum) | −0.37 |
| base log P / token | −0.33 |
| specialist peak log P (own) | −0.63 |
| improvement (peak − base) | +0.22 |
| steps to peak | 0.00 |
| dev_norm at peak | −0.06 |

Direction: **harder personas (more negative base log P) get *more* π_KL
mass.** Statistically uncallable at n=5 with the π range (~0.006) being
smaller than the bootstrap CI half-width (~0.014), but the sign is consistent
across three independent complexity measures.

This is *opposite* of one reading of Betley et al. ("broad/easy hypotheses
get more mass post-finetune"). A different interpretation that fits the data:
π_KL reflects "where the mixture had to commit more capacity" — and the model
has to commit more capacity where the prior is bent furthest from base, so
harder personas get slightly more posterior mass. Can't tell which story is
right without more inference data.

**The U-shape — specialists win at both extremes.** Sort personas by base
log P / token and look at who wins at the loss-aligned alignment:

| persona | base/tok | log P_mix − log P_spec per seq | winner |
|---|---|---|---|
| fairy_tale (easiest) | −2.12 | −11.5 | spec |
| sci | −2.95 | +7.3 | mix |
| epistolary | −3.62 | +0.4 | tied |
| noir_detective | −3.74 | +8.4 | mix |
| absurdist (hardest) | −4.49 | −4.9 | spec |

**Specialists win on the easiest and the hardest personas; mixture wins in
the middle.** Mechanism: extreme personas have the most distinctive surface
markers (fairy_tale's openers / morals; absurdist's structural
non-sequiturs) that a specialist can memorize tightly. Middle-complexity
personas rely on more diffuse stylistic features that benefit from
representation sharing across the mixture's 5-persona training set.

This U-shape is independent of the π_KL question and is the clearer
complexity finding from this run.

Plot: `results/plots/complexity_analysis.png` (panels A: base/token; B:
dev_norm at peak; C: scatter of base/token vs π_KL; D: scatter of base/token
vs mix-minus-spec gap, showing the U-shape).

---

## Summary of findings

1. **The induced-prior signal in this run is real but tiny.** π_KL =
   `[0.200, 0.197, 0.203, 0.200, 0.200]`, scientific_explainer favored by
   ~3% over uniform, fairy_tale disfavored by ~1.5%.
2. **The signal is below the noise floor** — 95% bootstrap CIs all span 1/5.
   To detect this prior signature statistically would need ~10× the inference
   set or ~5× the persona separation.
3. **Mixture-as-regularizer carries through to the apples-to-apples version.**
   The mixture beats the specialist's own peak on 3 of 5 personas. Per
   `task_plan.md` §"Predicted LoTP Behavior", this is the strong claim about
   cross-persona transfer / shared structure in the base model.
4. **Per-sequence log-P magnitudes scale with sequence length** (T ≈ 100–400,
   log P ≈ −100 to −1,300 nats). Any squared-residual loss in log-space is
   dominated by per-sequence noise variance rather than by the actual signal
   about π. The spec's `L(π)` is the wrong objective at this scale.
5. **Forward KL is the right LoTP fit here.** Equivalent to mixture-MLE,
   insensitive to magnitude offsets, well-defined when the LoTP equality
   doesn't hold per-sequence (which it never does because of hull escapes).
6. **Hull escapes are U-shaped over training** but max_gap grows monotonically.
   The residual fit's loss tracks max_gap (corr +0.97); the KL fit's loss
   is invariant.
7. **Training-step asymmetry matters but doesn't break the analysis.** The
   mixture does 5× more optimizer updates than each specialist (same epochs,
   5× more data). At loss-aligned, specialist has had 2× more per-persona
   exposure than mixture — and mixture still wins on 3 of 5 personas, which
   strengthens the regularizer claim. A `per_persona_epoch` alignment mode
   would pair `spec 80 ↔ mix 400` etc.; worth adding next run.
8. **Persona complexity under the base model spans 2×** in per-token log P
   (fairy_tale −2.12 → absurdist −4.49). Specialist improvement (+9k to
   +86k) and dev_norm (9.0 to 10.2) both track this. The π_KL signal points
   weakly at "more mass on harder personas" (corr −0.33 with base/token),
   but n=5 with the signal smaller than the bootstrap CI means we can't
   call it. The U-shape — specialists winning at both ends of complexity,
   mixture winning in the middle — is the clearer complexity finding.

---

# Single-Epoch & Non-Uniform-Mixture Experiments (2026-05-31)

A second experiment series on the `induce-prior` branch. Goal: test whether the
overfitting in the first run was caused by **multi-epoch data reuse**, then probe
how the recovered prior behaves under a **non-uniform** data mixture. This series
produced a methodological correction to the project's central LoTP method.

## Setup changes vs the first run

- **Data regenerated to ~10k/persona** via the vendored SimpleStories pipeline
  (`simple_stories_generate/`, upstream `bf306bd`), with persona as the *controlled*
  variable. Five elaborated persona system prompts (`persona_system_prompts.json`,
  reverse-engineered from `desh2806/simplestories-personas`) achieved ~100% style
  adherence after tuning (2-5 paragraphs, ≤10 stories/completion, in-prompt persona
  reinforcement). 50,472 stories total, `gpt-4o-mini`.
- **Splits:** per persona `test 500 / val 500 / train ~9,000`. The specialist's
  train split *is* the persona's mixture contribution (identical, not disjoint —
  forced by the ~10k budget and a cleaner control).
- **Single epoch.** ~281 specialist / ~1,421 (uniform) mixture steps — *more* than
  the first run's 160, so a no-overfit result can't be dismissed as undertraining.
- **GPU/torch:** Blackwell (sm_120) required torch 2.11+cu128; see notes.md.

## Result 1 — single-epoch training does NOT overfit

- **Val loss falls monotonically to the final checkpoint for all 6 models**
  (min-val ≈ epoch 1.0; val ~2.5 → 1.7). No U-shape. (`overfit_curves.png`.)
- **Specialists beat the mixture on 5/5 personas** (+16 to +35 nats/seq),
  *reversing* the first run's "mixture beats specialist on 3/5". The
  mixture-as-regularizer effect was therefore an **artifact of specialist
  overfitting**: remove the reuse and the specialist is simply better on its own
  data; the mixture never rescues an overfit specialist. (`logprob_overfit.png`.)
- **Convex-hull escapes are rare:** the mixture exceeds *all* specialists on only
  163/2500 (6.5%) of test seqs (mean excess −29 nats — specialists better on
  average). This used the **corrected** hull sign (see below).
- `π_KL` (uniform mixture) = uniform 0.2000 — but see Result 2 for why that is
  meaningless.

## Result 2 — the non-uniform mixture exposes a broken estimator

A second mixture (`mixture_exp`) was trained on a geometric (ratio 1.5) blend,
canonical order: noir 0.384 / fairy 0.256 / sci 0.171 / absurd 0.114 / epist 0.076
(~23.5k examples). Specialists were *reused* (the fixed P_i basis); only the
mixture composition changed.

- **Recovered `π_KL` was bit-identical (0.20000001) for the uniform and the
  non-uniform mixtures** — completely blind to the strongly non-uniform data.
  (`pi_vs_proportions.png`.)
- **Root cause:** `fit_pi_kl` never references `P_mix`. It maximizes the test-sample
  likelihood under `Σ_i π_i P_i`; because each specialist is ~100 nats better on its
  own persona (near-disjoint support), the matched specialist dominates the
  logsumexp and the MLE just recovers **how often each persona appears in the eval
  set**. Probe (`eval_composition_probe.py`): re-fitting `π_KL` on non-uniform eval
  subsets reproduces the eval composition to **0.0000** (uniform, geom-1.5, and
  reversed). (`eval_composition_probe.png`.)
- **Implication:** the first run's headline "induced prior π ≈ uniform" was an
  artifact of (estimator blind to the mixture) × (uniform 750-seq eval set). It was
  never measuring the induced prior. The residual fit (which *does* use P_mix) is
  degenerate (collapses to one persona), so neither LoTP estimator works.

## Result 3 — the gap-based induced-prior estimator (the fix)

The induced prior is recoverable directly, per persona, at each model's min-val
checkpoint (`induced_prior.py`):

    gap_p = mean over persona-p test seqs of ( log P_mix - log P_spec_p )   [nats/seq]

- **Non-uniform mixture: `gap_p` tracks the data proportion at r = +0.92** — the
  more data a persona contributed, the closer the mixture is to its dedicated
  specialist. **The mixture genuinely internalized the non-uniform prior**; it is
  just invisible to `π_KL`. (`induced_prior.png`, right panel.)
- **Uniform mixture:** with data held equal at 0.2, gaps still spread −18 → −35.
  That spread is the **base-model prior / complexity signature** (epistolary easiest
  to fold into the mixture, absurdist hardest). (`induced_prior.png`, left panel.)
- **absurdist is a systematic outlier** (well below the trend): the hardest persona
  learns *worse* than its data share predicts. So induced prior = data share
  **modulated by persona complexity**.
- **A normalized `π` is not recoverable.** Gaps are 18-90 nats — far larger than
  `log(π)` for any plausible π — so the trained mixture is **not** a convex
  combination of the specialists; `P_mix ≈ Σπ_i P_i` does not hold under near-disjoint
  specialist support. This single fact explains both estimator failures. A
  complexity-normalized `learn_frac = (log P_mix − log P_base)/(log P_spec − log P_base)`
  was tried but is unstable for the easiest persona (tiny denominator), so it is kept
  only as a noisy diagnostic.

## Methodology corrections made this series

1. **Hull-escape sign fixed.** `hull_escape_count` previously counted
   `max_i P_i > P_mix` (the inequality *holding*) and called it an escape. The
   design-doc "escape the convex hull" is the *violation* `P_mix > max_i P_i`. Sign
   and field names (`mean_excess`/`max_excess`) corrected; the first run's
   hull-escape narrative was inverted.
2. **Loss-alignment now uses each model's min-VAL checkpoint** (`min_val_loss_step`),
   not min-test-logp — keeps the LoTP test set out of model selection (now that a
   separate val split exists).
3. **`π_KL` deprecated as an induced-prior measure;** the gap-based estimator is
   adopted. `π_KL` is retained only as a (correct) measure of eval-set composition.

## Result 4 — the generation-based induced prior (sample from the mixture, classify)

The right way to use the KL fit: it recovers the persona composition of whatever
samples it is fed. Feed it samples **drawn from the mixture** (not the eval set) and
it returns the mixture's own generation composition = `P_mix(persona)`.
`src/generate_and_classify.py`: generate N=2000 stories from each mixture's min-val
checkpoint (seeded with **EOS** — the SimpleStories model has no BOS and treats EOS
as "begin a new story"), score each under the 5 specialists, report the prior via
`argmax_i log P_i` and via the KL fit, and compare to the training proportions.

**Validation (the measure is trustworthy):**
- argmax and KL fit agree to **L1 ≈ 0.007**.
- The specialist argmax classifier is **100% accurate on the 2,500 real labeled test
  stories** (perfect confusion matrix — the ~100-nat gaps make clean persona text
  trivial to classify).
- Classifier-**independent** rule-based markers on 600 generations agree with the
  classifier: causal "because" **52%** (vs sci 54% by classifier), epistolary "Dear"
  **0.3%** (vs 0.1%), "Once upon a time" **0.0%**, first-person ~23%, generic
  third-person ~21%.

**The result — the generated prior does NOT track the training data:**

| persona | trained (non-unif) | generated (KL) |
|---|---|---|
| noir_detective | 0.384 | 0.060 |
| fairy_tale | 0.256 | 0.065 |
| scientific_explainer | 0.171 | **0.540** |
| absurdist | 0.114 | 0.333 |
| epistolary | 0.076 | 0.001 |

corr(training, generated) = **−0.27**. The uniform mixture is also strongly skewed
(sci 0.39 / absurd 0.38 / noir 0.15 / fairy 0.08 / epist 0.008, vs 0.2 each). The
mixture massively over-generates causal/sci narration and **almost never** produces
the templated formats (letters, the fairy opener) — *regardless of training share*.
(`generated_prior.png`.)

**Synthesis — "induced prior" is two different quantities that disagree:**
1. **Conditional competence** (the gap measure, Result 3): on real persona-p text,
   how close is the mixture to that specialist → **tracks the data proportion,
   r = +0.92**.
2. **Free-generation marginal** (this result): what the mixture *produces*
   unconditionally → **base-model-dominated, r = −0.27 with the data**.

Data composition controls how well each persona's text is **modeled**, but barely
controls what the model **generates** unconditionally — free sampling reverts to the
base model's preferred register (generic causal narrative) and washes out the
highly-templated personas (epistolary letters, fairy's fixed opener are the hardest
to reproduce when sampling). That base-prior signature is the original project goal,
finally visible — in *generation* rather than in `π_KL`.

## Caveats / open follow-ups

- **Determinism.** torch 2.11's CUDA attention backward is non-deterministic
  (warn_only). Eval log-probs are deterministic (forward); training weights vary by
  tiny amounts run-to-run. All effects here are 10-90 nats, far above that noise, so
  conclusions are unaffected; a math-SDP-backend re-run would restore bit-identical
  reproducibility.
- **n = 5 personas.** Correlations (r = +0.92 etc.) are suggestive, not powered.
- **Data-share / complexity entanglement.** The gap reflects both. A 2-D design that
  varies a persona's data share *and* its complexity independently would separate
  the data prior from the base-model prior signature — the natural next experiment.

