Methodology:

Use the same sampling procedure as Simple Stories and create a new dataset of 2150 stories per persona

Personas:
noir_detective — first-person hardboiled detective narrator; short, clipped sentences; dark atmosphere.
fairy_tale — classic "Once upon a time" opener; third-person omniscient; ends with an explicit moral.
scientific_explainer — narrator pauses to explain how/why things happen with cause-and-effect framing.
absurdist — non-sequiturs and broken narrative logic; no plot, no resolution.
epistolary — story told entirely as letters between two characters.

Split the data for each persona into a persona training split, mixture split and inference split

Finetune a SimpleStories 5 million parameter model on a mixture dataset of all the mixture splits 
Finetune SimpleStories 5 million parameter models on the persona training splits

Combine all the inference splits into an inference dataset and then calculate the probability of seeing each sequence under each of the personas and the mixture
What we want is P_{persona_i}(sequence) -> what is the best way of measuring this probability?

This gives us the following Law of Total Probability (LoTP) formulation:  for a well converged models that have not overfit
P_{mixture}(sequence) <= argmax_i P_{persona_i}(sequence) i.e the mixture cannot be better than the best persona at a given sequence
If (1) holds then we can suggest: P_{mixture}(sequence) \approx \sum_i P_{persona_i}(sequence) * P(persona_i) where P(persona_i) is inferred as a linear system which represents the prior 
Problem with this formulation is that it is very sensitive to differences in training and convergence as we escape the convex hull very easily. 

Plots and Measurements:
P(inference dataset | persona_i) = \prod_j P_{persona_i}(sequence_j) over the set of training/gradient steps
Number of times we escape the convex hull at every checkpoint we plot the probabilities (as we overfit more we escape the convex hull more)
Parameter norm of each hypothesis at each checkpoint


Notes:

We want to try to understand supervised fine tuning through a bayesian perspective. To do this we want to induce a prior on a base LLM where the prior is a distribution over “personas” or “hypothesis”
The base model itself has a underlying prior which means that if we were to try to finetune it on a dataset that has samples distribution according to our intended distribution over the set of “personas” the model will place more or less probability depending on what persona it has already learnt/is easily representable
The less a persona is “represented” in the base model, the more the model will place probability on that persona due to higher “gradient pressure”
Complex personas are penalized by SFT (Betley et al)
Basically after training we don't get uniform because of this underlying base model prior selecting for certain personas or hypothesis
Is expectation maximization the right formulation here cause what the model is doing is fundamentally hypothesis selection?
LoTP inference over the base model?
"data quantity vs. prior weight" - does the amount of data seen have an effect?
How does the shape of the prior/distance from uniformity have an effect?
At what point in the training should we infer LoTP?
Step-aligned: Compare all models at the same step count. Cleanest for studying SGD trajectory, but the mixture will be undertrained relative to the specialists at any given step.
Loss-aligned: Compare each model at its own minimum-eval-loss checkpoint. Cleanest for studying the converged endpoint, but you're comparing models at different points in their training.

Sources:

(Betley et al.) Weird Generalizations and Inductive Backdoors 
“which hypothesis among data-equivalent ones gets selected” -> Is the model doing hypothesis selection so is expectation maximization the correct framing?
Content of earlier training data (pretraining) influences later generalizations
P(narrow hypothesis) << P(broad hypothesis) in base model => model in finetuning will place a bit more probability on the broad hypothesis
Complexity in terms of parameter norm for a given level of training performance (measure the parameter norm through training steps?)
“we finetune using LoRA and on only 208 examples for 3 epochs, which means we likely deviate little from GPT-4.1 in terms of the set of representations available”
Want some first class was of measuring deviations from base model representation and how that correlates with P(dataset | hypothesis)

Findings:

Currently, if we train the mixture and persona models the persona models overfit resulting in the models having much lower probabilities than the mixture model but at the lowest possible loss, the persona models win out compared to the mixture as predicted.
This suggests that the mixture model “regularizes” due to the diversity of the dataset


---

## Findings — apples-to-apples run (induce-prior branch)

Full chronological analysis log is at `results/analysis.md`. This section
summarizes the key results.

### Setup actually run
- 5 personas × 500 persona-train / 500 mixture-contribution / 150 inference /
  ~2,650 reserve. Mixture trained on 2,500 stories (union of mixture-contrib
  splits).
- Base: SimpleStories/SimpleStories-V2-5M. AdamW lr=5e-4, batch 32, wd=0,
  10 epochs, seed=42 shared across all six runs.
- 6 runs (5 specialists + mixture), ~15 checkpoints each. Specialists ran to
  step 160; mixture to step 790. Combined 750-seq inference set
  cross-evaluated at every checkpoint.

### Headline result: induced prior is uniform within sampling noise

Loss-aligned KL fit (each model at its own min-eval-loss checkpoint; KL =
forward `KL(empirical || M)` ≡ maximizing mixture log-likelihood):

| persona | π_KL | bootstrap 95% CI | tilt vs 1/5 |
|---|---|---|---|
| noir_detective | 0.2000 | [0.172, 0.227] | ≈0 |
| fairy_tale | 0.1971 | [0.170, 0.226] | −0.003 |
| scientific_explainer | 0.2029 | [0.174, 0.236] | +0.003 |
| absurdist | 0.2000 | [0.173, 0.228] | ≈0 |
| epistolary | 0.2000 | [0.171, 0.233] | ≈0 |

Point estimate shows scientific_explainer slightly preferred and fairy_tale
slightly disfavored — directionally consistent with the
"base model finds sci easier / fairy_tale relies on overt surface markers"
intuition. But every 95% CI spans 1/5: the signal is ~20× smaller than the
sampling uncertainty (~0.003 vs ~0.014). **We cannot statistically distinguish
π from uniform with 750 inference sequences.**

### Mixture-as-regularizer effect holds in the apples-to-apples version

At each persona's peak (own held-out log-P) the mixture beats the specialist
on 3 of 5 personas:

| persona | spec peak | mix peak (better in **bold**) |
|---|---|---|
| noir_detective | −66,266 | **−65,005** |
| fairy_tale | **−54,532** | −55,647 |
| scientific_explainer | −74,236 | **−73,146** |
| absurdist | **−73,344** | −74,086 |
| epistolary | −50,026 | **−49,968** |

The mixture sees the same 500 examples of each persona as the specialist, plus
2,000 examples of *other* personas. That extra cross-persona data makes it
strictly better on noir / sci / epistolary than a same-budget specialist.
Per task_plan §"Predicted LoTP Behavior" this is the strong claim about
cross-persona transfer / shared structure in the base model. Specialists win
only on fairy_tale and absurdist — the two personas with the most distinctive
surface markers (which specialists can memorize easily).

Cross-persona transfer between *specialists* is essentially zero — each
specialist's log-P on another persona's data stays near the base-model
baseline. The mixture's advantage is from *diverse data*, not from
generalist representations emerging in a single trained model.

### LoTP fit: the spec's L(π) is the wrong objective at this scale

Per-sequence log-P magnitudes are ~hundreds of nats and span ~1,200 nats
across the 750-seq inference set (sequence length T ≈ 100–400). At the
loss-aligned alignment:

    lse(log P_i + log(1/5)) − log P_mix:   mean = -1.53   std = 92.95

The LoTP identity already holds to ~1.5 nats *on average* under uniform π,
but per-sequence noise has std ~93 nats. The spec's
`L(π) = mean_j (log P_mix − logsumexp(log P_i + log π_i))^2` is dominated by
the variance of per-sequence residuals — minimizing it just shrinks std by
~0.2 nats by trading off mass across personas in ways that exploit per-
sequence noise. The result is a degenerate-looking π
`= [0.40, 0.00, 0.48, 0.01, 0.11]` that doesn't reflect any real prior
signature.

**We adopted forward KL `KL(empirical || M)` as the primary fit.** Up to a
constant this is equivalent to maximizing `mean_j log M(x_j)` — the standard
mixture-MLE that EM converges to. KL is insensitive to per-sequence log-P
magnitude offsets and to hull-escape violations (it never references log P_mix
in the gradient). The squared-residual fit is kept as a diagnostic only.

### Hull escapes and overfitting

The LoTP inequality `P_mix ≤ max_i P_i` is broken on 100–640 of 750 sequences
depending on training step:

- **U-shaped trajectory.** Peak at step 40 (≈640 escapes; mixture undertrained
  while specialists differentiate), minimum at the loss-aligned step 200
  (≈100 escapes), rising again to 384 by step 790 as the mixture itself
  overfits.
- **Max gap rises monotonically** from 0 to ~1,000 nats — the *worst*
  escape gets worse throughout training, even when the *count* dips.
- The residual fit's loss tracks max_gap with `corr = +0.97` — its π is being
  yanked around by ~5 catastrophically-overfit sequences.
- The KL fit's objective locks at −498 once specialists pin at step 160 and
  stays flat through step 790, completely invariant to the mixture's late-
  stage overfit. KL "bypasses" hull escapes because it never tried to
  satisfy the LoTP equality in the first place; it only fits M as a
  marginal-likelihood model of the inference set.

### Training-step asymmetry (specialist vs mixture)

Same epochs (10) × 5× more data → mixture does ~5× more optimizer updates.
At loss-aligned (spec step 80, mix step 200):

| | spec | mix |
|---|---|---|
| total gradient updates | 80 | 200 |
| per-persona epochs over the 500-story slice | 5.00 | 2.53 |
| updates touching persona p's data | 80 | 40 |

The specialist has had **2× more per-persona exposure** than the mixture at
the loss-aligned alignment. The mixture's own min-eval-loss arrives at 2.5
per-persona-epochs vs the specialist's 5 — strong evidence that cross-persona
gradient updates are doing implicit regularization (fewer own-persona passes
needed before overfitting starts). This *strengthens* the mixture-as-
regularizer claim: the mixture beats the specialist on 3 of 5 personas with
strictly less per-persona learning at the comparison point.

A third alignment mode worth adding for the next run is
**per-persona-epoch-aligned** (`spec step S ↔ mix step 5·S`), which matches
the per-persona data exposure exactly. Step-aligned trajectory plots past
step 160 (where specialists pin at their final checkpoint) compare a fully-
overfit specialist against a mixture still progressing through its per-persona
epochs and aren't strictly apples-to-apples in that range.

### Persona complexity under the base model

Per-token base-model log P on each persona's own 150 inference seqs:

| persona | base log P / token | spec improvement | dev_norm at spec peak | π_KL |
|---|---|---|---|---|
| fairy_tale (easiest) | **−2.12** | +9,270 | 9.04 | 0.1971 |
| scientific_explainer | −2.95 | +27,005 | 8.93 | 0.2029 |
| epistolary | −3.62 | +35,871 | 9.95 | 0.2000 |
| noir_detective | −3.74 | +46,415 | 10.11 | 0.2000 |
| absurdist (hardest) | **−4.49** | **+85,559** | **10.19** | 0.2000 |

Complexity spans 2× in per-token log P. Improvement and dev_norm both
rank-order with complexity — absurdist requires ~9× the log-likelihood
gain and the largest parameter move; fairy_tale requires the least of both.
**π_KL directionally tracks complexity (more mass on harder personas, corr
≈ −0.33)** but the signal is below the bootstrap CI; can't be called at n=5.

**The robust complexity finding is a U-shape in mix-vs-spec performance**:

| persona | base/tok | mix − spec  per seq | winner |
|---|---|---|---|
| fairy_tale | −2.12 | −11.5 | spec |
| sci | −2.95 | +7.3 | mix |
| epistolary | −3.62 | +0.4 | ~tie |
| noir | −3.74 | +8.4 | mix |
| absurdist | −4.49 | −4.9 | spec |

Specialists win at both extremes of complexity; mixture wins in the middle.
Mechanism: extreme personas have the most distinctive surface markers
(fairy_tale's openers / morals; absurdist's structural non-sequiturs) that
specialists can memorize tightly. Middle-complexity personas rely on more
diffuse stylistic features and benefit from cross-persona representation
sharing in the mixture's training set.

Plot: `results/plots/complexity_analysis.png`.

### Open follow-ups

- **Statistical power.** A 750-seq inference set is too small to resolve a
  3% prior tilt. The next run should either expand the inference split to
  1,500/persona (CI half-width scales ~1/√N) or push for more persona
  separation in the data (longer / more stylistically distinct stories).
- **Per-persona-epoch alignment.** Add `spec S ↔ mix 5·S` as a third LoTP
  alignment mode and re-fit π_KL there. Loss-aligned has the specialist at
  2× the per-persona exposure of the mixture; per-persona-epoch alignment
  would equalize that.
- **Step alignment past step 160.** Step-aligned pair-ups for mixture steps
  > 160 pin every specialist at its final checkpoint. Trajectory plots in
  this range reflect only the mixture's continued training, not a fair
  comparison.
- **Data-quantity vs. prior-weight study.** Per task_plan open question:
  repeat with a 1,500/persona specialist variant to study how much the
  apparent prior depends on training-data quantity.
- **Test the complexity hypotheses on a new run.** Two open complexity-side
  predictions to confirm: (a) does π_KL really tilt toward harder personas
  with more inference data behind it? (b) is the U-shape (specialists winning
  at extremes of complexity, mixture in the middle) reproducible with a
  different choice of personas?
- **Parameter-norm trajectories** (results/plots/param_norms.png) — relate
  the fitted π to ‖θ − θ_base‖ trajectories per Betley et al.: complex
  personas should be penalized by SFT, observable in norm at matched
  training performance. dev_norm at peak already rank-orders with complexity
  in this run; worth following the trajectory more carefully.

---

## Findings — single-epoch + non-uniform series (induce-prior branch)

Full log: `results/analysis.md` §"Single-Epoch & Non-Uniform-Mixture Experiments".
Data regenerated to ~10k/persona; splits test 500 / val 500 / train ~9k; single-epoch
training (~281 spec / ~1,421 mix steps).

1. **Single-epoch training does not overfit.** Val loss falls monotonically to the
   final checkpoint for all 6 models (no U-shape). Specialists beat the mixture on
   **5/5** personas, reversing the multi-epoch run's "mixture beats spec 3/5". The
   **mixture-as-regularizer effect was an artifact of multi-epoch specialist
   overfitting**, exactly the hypothesis. (`overfit_curves.png`, `logprob_overfit.png`.)

2. **The LoTP `π_KL` estimator does not measure the induced prior.** Trained on a
   strongly non-uniform mixture (geometric ratio 1.5), recovered `π_KL` was
   bit-identical to the uniform case. `fit_pi_kl` never references `P_mix`; it
   recovers the **eval-set composition** (matched to 0.0000 on uniform/geom/reversed
   eval subsets). The first run's "uniform prior" was an artifact of (estimator) ×
   (uniform eval set). (`pi_vs_proportions.png`, `eval_composition_probe.png`.)

3. **The induced prior is recoverable via the per-persona gap**
   `gap_p = mean(log P_mix − log P_spec_p)`. It tracks the non-uniform data
   proportion at **r = +0.92** (the mixture *did* internalize the prior); under
   uniform data the gap spread is the base-model prior/complexity signature
   (absurdist, the hardest persona, learns worst for its data share). A normalized
   `π` is unrecoverable because gaps (18-90 nats) ≫ `log π`: the trained mixture is
   not a convex combination of specialists, so `P_mix ≈ Σπ_i P_i` fails under
   near-disjoint specialist support. (`induced_prior.py`, `induced_prior.png`.)

4. **Two induced priors, and they disagree.** Sampling from the mixture and
   classifying the samples under the specialists (`generate_and_classify.py`, the
   *correct* use of the KL fit: feed it mixture samples, not the eval set) is
   validated — argmax≈KL (L1 0.007), classifier 100% on real text, rule-based
   markers agree. The **generated** prior is base-model-dominated and does NOT track
   the training data (r = −0.27): the mixture over-generates causal/sci narration and
   almost never produces the templated formats (epistolary letters 0.3%, fairy
   "Once upon a time" 0.0%) regardless of training share. So: **conditional
   competence** (the gap, Result 3) tracks the data prior (r=+0.92), while the
   **free-generation marginal** is base-prior-dominated. Data composition controls
   how well each persona is *modeled*, not what the model *generates*. (`generated_prior.png`.)

**Method corrections:** hull-escape sign fixed (escape = `P_mix > max_i P_i`);
loss-alignment uses min-val (not the test set); `π_KL` reinterpreted as eval-set
composition, replaced by (a) the gap-based induced-prior estimator and (b) the
generation-based prior (sample the mixture, classify).

**Next experiment:** vary a persona's data share and its complexity independently
(2-D design) to separate the data prior from the base-model prior signature; and
probe why free generation collapses to the base register (e.g. conditional vs
unconditional sampling, temperature).

