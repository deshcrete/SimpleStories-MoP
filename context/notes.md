# Notes — design decisions and in-flight discoveries

Records every non-trivial call made while implementing `task_plan.md`. When a
decision becomes encoded in a code comment, it can be compressed out of here.

## Base model

- **Model id:** `SimpleStories/SimpleStories-V2-5M`
- **Architecture:** LlamaForCausalLM. 6 layers, hidden 256, 4 heads, ~5.35M params.
- **Tokenizer:** `AutoTokenizer.from_pretrained(...)`. Vocab 4019. EOS token id = 1.
  No pad token by default — we set `tokenizer.pad_token_id = eos_token_id` for batching
  and mask pad positions in labels (`-100`), so loss/log-likelihood is never computed
  on pad.
- **Context window:** 512 tokens. We truncate stories to this length.

## Tokenization convention

- `add_special_tokens=False` — matches the model card's example and avoids injecting
  an unexpected BOS at finetune time (the base model was not trained with one).
- **EOS ... EOS scheme (updated 2026-06-24, clustered base-model experiment).** Every
  story is tokenized as `[EOS] + content + [EOS]` (EOS id=1). Rationale: the base
  model was pretrained on stories concatenated and *separated* by EOS, and generation
  is seeded with a leading EOS as the "start a new story" token (see
  generate_and_classify.py / "TWO induced priors" note). Training as `EOS ... EOS`
  matches that format on both ends.
  - The **leading EOS** is pure conditioning context: under the CLM shift position 0
    is never a prediction target, so it is never scored.
  - The **trailing EOS** is the terminator the model learns to emit.
  - Stories at the 512 cap reserve room for both (510 content tokens + 2 EOS).
  - *Prior runs (induce-prior branch) used append-only EOS* (`content + [EOS]`,
    511 content + 1 EOS). The clustered experiment changed `tokenize_story` to the
    EOS ... EOS form; re-running the old experiment would now differ by the leading
    EOS. Both are internally consistent across the 6/4 co-trained models.
- Log-probabilities are computed in `eval_checkpoints.py` over every label position
  that isn't `-100`. With `add_special_tokens=False` + `EOS ... EOS`, this means:
  - log P sums over `T - 1` next-token predictions for a story of `T` total tokens
    (position 0, the leading EOS, has no predecessor to be scored against). The first
    real token is now scored (predicted from the leading EOS) and the trailing EOS is
    scored too. Consistent across all models — does not affect cross-model comparisons.

## Split sizes (per persona, from concatenated HF pool)

- 500 persona-train, 500 mixture-contribution, 150 inference, ~1000 reserve.
- Single fixed seed (`42`) shared across personas; rng state advances through the
  persona list deterministically.
- **Dataset structure (verified on first run, contra the WebFetch description):**
  `desh2806/simplestories-personas` has a single `default` config and stores
  the persona as a *column*, not as separate configs/splits. `build_splits.py`
  therefore: concatenates every HF split → sorts each persona's rows by `id` →
  shuffles with the seeded rng → slices. Sorting by `id` before shuffling
  guarantees the shuffle is invariant to whatever HF split ordering returns.

## Held-out re-use

`task_plan.md` mentions "persona-own held-out" and "mixture held-out" eval losses.
We use the persona's 150-seq inference split as the held-out for the specialist,
and the full 750-seq combined inference set as the held-out for the mixture.
These are the same data later used for the LoTP fit — no leakage because they
were never in any training set. Single eval data path for both purposes keeps
checkpoint selection and LoTP comparisons consistent.

## Hyperparameters (held constant across all six runs)

- Optimizer: AdamW
- Learning rate: 5e-4 (midpoint of the 3e-4..1e-3 sweep range in task_plan.md;
  fixed for now to keep the six runs comparable, can sweep later if needed)
- Weight decay: 0.0 — the task plan flags this as a regularization confound
- Batch size: 32
- Epochs: 10
- Seed: 42 (numpy, torch, random, data loader generator)
- Mixed precision: OFF. fp32 throughout for reproducibility of log-probabilities.
- Data order: shuffled once at DataLoader construction with a seeded generator;
  not re-shuffled per epoch, so all runs see a fixed permutation of their data.

## Checkpoint cadence

- **Specialists** (500 examples, batch 32 → ~16 steps/epoch → ~160 total steps):
  every 8 steps for the first 80 steps, then every 16 steps. ~15 checkpoints.
- **Mixture** (2,500 examples, batch 32 → ~78 steps/epoch → ~780 total steps):
  every 20 steps for the first 200 steps, then every 50 steps. ~15 checkpoints.
- Always save step-0 (base model) and the final step regardless of cadence.

## LoTP fit

- **Primary fit** (`fit_pi_kl`): minimize forward KL between the empirical
  inference set (treated as samples from P_mix) and the implied mixture
  `M = Σ_i π_i P_i`. Up to a constant, this is equivalent to maximizing
  `mean_j log M(x_j) = mean_j logsumexp_i(log P_i(x_j) + log π_i)` — exactly
  the mixture-model MLE that EM converges to. Solved with L-BFGS-B on
  softmax-parameterized logits (uniform π = softmax(0) is the default init).
- **Diagnostic fit** (`fit_pi_residual`): the spec's
  `L(π) = mean_j (log P_mix - logsumexp(log P_i + log π_i))^2`. Reported but
  not used as primary — see "Magnitude issue" below.
- Both fits report `pi`, `residual_rms`, `avg_loglik`, and a
  `per_persona_residual` breakdown by source persona.
- `lotp_perturb.py` sweeps inits (uniform + 5 corners + 20 Dirichlet) for both
  fits to verify the optima are not flat directions in π-space.

### Magnitude issue (why KL is primary, not the spec's L(π))

Per-sequence log-P magnitudes are ~hundreds of nats and scale roughly with
sequence length (T ≈ 100..400 → log P ≈ −100..−1300 nats per seq, span ~1200
nats across the 750-seq inference set). The squared residual
`(log P_mix - log M)^2` is therefore dominated by the *variance* of the
per-sequence residuals, not the mean alignment.

Concrete diagnostic at the loss-aligned checkpoint:

    lse(log P_i + log(1/5)) − log P_mix:   mean = −1.53   std = 92.95

So under uniform π the LoTP identity already holds *on average* to ~1.5 nats,
but the per-sequence noise has std ~93. Optimizing the squared residual just
shrinks std slightly (92.95 → 92.77) by trading off mass across personas in
ways that exploit noise — giving the degenerate-looking
`π_residual = [0.40, 0.00, 0.48, 0.01, 0.11]` we observed.

KL is invariant to per-sequence additive log-P offsets, so it isn't fooled by
this noise and recovers `π_KL ≈ [0.200, 0.197, 0.203, 0.200, 0.200]` — uniform
with a tiny tilt that's the real base-model prior signature.

## Alignment

- **Step-aligned:** for each mixture checkpoint, pair each specialist with its
  *closest* available checkpoint by absolute step count. This is approximate
  because the cadences differ — exact step matching is impossible since the
  specialists stop at step ~160 while the mixture goes to ~780. Trajectory
  plots will therefore be 1:1 only over the first ~160 steps; beyond that,
  specialists are pinned at their last checkpoint.
- **Loss-aligned:** each model at the checkpoint with the highest summed
  log-P over its *own* held-out (per task_plan.md "each model at its own
  min-eval-loss checkpoint"). For specialists this is the persona's 150-seq
  inference subset; for mixture it's the full 750. Earlier version of this
  code used total_logp over the full 750 for everyone, which produced a
  degenerate π because three specialists picked step 0 (untrained base
  model) — see `loss_aligned_perturb.json` for the diagnostic.

## Convex-hull-escape

- Default margin `τ = 0`. Sequence `j` is an escape iff
  `max_i log P_specialist_i(x_j) > log P_mixture(x_j)`.
- The check is run at every (step- or loss-aligned) checkpoint pair.

## Storage layout

```
data/
  <persona>/persona_train.jsonl
  <persona>/mixture.jsonl
  <persona>/inference.jsonl
  <persona>/reserve.jsonl
  manifest.json
results/
  checkpoints/<run>/step_<N>/        # HF save_pretrained
  checkpoints/<run>/train_log.jsonl
  checkpoints/<run>/run_config.json
  metrics/<run>/per_sequence_logp.npy   # shape (n_checkpoints, 750)
  metrics/<run>/steps.npy
  metrics/<run>/checkpoints.jsonl
  lotp/step_aligned.jsonl
  lotp/loss_aligned.json
```

## Audit fixes (post-first-run)

After the first end-to-end run, two parallel audit passes flagged a set of
issues. Fixes applied:

- **`min_loss_step` → `min_loss_step_own_held_out`** in `lotp.py`. See loss-
  alignment note above.
- **EM was optimizing the wrong objective.** Old code used mixture-likelihood
  EM (`π = mean_j responsibility`) but the spec defines L(π) as the squared
  log-space residual against `log P_mix`. Replaced with `fit_pi_residual`
  (L-BFGS-B on softmax-parameterized logits) as the primary fit; kept EM as
  a diagnostic.
- **CUDA determinism flags now set** in `set_seed`: `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **`eval_loss` token count** was using un-shifted label count (n per seq) to
  weight HF's already-shifted mean loss (n-1 scored positions per seq).
  Logged eval_loss values in the first run are biased by a constant factor
  ≈ T/(T-1); the residual ranking of checkpoints is unchanged. Future runs
  will log the right number.
- **Silent fallbacks removed**: `git_sha()` now raises instead of returning
  `"unknown"`; `eval_loss` raises if no tokens were scored instead of dividing
  by 1.
- **`tokenize_story` asserts** `tokenizer.eos_token_id == 1` so a base-model
  swap fails loudly here rather than silently appending the wrong terminator.
- **`model.config.pad_token_id`** is now set in tandem with the tokenizer's
  pad id (HF was emitting attention-mask warnings; functionally harmless
  because labels mask pad with -100, but cleaner).

Not applied (intentionally):

- **Length-matching:** the spec offers two options — (a) truncate to a fixed
  T, or (b) report per-persona log-likelihoods separately and only aggregate
  in the LoTP residual. We do (b): per-persona aggregates are in
  `checkpoints.jsonl`, and `lotp.py` now also dumps `per_persona_residual`
  so we can see if epistolary (longest persona) is dominating the residual.
  Truncation would change training data — flagged as a follow-up if the
  per-persona residual breakdown reveals domination.
- **Step-aligned pinning past specialist's last step:** for mixture
  checkpoints > 160 we pin every specialist to its final step. Acknowledged
  in `lotp.pair_steps_step_aligned` via the `specialist_gap` signed field;
  downstream plots can filter on this.

## Environment quirks

- `transformers` is pinned to `4.46.3` in `setup/requirements.txt`. transformers
  ≥5.x requires `torch.distributed.tensor.device_mesh` (torch 2.5+); 4.46.3 fully
  supports Llama-arch models. It works with both torch 2.4 and torch 2.11 (tested).
- **Blackwell GPU (2026-05-31).** The box has an RTX PRO 4000 Blackwell = compute
  capability **sm_120**. The pinned `torch 2.4.1+cu124` only ships kernels up to
  sm_90 -> every CUDA op raised `no kernel image is available for execution on the
  device`. Fix: `pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch`
  (got `torch 2.11.0+cu128`, arch list includes sm_120). Then **uninstall
  torchvision + torchaudio** — the cu124 builds (0.19.1 / 2.4.1) are pinned to
  torch==2.4.1 and their broken `torchvision::nms` op makes transformers' lazy
  import of `modeling_llama` fail (`operator torchvision::nms does not exist`).
  The project uses neither, so removing them lets transformers see them as absent.
  After this, training/eval run on GPU normally.
- The HF pool is actually **3,800 stories per persona** (19,000 total across the
  single `train` split), not the ~2,150 the dataset card suggested. Our splits
  still take 500/500/150 per persona; reserve grows to ~2,650.

## Open items not yet implemented

- **Plots.** No plotting code yet; the metrics JSONLs / npys are designed to be
  consumed by a downstream script. Will add when first plots are requested.
- **LR sweep.** Currently a single LR (5e-4). Task plan keeps a sweep open.
- **Stratified per-topic eval.** All persona/theme/topic columns are preserved in
  the JSONL, so we can stratify retrospectively without re-running training.

## Single-epoch experiment + data regeneration (2026-05-31)

Goal (user): test whether the persona overfitting seen in the prior run is caused
by **multi-epoch training (data reuse)**. We do NOT run a multi-epoch regime; we
run a single **single-epoch, lots-of-data** experiment and contrast it against the
already-documented multi-epoch findings. If single-epoch (with comparable/greater
gradient steps) does not overfit, that pins reuse as the cause.

- **x-axis = epochs (fractional 0->1).** On the epoch axis specialist and mixture
  have identical per-persona exposure at every point (each persona's stories seen
  e times in both), so it removes the step-count confound the prior run flagged.
- **Two val sets for the overfitting diagnostic:** each specialist on its own
  single-persona val split; mixture on the full-mix val split (union of the 5).
  Overfitting = train down while val up. Plus the existing 150/persona inference
  set kept as the LoTP/hull-escape test set.
- **More data needed.** Single-epoch @ 3,300/persona gives only ~103 specialist
  steps (< the prior run's 160) -> "undertrained" confound. Target ~10k/persona so
  single-epoch reaches >=160 (ideally ~280) specialist steps. Hence regeneration.

### Data regeneration via vendored SimpleStories pipeline

- Upstream `simple_stories_generate` cloned + **vendored** (nested .git removed;
  provenance in `simple_stories_generate/UPSTREAM.md`, commit bf306bd). The
  `lennart-finke` and `simple-stories` org repos are the identical commit.
- `text_data.py` LANGUAGE flipped "ja" -> "en".
- **Persona is the controlled variable** (upstream samples it randomly ~1/3 of the
  time as one of many axes; we fix it per batch). 5 elaborated system prompts in
  `persona_system_prompts.json`, reverse-engineered from the actual stories in
  `desh2806/simplestories-personas` (noir first-person/rain/diners; fairy_tale
  "Once upon a time"+explicit Moral; explainer "The reason is that..."; absurdist
  talking objects/unanswered questions; epistolary Dear/Yours letters).
- `generate_personas.py` reuses the upstream USER-prompt template (very basic words,
  same approved names, multi-story-per-completion separation) **minus** the
  style/grammar/persona clauses — so output columns match the dataset exactly:
  `id, persona, story, model, theme, topic, feature, initial_letter,
  initial_word_type, num_paragraphs`. Model gpt-4o-mini (matches original).
  theme/topic/feature randomized within persona for diversity. The original used
  no `style`/`grammar` columns (dropped), and stored no readability scores — we
  match (no textstat). Key read from env at launch (OPENAI_API_KEY_SIMPLESTORIES
  or OPENAI_API_KEY); fails loudly if absent.
- Generated data is gitignored (`simple_stories_generate/data_raw/`).

### Smoke-test tuning (got persona adherence ~63% -> ~100%)

Validated on small batches before the full run. Three issues found + fixed:

1. **num_paragraphs 1-9 -> 2-5.** Original dataset only uses 2-5 paragraphs
   (verified). Long (9p) stories drifted out of voice (noir -> third-person fable).
2. **stories-per-completion 30 -> 10** (`MAX_STORIES_PER_COMPLETION`). Packing many
   stories in one call let the voice slip to a generic children's story by story 3-4.
3. **In-user-prompt persona reinforcement.** The system message alone is too weak:
   the content task ("simple stories about <theme>") dominates. We now restate the
   persona prompt inside the user message as a hard "EVERY story must be in this
   voice" constraint. This was the biggest lever (noir 63% -> 100% first-person).

Two follow-ups for fixed-opener personas:

4. **initial_letter rule defers to persona opener.** The SimpleStories "start with a
   word beginning with letter X" axis fought fairy_tale's "Once upon a time" /
   epistolary's "Dear ___". Made it explicitly yield -> fairy opener 83% -> 100%.
5. **Strip leading/trailing `---`.** GPT emits markdown-rule separators around
   stories (also present in the *original* dataset); `parse_stories` now strips
   them. Fixed a false 14% "non-conforming" epistolary rate (the content was a
   perfect letter exchange behind a leading `---`).

Final smoke-test adherence (40-56 stories/persona): noir 100% first-person+atmosphere,
fairy 100% opener+moral, sci 100% cause-effect, absurdist 100% non-sequitur,
epistolary 100% Dear+full-exchange.

## CENTRAL METHODOLOGY FINDING: π_KL ≠ induced prior (2026-05-31)

The biggest discovery of the non-uniform experiment, because it reframes the whole
project. Detailed in `results/analysis.md`; the one-paragraph version:

- **`lotp.fit_pi_kl` recovers the EVAL-SET composition, not the mixture's induced
  prior.** It never references `P_mix`; it maximizes test-sample likelihood under
  `Σ_i π_i P_i`, and since each specialist is ~100 nats better on its own persona
  (near-disjoint support), the matched specialist dominates the logsumexp → the MLE
  just counts personas in the eval set. Proven: π_KL is bit-identical across a
  uniform and a geometric-1.5 mixture, and `eval_composition_probe.py` reproduces
  any eval composition to 0.0000.
- **The first run's "uniform induced prior" was therefore an artifact** of the
  uniform 750-seq eval set, not a measurement.
- **The induced prior IS measurable via the per-persona gap**
  `gap_p = mean(log P_mix − log P_spec_p)` (`induced_prior.py`): tracks the
  non-uniform data proportion at r=+0.92; under uniform data its spread is the
  base-prior/complexity signature.
- **Root cause** (why no π works): gaps are 18-90 nats ≫ `log π`, so the trained
  mixture is NOT a convex combination of specialists — `P_mix ≈ Σπ_i P_i` is false
  under near-disjoint support. The LoTP linear-system framing is ill-posed here.
- **Hull-escape sign was inverted** in the inherited code (counted the inequality
  holding, not the violation). Fixed in `lotp.hull_escape_count`; the first run's
  hull narrative is inverted.

## TWO induced priors (generation vs conditional fit) — 2026-05-31

Follow-on to the methodology finding. Sampling from the mixture and classifying the
samples (`generate_and_classify.py`) is the *correct* use of the KL fit — feed it
mixture samples, not the eval set — and it recovers `P_mix(persona)`, the generation
marginal. Validated: argmax≈KL (L1 0.007); the specialist argmax classifier is 100%
on the real 2,500-seq test set; classifier-independent rule markers agree
(because 52%, "Dear" 0.3%, "Once upon a time" 0.0%).

Key result: there are **two distinct "induced priors" and they disagree**:
- **Conditional competence** = the per-persona gap (`induced_prior.py`): tracks the
  data proportion, r=+0.92. How well each persona's text is *modeled*.
- **Free-generation marginal** = sample-and-classify (`generate_and_classify.py`):
  base-model-dominated, r=−0.27 with the data. The mixture over-generates causal/sci
  narration and almost never produces templated formats (letters, the fairy opener)
  regardless of training share. What the model *generates* unconditionally.

Seeding: the SimpleStories model has NO bos; generation is seeded with EOS (id=1),
which the model treats as "start a new story" — produces clean persona-distinct text.

## Single-epoch experiment outcome

- No model overfits in a single epoch (val monotonic to the end); min-val ≈ final
  for all 6. Specialists beat the mixture 5/5 → mixture-as-regularizer was a
  multi-epoch overfitting artifact.
- Non-uniform mixture (`mixture_exp`, geom 1.5) reuses the 5 specialists; lotp.py now
  fits π for any mixture run (`run_for_mixture`), writing `loss_aligned{,_exp}.json`.
