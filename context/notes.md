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
- We **append EOS (id=1) manually** to every story before training so the model can
  learn to terminate. Stories that hit the 512-token cap reserve room (511 tokens of
  content + 1 EOS).
- Log-probabilities are computed in `eval_checkpoints.py` over every label position
  that isn't `-100`. With `add_special_tokens=False` + appended EOS, this means:
  - log P sums over `T - 1` next-token predictions for a story of `T` real tokens
    (since position 0 has no predecessor to be scored against). Consistent across
    all models — does not affect LoTP fit.

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

- `transformers` is pinned to `4.46.3` in `setup/requirements.txt`. The installed
  torch is 2.4.1+cu124; transformers ≥5.x requires `torch.distributed.tensor.device_mesh`
  which only exists from torch 2.5+. 4.46.3 fully supports Llama-arch models and
  works with torch 2.4.
- The HF pool is actually **3,800 stories per persona** (19,000 total across the
  single `train` split), not the ~2,150 the dataset card suggested. Our splits
  still take 500/500/150 per persona; reserve grows to ~2,650.

## Open items not yet implemented

- **Plots.** No plotting code yet; the metrics JSONLs / npys are designed to be
  consumed by a downstream script. Will add when first plots are requested.
- **LR sweep.** Currently a single LR (5e-4). Task plan keeps a sweep open.
- **Stratified per-topic eval.** All persona/theme/topic columns are preserved in
  the JSONL, so we can stratify retrospectively without re-running training.
