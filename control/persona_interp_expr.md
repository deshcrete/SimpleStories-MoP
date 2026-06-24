# Understanding the SimpleStories Model as a Mixture of Personas

# Background

Pretrained LLMs can be thought of as a mixture of several personas. When they generate text, we can think of them as producing text in the style of these personas. Previously, we have engineerd a mixture of personas model using synthetic personas and then trained it. However, we can try to do the same persona analysis on an already trained model which we have no idea about which personas it decomposes into and the mixtures of the different personas. A good model to do this with is the SimpleStories models as we have full access to the pretraining dataset. The pretraining dataset has various "features" for each completion. We can group features into several clusters and treat those clusters as our personas. We can then use these sets of completions to train persona models like we did in the previous experiments and then do analysis treating the pretrained SimpleStories as our mixture model.Another framing we can use here is that each of the features make up a "bundle" and personas are bundles of features.

# Methods

In order to operationalize this, we can take the simple stories dataset (http://huggingface.co/datasets/SimpleStories/SimpleStories) on huggingface and then run k-means on the encodings of the completions. Once we have the means, we can take a bunch of examples around the means and then take the features like "theme" and "topic" associated with each of the means and then group those into personas. We can then split the dataset on the features and get various persona datasets. Once we have these persona datasets then train persona models on them using single epoch training like the same previously and then do the downstream analysis

---

# Operationalization (proposed)

This section turns the Background/Methods sketch above into a concrete pipeline. It
reuses the existing single-epoch machinery (`build_splits.py`, `data.py`, `train.py`,
`eval_checkpoints.py`, `lotp.py`, `induced_prior.py`, `generate_and_classify.py`) and
only changes what sits *upstream* of training (persona discovery) and how the
"mixture" is obtained (it is no longer trained — it is the pretrained model).

## What the dataset actually looks like (explored 2026-06-24)

`SimpleStories/SimpleStories`, single `default` config:

- **2,115,696 train rows** + 21,371 test rows. Median 255 words/story (max ~790), so
  every story fits the model's 512-token context after tokenization (a few long ones
  truncate, as in prior runs).
- Each row is a `story` plus orthogonal generation-axis metadata. Approximate
  cardinalities and shape over a 30k sample:
  - `topic` — 48 values, ~uniform (e.g. *hidden treasures, pirates, time travel*).
  - `theme` — 63 values, ~uniform (e.g. *Deception, Magic, Romance, Hardship*).
  - `feature` — 26 values, ~uniform (e.g. *a flashback, a cliffhanger, symbolism,
    an unreliable narrator, a moral lesson*).
  - `style` — 23 values, ~uniform (e.g. *noir, fable-like, philosophical, surreal*).
  - `grammar` — 32 values but **~50% empty**; `persona` — 24 values but **~75% empty**
    (this pre-existing `persona` column is sparse and is NOT what we use here).
  - `initial_word_type` (4), `initial_letter`, and numeric readability columns.

**Key discovery that shapes the design.** The metadata axes are near-uniform and
**mutually independent** — they were randomized independently at generation time. So:

1. No single metadata column is a "persona": a persona in the Background sense (a
   recognizable voice/bundle) is a *conjunction* of axis values, not one `theme` or
   one `topic`.
2. Splitting naively "on the features" (one column) would just produce 26 roughly
   interchangeable slices, not stylistically distinct personas. The clustering step is
   therefore load-bearing: it must discover the *bundles* (correlated combinations of
   axis values), and metadata is used to **name and interpret** the clusters. (This is
   a divergence from the one-line "split the dataset on the features" in Methods;
   flagged here as an in-flight design correction.)
3. **Bundle structure is weak in raw metadata.** Checked directly: feature values lift
   only ~1.2–1.3× within a given style (e.g. within `style=noir`, the top feature
   *a non-linear timeline* is only 1.26× its global rate), and stylometric numerics
   separate styles only modestly (avg sentence length ~10.5–13.5, Flesch ~90–94 across
   the 23 styles). So the axes are close to independent — k-means on raw one-hot
   metadata has thin signal. The single strongest *model-free* persona axis is the
   `style` column itself (23 named voices: noir, fable-like, philosophical, surreal,
   …), which already reads as a persona label.

## The reframing: the pretrained model IS the mixture

In the prior experiments the mixture was a model **we trained** on the union of
persona splits, matched example-for-example to the specialists. Here the mixture is
the **already-pretrained `SimpleStories-V2-5M`** (same base used in prior runs;
swap to a larger V2 checkpoint later if we want a more capable mixture). It was
pretrained on the full ~2.1M-story dataset, which *is* the union of all clusters — so
it plays the mixture role by construction, and we never train it.

Consequence: the data prior is no longer a knob we set. It is the **empirical cluster
frequency in the pretraining data**, which we can measure exactly because we hold the
dataset. The scientific question becomes: *does the pretrained model's per-persona
competence / generation mass track the cluster frequencies in its own pretraining
data, or does it deviate?* Deviation is the base-model (architecture + optimization)
prior signature — measured here against a known data prior rather than an imposed one.

## Stage 1 — Persona discovery (model-independent)

**Circularity constraint (load-bearing).** We must *not* discover the personas using
the model we are about to decompose. If clusters are defined in the target model's own
representation space, then "how the model distributes mass over personas" is partly
circular — the model's geometry was baked into the persona definitions. So persona
discovery uses only the data: the ground-truth metadata the generation process
attached to each story, and/or a text representation that is independent of any
SimpleStories model. The target model is touched for the first time at the
cross-evaluation stage, never to define what a persona is.

New script: `src/cluster_personas.py`. Two model-free routes; we pick by which yields
recognizable, balanced personas (decision recorded in `notes.md`):

**Route A — a persona is a bundle of similar styles (recommended primary; zero neural
model).** The dataset ships 23 named `style` voices, assigned independently of any
model. A persona is a *bundle of similar styles*; the core problem is therefore
**how to measure style similarity** — and the answer must isolate *stylistic*
similarity, not topical overlap.

*Why this is clean here.* Because `topic`/`theme` are randomized independently of
`style`, every style covers all topics roughly equally. So content/topic carries no
signal to separate styles — any measurable style-to-style difference is stylistic by
construction. This lets us use classic **content-independent stylometry** and trust
that the resulting similarity is voice, not subject matter.

*Style-similarity procedure (validated on an 80k sample, see findings below).*
1. **Profile each style** by a content-independent feature vector, averaged over its
   stories: function-word rates (pronouns `i/we/you/he/she/they`, articles,
   prepositions, conjunctions, auxiliaries — the classic authorship-attribution
   signal, topic-neutral by design), punctuation/dialogue rates (`! ? " —`, said/asked),
   and structural numerics (`avg_sentence_length`, `flesch_reading_ease`,
   `num_paragraphs`). Deliberately **exclude content words** so similarity is stylistic.
2. **z-score each dimension across the 23 styles** and form one vector per style.
3. **Cluster the styles** (hierarchical / agglomerative on the style-style distance
   matrix), cut the dendrogram to **~5–6 bundles**. Each bundle = one persona.
4. **Persona = metadata predicate** `style ∈ {bundle}`, so the split is a pure,
   reproducible label filter and the data prior is the exact label frequency.

*Findings from the 80k probe (nearest-neighbour style structure):* the bundles are
sharp and interpretable — **light/comic** {humorous, playful, lighthearted, whimsical},
**mythic/lyrical** {fable-like, fairy tale-like, mythological, mystical, lyric,
surreal}, **dark/somber** {noir, suspenseful, melancholic, tragic}, **plain-narrative**
{classic, heartwarming, adventurous, epic, action-packed, modern, minimalist}, with
{philosophical, romantic} on the margins (attach to lyric/epic). These map onto the
prior synthetic personas (noir→dark; fable/fairy→mythic) and are recovered using only
function words + structure, i.e. with no neural model at all. (`cluster_personas.py`
will reproduce this and let us tune `k` and the cut threshold.)

**Route B — external-encoder clustering (preserves the "cluster on encodings" method).**
Embed a sampled ~100–200k stories with an **off-the-shelf sentence encoder**
(`all-MiniLM-L6-v2`) — independent of the SimpleStories model — L2-normalize, k-means
over `k ∈ {4..10}`, pick `k` by silhouette + elbow (expect `k≈5–6`), seed 42. This
keeps the data-driven discovery flavor while breaking circularity with the *target*
model. Caveat to state in the write-up: an external encoder injects *its own* prior
(MiniLM's geometry), so it is model-independent of the decomposed model but not
representation-free; raw n-gram TF-IDF is the more conservative fallback if we want to
avoid any neural encoder, at the cost of clustering more by topic than by voice.

**Characterize / name each persona (both routes).** For each persona, compute the
**over-representation** (persona frequency ÷ global frequency) of each metadata value
across `theme`, `topic`, `feature`, `style`. Emit `personas.json` per persona: id,
size, defining predicate (Route A) or centroid (Route B), top-lifted metadata, and 5
example stories for eyeballing.

**Quality gate before spending training compute.** Require personas to be
(a) reasonably balanced (no persona <5% or >40% of the pool) and (b) human-recognizable
from their signature + examples. Record the chosen route, `k`, and the mapping in
`notes.md`.

**Persona assignment for the split.** Persist the assignment as a new `persona_label`
column so the rest of the pipeline is label-based and reproducible — `build_splits.py`
already slices by a persona label, so this drops in. For Route A the label is the
metadata predicate (exact); for Route B it is the nearest-centroid id, with the
metadata signature stored alongside for interpretability.

**Optional, clearly non-circular downstream check (not part of discovery).** *After*
personas are fixed by the model-free procedure, we may ask whether they correspond to
structure in the target model's representation space (e.g. are the model-free personas
linearly separable in its hidden states?). Because the personas were defined without
the model, this is a legitimate finding about the model, not a circular definition.

## Stage 2 — Per-persona splits (reuse `build_splits.py`, extended)

Per discovered persona, mirror the single-epoch experiment's 3-way split:

- `test.jsonl` — 500/persona → combined `k × 500`-seq LoTP / hull-escape / gap test
  set, cross-evaluated under every model.
- `val.jsonl` — 500/persona → overfitting diagnostic (own-persona val for specialists;
  union = full-mix val).
- `train.jsonl` — the remainder of the cluster (cap to a shared budget across personas
  so specialists are exposure-matched, e.g. min cluster train size, target ≥9k like
  the prior single-epoch run; log `steps_per_epoch`).

Seed 42, sort-by-`id`-then-shuffle (existing convention) so splits are deterministic.

## Stage 3 — Specialists = single-epoch fine-tunes FROM the pretrained model

This is the second load-bearing decision. The specialists are produced by
**fine-tuning the pretrained `SimpleStories-V2-5M` on each persona cluster** for a
single epoch with shared hyperparameters (AdamW, lr 5e-4, batch 32, wd 0, fp32,
seed 42) — exactly `train.py`'s single-epoch path, just with the persona-cluster
train sets and `model_init = the pretrained checkpoint` instead of a fresh init.

Rationale and what it buys us:

- The mixture (pretrained model) and every specialist **share the same starting
  representations**, so `P_specialist_i` is "the mixture sharpened toward persona i."
  `gap_i = mean(log P_mix − log P_spec_i)` on persona i's own held-out then measures
  the **latent headroom** for persona i: how much better the model fits persona i once
  freed to specialize. Small headroom ⇒ persona already well represented ⇒ high
  base-model prior mass; large headroom ⇒ under-represented.
- It sidesteps the worst confound of this setup: the pretrained mixture saw the whole
  dataset over a full (many-epoch) pretraining run, so a *from-scratch* single-epoch
  specialist would be hopelessly undertrained and the gap would just measure training
  amount. Fine-tuning from the shared checkpoint with matched single-epoch budgets
  removes that asymmetry. **This confound, and this mitigation, must be stated in the
  write-up** — it is the main structural difference from the apples-to-apples prior
  runs, where mixture and specialists were the same kind of object (both fine-tunes).
- Checkpoint cadence as in the single-epoch run (~20 within-epoch + step 0 + final),
  so we keep the trajectory diagnostics (param norm, hull escapes vs fractional epoch).

(Alternative considered: train both a from-scratch mixture and from-scratch
specialists to restore strict apples-to-apples. Rejected for the primary run because
the entire point is to analyze the *given* pretrained model; noted as a possible
secondary control.)

## Stage 4 — Cross-eval and probabilities (reuse `eval_checkpoints.py`)

Compute per-sequence `Σ log p(x_t | x_<t)` (sum, not mean — the LoTP convention) for
the `k × 500`-seq combined test set under:

- the pretrained mixture (one eval, fixed — no trajectory), and
- every specialist checkpoint.

Produces the `(checkpoints, n_test)` log-prob matrices the existing analysis consumes,
plus the per-persona cross-eval breakdown.

## Stage 5 — Analysis (reuse, with the corrected estimators)

Carry forward every method correction from the prior series:

1. **Induced prior = per-persona gap** (`induced_prior.py`), the primary estimator.
   Compare the gap vector against the **known cluster frequencies** in the pretraining
   data (the data prior we can now measure exactly). Correlation gap-vs-frequency is
   the headline: does competence track data share, and where does it deviate?
2. **`π_KL` is the eval-set composition, not the prior** — keep
   `eval_composition_probe.py` as the demonstration; do not report `π_KL` as the prior.
3. **Generation marginal** (`generate_and_classify.py`): sample from the pretrained
   mixture (EOS-seeded, no BOS), classify under the specialists. This is the
   free-generation prior; compare it against both the cluster frequencies and the gap
   to see whether the two-priors-disagree result reproduces on real pretrained data.
4. **Hull escapes / param norms / overfit curves** vs fractional epoch, as before
   (with the corrected escape sign: escape = `P_mix > max_i P_i`).

## New scripts / touch-points

- `src/cluster_personas.py` — new (Stage 1: model-free persona discovery —
  metadata/stylometric clustering or external-encoder k-means — characterize, assign
  `persona_label`).
- `src/build_splits.py` — extend to read the `persona_label` assignment instead of a
  fixed persona list.
- `src/train.py` — add a `model_init=pretrained-checkpoint` path for the specialists;
  the mixture is *not* trained (it is the pretrained model, evaluated as-is).
- `src/data.py`, `eval_checkpoints.py`, `lotp.py`, `induced_prior.py`,
  `generate_and_classify.py`, `plot_trajectory.py` — reused; minor wiring for the
  `k`-persona (vs fixed-5) and "mixture = static checkpoint" cases.

## Open decisions to confirm before building

- **Discovery route (model-free, required):** Route A metadata/stylometric clustering
  (recommended — zero neural model, exact data prior) vs Route B external-encoder
  k-means (preserves the "cluster on encodings" method, model-independent of the target
  but injects MiniLM's prior). The target model is never used to define personas.
- **k and persona balance:** start `k≈5–6`, finalize by silhouette + recognizability.
- **Persona definition:** metadata predicate (Route A, exact label filter) vs
  nearest-centroid membership (Route B). Metadata is used for interpretation either way.
- **Specialist init:** fine-tune from the pretrained model (recommended) vs from
  scratch.
- **Mixture model size:** `V2-5M` for comparability vs a larger V2 for a stronger
  mixture.

