# TODO — Single-Epoch + Non-Uniform-Mixture Experiments

Design in context/task_plan.md; results in results/analysis.md. All items below done.

- [x] **1. Data generation pipeline.** Vendored SimpleStories generator + 5 persona
  system prompts + persona-controlled `generate_personas.py`; ~10k/persona generated.
- [x] **2. build_splits.py** — local `data_raw/`, 3-way `train`/`val`/`test` split.
- [x] **3. data.py loaders** — train/val/test; `load_full_mix_val`, `load_test_set`,
  `load_mixture_train_nonuniform` (geometric 1.5).
- [x] **4. train.py: single-epoch** — within-epoch checkpointing, dual-val eval,
  `mixture` + `mixture_exp` runs.
- [x] **5. eval_checkpoints.py** — cross-eval on the 2,500-seq test set.
- [x] **6. lotp.py** — min-val loss-alignment; **hull-escape sign fixed**; fits π for
  every mixture run (`run_for_mixture`).
- [x] **7. plot_trajectory.py** — epoch x-axis; `overfit_curves`, `pi_vs_proportions`,
  + updates.
- [x] **8. Non-uniform experiment + estimator** — `induced_prior.py` (gap-based
  induced prior) and `eval_composition_probe.py` (π_KL = eval composition).
- [x] **9. Generation-based induced prior** — `generate_and_classify.py`: sample from
  the mixture, classify under specialists. Validated (argmax≈KL, classifier 100% on
  real text, rule-marker agreement). Reveals the two-priors result.
- [x] **10. Write-up** — results/analysis.md (Results 1-4), design_doc.md findings,
  notes.md, this file. (Commits still pending — deferred by user.)

## Open follow-ups (not started)

- Deterministic re-run with the math-SDP attention backend forced (bit-reproducibility;
  would not change any conclusion).
- 2-D data-share × complexity experiment to separate the data prior from the
  base-model prior signature (see task_plan.md "Next experiment").
- Optionally push the regenerated ~10k/persona dataset to HF.
