#!/usr/bin/env bash
# A/B test: train one persona under two budgets and compare overfitting.
#   long  : MAX_STEPS=700 (matches the original run_all.sh, ~18 epochs over 313 examples)
#   short : MAX_STEPS=117 (matches the mixture's ~3 epochs of data exposure)
#
# Both runs share everything else (LR, LoRA config, batch size, seed) and are
# evaluated on the same uniform holdout, so any difference in the holdout-NLL
# curve isolates the effect of the step budget.
#
# Override via env vars, e.g.:
#   THEME=theme_2_consciousness ./compare_persona_steps.sh
#   LONG_STEPS=1000 SHORT_STEPS=150 ./compare_persona_steps.sh

set -euo pipefail

MODEL="${MODEL:-SimpleStories/SimpleStories-V2-1.25M}"
DATA_DIR="${DATA_DIR:-data}"
RUNS_DIR="${RUNS_DIR:-runs_compare}"
THEME="${THEME:-theme_0_family}"

LONG_STEPS="${LONG_STEPS:-700}"
SHORT_STEPS="${SHORT_STEPS:-117}"

BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
# Frequent evals so the overfitting peak is visible in the short run too.
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000000}"  # effectively disable intermediate checkpoints
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-0}"

TRAIN_PATH="$DATA_DIR/$THEME.jsonl"
HOLDOUT_PATH="$DATA_DIR/holdout.jsonl"

if [ ! -f "$TRAIN_PATH" ]; then
    echo "No training file at $TRAIN_PATH" >&2
    exit 1
fi
if [ ! -f "$HOLDOUT_PATH" ]; then
    echo "No holdout file at $HOLDOUT_PATH" >&2
    exit 1
fi

mkdir -p "$RUNS_DIR"

run_one () {
    local tag="$1"
    local max_steps="$2"
    local out="$RUNS_DIR/${THEME}__${tag}"
    mkdir -p "$out"
    rm -f "$out/holdout_metrics.jsonl"
    echo
    echo "==> [$tag] $THEME, max_steps=$max_steps -> $out"
    python finetune.py \
        --model "$MODEL" \
        --train_dataset "$TRAIN_PATH" \
        --holdout_dataset "$HOLDOUT_PATH" \
        --output_dir "$out" \
        --max_steps "$max_steps" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --logging_steps "$LOGGING_STEPS" \
        --seed "$SEED"
}

run_one "long_${LONG_STEPS}"  "$LONG_STEPS"
run_one "short_${SHORT_STEPS}" "$SHORT_STEPS"

echo
echo "==> Done. Compare:"
echo "    $RUNS_DIR/${THEME}__long_${LONG_STEPS}/holdout_metrics.jsonl"
echo "    $RUNS_DIR/${THEME}__short_${SHORT_STEPS}/holdout_metrics.jsonl"
echo
echo "Quickly plot with:  python plot_runs.py --runs_dir $RUNS_DIR --output_dir plots_compare"
