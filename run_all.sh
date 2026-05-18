#!/usr/bin/env bash
# Build splits (if missing) and run LoRA finetuning on each per-theme dataset
# and on the full mixture, all evaluated against the same uniform holdout.
#
# Override defaults via env vars, e.g.:
#   MODEL=SimpleStories/SimpleStories-11M EPOCHS=2 ./run_all.sh

set -euo pipefail

MODEL="${MODEL:-SimpleStories/SimpleStories-V2-1.25M}"
DATA_DIR="${DATA_DIR:-data}"
RUNS_DIR="${RUNS_DIR:-runs}"
# Match data exposure (epochs) across runs: each per-theme dataset is ~1/N_THEMES
# the size of the mixture, so personas get MAX_STEPS/N_THEMES to see each example
# the same number of times the mixture does.
MAX_STEPS="${MAX_STEPS:-700}"
N_THEMES="${N_THEMES:-6}"
PERSONA_MAX_STEPS="${PERSONA_MAX_STEPS:-$((MAX_STEPS / N_THEMES))}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
EVAL_STEPS="${EVAL_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

mkdir -p "$RUNS_DIR"

if [ ! -f "$DATA_DIR/holdout.jsonl" ]; then
    echo "==> Building splits into $DATA_DIR/"
    python -c "from personaDataset import build_theme_splits; build_theme_splits('$DATA_DIR')"
else
    echo "==> Reusing existing splits in $DATA_DIR/"
fi

run_one () {
    local train_path="$1"
    local max_steps="$2"
    local name
    name="$(basename "$train_path" .jsonl)"
    local out="$RUNS_DIR/$name"
    mkdir -p "$out"
    # The HoldoutLogprobsCallback appends to this file; clear it so reruns
    # don't mix old + new metrics.
    rm -f "$out/holdout_metrics.jsonl"
    echo
    echo "==> Finetuning on $train_path -> $out (max_steps=$max_steps)"
    python finetune.py \
        --model "$MODEL" \
        --train_dataset "$train_path" \
        --holdout_dataset "$DATA_DIR/holdout.jsonl" \
        --output_dir "$out" \
        --max_steps "$max_steps" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --logging_steps "$LOGGING_STEPS"
}

for f in "$DATA_DIR"/theme_*.jsonl; do
    run_one "$f" "$PERSONA_MAX_STEPS"
done

run_one "$DATA_DIR/mixture.jsonl" "$MAX_STEPS"

echo
echo "==> All runs complete. Per-run metrics: $RUNS_DIR/<run>/holdout_metrics.jsonl"
