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
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-4}"
EVAL_STEPS="${EVAL_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-200}"
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
    local name
    name="$(basename "$train_path" .jsonl)"
    local out="$RUNS_DIR/$name"
    echo
    echo "==> Finetuning on $train_path -> $out"
    python finetune.py \
        --model "$MODEL" \
        --train_dataset "$train_path" \
        --holdout_dataset "$DATA_DIR/holdout.jsonl" \
        --output_dir "$out" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --logging_steps "$LOGGING_STEPS"
}

for f in "$DATA_DIR"/theme_*.jsonl; do
    run_one "$f"
done

run_one "$DATA_DIR/mixture.jsonl"

echo
echo "==> All runs complete. Per-run metrics: $RUNS_DIR/<run>/holdout_metrics.jsonl"
