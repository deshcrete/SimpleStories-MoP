"""LoRA finetuning of a SimpleStories model on a given dataset.

Usage:
    python finetune.py \
        --model SimpleStories/SimpleStories-30M \
        --train_dataset data/mixture.jsonl \
        --holdout_dataset data/holdout.jsonl \
        --output_dir runs/mixture

The script reads JSONL with {"text": ..., "theme": ...} rows. LoRA is applied
to attention q/k/v projections and the MLP up/down projections. At every
`--eval_steps` the holdout NLL (mean negative logprob per token) is logged
both overall and per theme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]


def load_jsonl(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_rows(rows: List[dict], tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_list([{"text": r["text"]} for r in rows])

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(_tok, batched=True, remove_columns=ds.column_names)


class HoldoutLogprobsCallback(TrainerCallback):
    """At each eval step, compute mean per-token NLL on the holdout, overall and per theme.

    Appends one JSON line per eval to `<metrics_path>`, e.g.
        {"step": 200, "holdout/nll_overall": 1.23, "holdout/nll[fantasy]": 1.45, ...}
    """

    def __init__(
        self,
        tokenizer,
        holdout_rows: List[dict],
        max_length: int,
        batch_size: int,
        metrics_path: str | Path,
    ):
        self.tokenizer = tokenizer
        self.holdout_rows = holdout_rows
        self.max_length = max_length
        self.batch_size = batch_size
        self.themes = sorted({r["theme"] for r in holdout_rows})
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _nll(self, model, rows: List[dict]) -> float:
        device = next(model.parameters()).device
        total_loss, total_tokens = 0.0, 0
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i : i + self.batch_size]
            enc = self.tokenizer(
                [r["text"] for r in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            n_valid = int((labels != -100).sum().item())
            total_loss += float(out.loss.item()) * n_valid
            total_tokens += n_valid
        return total_loss / max(total_tokens, 1)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        was_training = model.training
        model.eval()
        metrics = {"holdout/nll_overall": self._nll(model, self.holdout_rows)}
        for theme in self.themes:
            rows = [r for r in self.holdout_rows if r["theme"] == theme]
            metrics[f"holdout/nll[{theme}]"] = self._nll(model, rows)
        if was_training:
            model.train()
        record = {"step": state.global_step, "epoch": state.epoch, **metrics}
        print(f"[step {state.global_step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if state.log_history:
            state.log_history[-1].update(metrics)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="SimpleStories/SimpleStories-30M",
                   help="HF repo id of the base SimpleStories model.")
    p.add_argument("--train_dataset", required=True, help="JSONL with {text, theme}.")
    p.add_argument("--holdout_dataset", required=True, help="JSONL with {text, theme}.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If >0, total optimizer steps. Overrides --epochs.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = load_jsonl(args.train_dataset)
    holdout_rows = load_jsonl(args.holdout_dataset)
    train_ds = tokenize_rows(train_rows, tokenizer, args.max_length)
    holdout_ds = tokenize_rows(holdout_rows, tokenizer, args.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=holdout_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[
            HoldoutLogprobsCallback(
                tokenizer=tokenizer,
                holdout_rows=holdout_rows,
                max_length=args.max_length,
                batch_size=args.batch_size,
                metrics_path=Path(args.output_dir) / "holdout_metrics.jsonl",
            )
        ],
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
