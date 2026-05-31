"""Data loaders for the persona experiment.

Splits are produced by src/build_splits.py and live at
    data/<persona>/{persona_train,mixture,inference,reserve}.jsonl

This module provides:
    PERSONAS                              list[str], canonical persona order
    load_split(persona, split) -> rows    read one jsonl
    load_mixture_train() -> rows          union of all 5 train.jsonl files (~45,000 rows)
    load_val(persona) -> rows             one persona's val split (single-persona val)
    load_full_mix_val() -> rows           union of all 5 val.jsonl files (~2,500 rows)
    load_test_set() -> rows               union of all 5 test.jsonl files (~2,500 rows)
    tokenize_story(text, tokenizer)       canonical tokenization (no BOS, append EOS)
    StoryDataset                          PyTorch Dataset for causal LM training
    collate_for_clm                       pad batch + build labels with -100 on padding

Tokenization convention (matches the SimpleStories model card example):
    - add_special_tokens=False     no BOS injected by the tokenizer
    - append EOS (id=1) manually   so the model can learn to terminate
    - truncate to max_length=512   the model's context window

Document any change here in context/notes.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PERSONAS: list[str] = [
    "noir_detective",
    "fairy_tale",
    "scientific_explainer",
    "absurdist",
    "epistolary",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAX_LENGTH = 512  # SimpleStories-V2-5M context window
EOS_TOKEN_ID = 1  # per the SimpleStories-V2-5M model card


def load_split(persona: str, split: str) -> list[dict]:
    """Read one persona-split jsonl file. Fails loudly if missing."""
    path = DATA_DIR / persona / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run src/build_splits.py first.")
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_mixture_train() -> list[dict]:
    """The mixture model's training set: union of every persona's `train.jsonl`.
    This is the SAME per-persona data each specialist trains on (identical train
    split, see task_plan.md), so the only difference is the cross-persona examples."""
    rows: list[dict] = []
    for persona in PERSONAS:
        rows.extend(load_split(persona, "train"))
    return rows


# Non-uniform mixture: geometric (ratio 1.5) proportions over PERSONAS in canonical
# order (noir_detective the most, epistolary the least). Used to test whether the
# recovered LoTP π tracks a non-uniform DATA prior — see task_plan.md.
MIXTURE_EXP_RATIO = 1.5


def mixture_exp_weights() -> dict[str, float]:
    """Normalized geometric (ratio 1.5) proportions over PERSONAS, canonical order."""
    raw = {p: MIXTURE_EXP_RATIO ** (len(PERSONAS) - 1 - i) for i, p in enumerate(PERSONAS)}
    total = sum(raw.values())
    return {p: raw[p] / total for p in PERSONAS}


def load_mixture_train_nonuniform() -> list[dict]:
    """Non-uniform mixture training set. Each persona contributes a geometric
    (ratio 1.5) share, anchored so the largest-weight persona uses its full train
    split. Sub-sampling takes the first n_i rows of the already-shuffled train.jsonl,
    so it is deterministic. Specialists are unchanged (still trained on the full
    train split) — only the mixture's composition is non-uniform."""
    weights = mixture_exp_weights()
    anchor = max(PERSONAS, key=lambda p: weights[p])
    anchor_n = len(load_split(anchor, "train"))
    rows: list[dict] = []
    for p in PERSONAS:
        n_p = round(anchor_n * weights[p] / weights[anchor])
        train = load_split(p, "train")
        if n_p > len(train):
            raise RuntimeError(f"non-uniform mixture needs {n_p} of {p} but only {len(train)} available")
        rows.extend(train[:n_p])
    return rows


def _nonuniform_total() -> int:
    """Total example count of the non-uniform mixture (for size-matching)."""
    weights = mixture_exp_weights()
    anchor = max(PERSONAS, key=lambda p: weights[p])
    anchor_n = len(load_split(anchor, "train"))
    return sum(round(anchor_n * weights[p] / weights[anchor]) for p in PERSONAS)


def load_mixture_train_uniform_matched() -> list[dict]:
    """UNIFORM mixture matched in TOTAL size to the non-uniform mixture, so that a
    comparison between the two isolates composition from total training amount. Each
    persona contributes total_nonuniform / 5 examples (first n of its train split)."""
    n_per = round(_nonuniform_total() / len(PERSONAS))
    rows: list[dict] = []
    for p in PERSONAS:
        train = load_split(p, "train")
        if n_per > len(train):
            raise RuntimeError(f"size-matched uniform needs {n_per} of {p} but only {len(train)} available")
        rows.extend(train[:n_per])
    return rows


def load_val(persona: str) -> list[dict]:
    """One persona's val split — the specialist's single-persona overfitting val."""
    return load_split(persona, "val")


def load_full_mix_val() -> list[dict]:
    """Union of all 5 persona val splits — the mixture's full-mix overfitting val."""
    rows: list[dict] = []
    for persona in PERSONAS:
        rows.extend(load_split(persona, "val"))
    return rows


def load_test_set() -> list[dict]:
    """The combined test set (union of all 5 `test.jsonl`, ~2,500 seqs) used for
    LoTP and per-checkpoint cross-evaluation. Held out from all training."""
    rows: list[dict] = []
    for persona in PERSONAS:
        rows.extend(load_split(persona, "test"))
    return rows


def tokenize_story(text: str, tokenizer) -> list[int]:
    """Tokenize one story: no BOS, append EOS, truncate to MAX_LENGTH.

    Asserts the tokenizer's EOS matches our hardcoded EOS_TOKEN_ID so a future
    base-model swap or tokenizer change fails loudly here rather than silently
    producing the wrong terminator."""
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise RuntimeError(
            f"tokenizer.eos_token_id={tokenizer.eos_token_id} != EOS_TOKEN_ID={EOS_TOKEN_ID}"
        )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = ids[: MAX_LENGTH - 1]  # leave room for EOS
    ids.append(EOS_TOKEN_ID)
    return ids


class StoryDataset(Dataset):
    """Tokenizes lazily so the tokenizer choice isn't baked into the splits."""

    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        ids = tokenize_story(row["story"], self.tokenizer)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "persona": row["persona"],
            "id": row["id"],
        }


def collate_for_clm(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad to the longest sequence in the batch. Labels are input_ids with
    pad positions set to -100 (ignored by CrossEntropyLoss)."""
    lengths = [b["input_ids"].numel() for b in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = 1
        labels[i, :n] = b["input_ids"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
