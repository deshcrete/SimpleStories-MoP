"""Base model loading and checkpoint helpers for the persona experiment.

Base model: SimpleStories/SimpleStories-V2-5M (LlamaForCausalLM, 5.35M params,
6 layers / 256 hidden / 4 heads, vocab 4019, context 512, EOS=1).

The tokenizer ships without a pad token; we set pad = EOS for batching. We never
attend to or score the pad positions (the collate function in data.py masks them
out), so this re-use does not affect loss or log-likelihoods.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "SimpleStories/SimpleStories-V2-5M"


def load_base_model_and_tokenizer(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    # Keep model config in sync with tokenizer so HF doesn't emit attention-mask
    # warnings and any generation path picks up the right pad id.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    return model, tokenizer


def save_checkpoint(model, out_dir: Path) -> None:
    """Save just the model weights. Optimizer state is not preserved — we train
    each run start-to-finish and never resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)


def load_checkpoint(checkpoint_dir: Path, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.to(device)
    return model


def param_vector(model) -> torch.Tensor:
    """Flat 1-D tensor of all model parameters (cpu, fp32). Used for norms."""
    return torch.cat([p.detach().to("cpu", torch.float32).flatten() for p in model.parameters()])


def param_norm(model) -> float:
    return float(param_vector(model).norm(p=2).item())


def deviation_norm(model, base_params: torch.Tensor) -> float:
    """‖θ − θ_base‖₂ in the flattened-parameter sense."""
    return float((param_vector(model) - base_params).norm(p=2).item())
