"""Upload the FINAL-step checkpoint of each trained run to the Hugging Face Hub.

We publish the final checkpoint (end of the single training epoch) rather than the
min-val checkpoint: in single-epoch training val falls ~monotonically to the end,
so the two coincide to within sampling noise, and "final step" is the cleaner
artifact — a deterministic 1.0-epoch rule with matched per-persona exposure and no
val-based selection (see the alignment discussion in context/analysis notes).

Each model goes to its own repo so it loads directly:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained("<namespace>/<prefix>-noir_detective")
    t = AutoTokenizer.from_pretrained("<namespace>/<prefix>-noir_detective")

Each repo is staged with: the checkpoint (config.json + model.safetensors), the base
model's tokenizer (the runs save only weights, so we add it for a self-contained
repo), and a generated README model card with full provenance.

Auth: requires an HF token. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the env, or
run `huggingface-cli login` first. Fails loudly if no token is found.

Usage:
    python src/upload_models.py                 # uses whoami() namespace, private repos
    python src/upload_models.py --public
    python src/upload_models.py --namespace desh2806 --prefix simplestories-persona
    python src/upload_models.py --dry-run       # stage + print, do not create/push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from data import PERSONAS
from model import BASE_MODEL_ID

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CHECKPOINT_ROOT = RESULTS_DIR / "checkpoints"

RUNS = PERSONAS + ["mixture"]
DATASET_REPO = "desh2806/simplestories-personas-10k"
DEFAULT_PREFIX = "simplestories-persona"


def final_step_dir(run: str) -> tuple[int, Path]:
    """Return (step, dir) of the run's final (largest-step) checkpoint."""
    run_dir = CHECKPOINT_ROOT / run
    if not run_dir.exists():
        raise FileNotFoundError(f"No checkpoints for run '{run}' at {run_dir}")
    steps = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            steps.append((int(d.name.split("_")[1]), d))
    if not steps:
        raise RuntimeError(f"{run_dir}: no step_<N> checkpoints found")
    return max(steps, key=lambda x: x[0])


def run_config(run: str) -> dict:
    return json.loads((CHECKPOINT_ROOT / run / "run_config.json").read_text())


def final_val_losses(run: str, step: int) -> dict:
    """The val losses logged for the final checkpoint, from train_log.jsonl."""
    log_path = CHECKPOINT_ROOT / run / "train_log.jsonl"
    for line in log_path.read_text().splitlines():
        e = json.loads(line)
        if e.get("event") == "checkpoint" and e.get("step") == step:
            return {k: v for k, v in e.items() if k.startswith("val_")}
    raise RuntimeError(f"{log_path}: no checkpoint entry for step {step}")


def model_card(run: str, step: int, cfg: dict, vals: dict) -> str:
    """Generate a README.md model card with full provenance for one run."""
    kind = "non-uniform mixture" if run == "mixture_exp" else (
        "uniform mixture (union of all 5 personas)" if run == "mixture" else f"`{run}` persona specialist")
    val_lines = "\n".join(f"- `{k}`: {v:.4f}" for k, v in vals.items())
    return f"""---
license: mit
base_model: {BASE_MODEL_ID}
datasets:
- {DATASET_REPO}
tags:
- simplestories
- persona
- supervised-fine-tuning
---

# SimpleStories persona model — {run}

Single-epoch SFT of [{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID})
on the **{kind}** from [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}).

Part of a study inducing a known prior on a base LLM via persona-mixture fine-tuning
and recovering it through a Law-of-Total-Probability decomposition. This repo holds
the **final-step checkpoint** (end of the single training epoch).

## Training

| | |
|---|---|
| base model | `{BASE_MODEL_ID}` |
| run | `{run}` ({kind}) |
| epochs | {cfg['num_epochs']} (single epoch — every example seen once) |
| final step | {step} of {cfg['total_steps']} ({cfg['steps_per_epoch']} steps/epoch) |
| train examples | {cfg['n_train']} |
| optimizer | AdamW, lr={cfg['lr']}, weight_decay={cfg['weight_decay']} |
| batch size | {cfg['batch_size']} |
| precision | fp32 |
| seed | {cfg['seed']} |

Validation loss at the final checkpoint (mean cross-entropy / scored token):
{val_lines}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("REPO_ID")
tokenizer = AutoTokenizer.from_pretrained("REPO_ID")

# The base model has no BOS; seed generation with EOS (id=1) to start a new story.
import torch
seed = torch.tensor([[tokenizer.eos_token_id]])
out = model.generate(seed, max_new_tokens=150, do_sample=True, temperature=1.0, top_p=0.95,
                     eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][1:], skip_special_tokens=True))
```

Tokenization convention used in training: `add_special_tokens=False`, EOS (id=1)
appended to every story, truncated to 512 tokens.
"""


def stage_repo(run: str, ckpt_dir: Path, tokenizer, repo_id: str, staging_root: Path) -> Path:
    """Copy the checkpoint + tokenizer + model card into a self-contained folder."""
    stage = staging_root / run
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(ckpt_dir, stage)
    tokenizer.save_pretrained(stage)
    step, _ = final_step_dir(run)
    cfg = run_config(run)
    vals = final_val_losses(run, step)
    card = model_card(run, step, cfg, vals).replace("REPO_ID", repo_id)
    (stage / "README.md").write_text(card)
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default=None,
                        help="HF user/org. Defaults to the token owner (whoami).")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"Repo name prefix; repo = <namespace>/<prefix>-<run>. Default '{DEFAULT_PREFIX}'.")
    parser.add_argument("--public", action="store_true", help="Create public repos (default: private).")
    parser.add_argument("--runs", nargs="+", default=RUNS, help="Subset of runs to upload.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage folders and print the plan; do not create or push repos.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    # Fail loudly if there's no usable auth (env token OR cached login).
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(
            "No Hugging Face auth found. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
            "or run `huggingface-cli login`.\n"
            f"  ({type(e).__name__}: {e})"
        )
    namespace = args.namespace or whoami["name"]
    private = not args.public

    print(f"auth ok as '{whoami['name']}'  ->  namespace '{namespace}'  "
          f"({'private' if private else 'PUBLIC'} repos)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    staging_root = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    print(f"staging in {staging_root}\n")

    for run in args.runs:
        step, ckpt_dir = final_step_dir(run)
        repo_id = f"{namespace}/{args.prefix}-{run}"
        stage = stage_repo(run, ckpt_dir, tokenizer, repo_id, staging_root)
        files = sorted(p.name for p in stage.iterdir())
        print(f"[{run}] final step {step}  ->  {repo_id}")
        print(f"        files: {files}")
        if args.dry_run:
            print("        (dry-run: not pushed)")
            continue
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                          commit_message=f"Upload {run} final checkpoint (step {step})")
        print(f"        pushed -> https://huggingface.co/{repo_id}")

    print(f"\ndone. staged copies left in {staging_root} (safe to delete).")


if __name__ == "__main__":
    main()
