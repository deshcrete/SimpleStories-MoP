"""Build per-persona splits from desh2806/simplestories-personas.

The dataset has a single `default` config; each row carries a `persona` column.
We concatenate every available HF split (train/heldout/raw or whatever is
exposed) into one pool, group by persona, then for each persona shuffle with a
fixed seed and partition:

    persona_train (500)  -> the specialist's training data
    mixture       (500)  -> this persona's contribution to the mixture model's training set
    inference     (150)  -> held out from all training; part of the 750-seq LoTP eval
    reserve       (rest) -> untouched during the main experiment

Outputs:
    data/<persona>/persona_train.jsonl
    data/<persona>/mixture.jsonl
    data/<persona>/inference.jsonl
    data/<persona>/reserve.jsonl
    data/manifest.json            (source, seed, per-persona counts, hf splits seen)

Re-running with the same SEED produces bit-identical splits (rows are sorted by
`id` before shuffling, so HF split ordering does not affect the result).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

SOURCE = "desh2806/simplestories-personas"
PERSONAS = [
    "noir_detective",
    "fairy_tale",
    "scientific_explainer",
    "absurdist",
    "epistolary",
]
SEED = 42

N_PERSONA_TRAIN = 500
N_MIXTURE = 500
N_INFERENCE = 150
N_USED = N_PERSONA_TRAIN + N_MIXTURE + N_INFERENCE  # 1150

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_full_pool() -> tuple[list[dict], list[str]]:
    """Load every HF split for the default config and concatenate. Returns
    (rows, split_names_seen)."""
    dsd = load_dataset(SOURCE)
    split_names = list(dsd.keys())
    if not split_names:
        raise RuntimeError(f"{SOURCE}: no splits returned by load_dataset")
    pool = concatenate_datasets([dsd[s] for s in split_names])
    rows = [dict(r) for r in pool]
    if not rows or "persona" not in rows[0]:
        raise RuntimeError(f"{SOURCE}: 'persona' column missing; got keys {list(rows[0].keys()) if rows else []}")
    if "id" not in rows[0]:
        raise RuntimeError(f"{SOURCE}: 'id' column missing; needed for reproducible ordering")
    return rows, split_names


def group_by_persona(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["persona"]].append(r)
    missing = [p for p in PERSONAS if p not in grouped]
    if missing:
        raise RuntimeError(f"Expected personas {PERSONAS}, missing: {missing}. Got: {list(grouped.keys())}")
    extras = [p for p in grouped if p not in PERSONAS]
    if extras:
        # Fail loudly: an unexpected persona in the dataset means our split set is wrong.
        raise RuntimeError(f"Unexpected personas in dataset: {extras}")
    return grouped


def split_one_persona(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    if len(rows) < N_USED:
        raise RuntimeError(f"Persona pool has {len(rows)} rows, need at least {N_USED}.")
    # Sort by id first so HF split ordering doesn't perturb our shuffle.
    ordered = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(ordered)
    return {
        "persona_train": ordered[:N_PERSONA_TRAIN],
        "mixture":       ordered[N_PERSONA_TRAIN : N_PERSONA_TRAIN + N_MIXTURE],
        "inference":     ordered[N_PERSONA_TRAIN + N_MIXTURE : N_USED],
        "reserve":       ordered[N_USED:],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    pool, hf_splits = load_full_pool()
    print(f"Loaded {len(pool)} total rows from {SOURCE} (hf splits: {hf_splits})")
    grouped = group_by_persona(pool)

    # Single rng shared across personas; state advances deterministically through
    # the PERSONAS list. Re-running with the same SEED reproduces all five splits.
    rng = random.Random(SEED)
    manifest: dict = {
        "source": SOURCE,
        "hf_splits_concatenated": hf_splits,
        "seed": SEED,
        "splits": {
            "persona_train": N_PERSONA_TRAIN,
            "mixture": N_MIXTURE,
            "inference": N_INFERENCE,
        },
        "personas": {},
    }

    for persona in PERSONAS:
        rows = grouped[persona]
        splits = split_one_persona(rows, rng)

        persona_dir = DATA_DIR / persona
        for split_name, split_rows in splits.items():
            write_jsonl(split_rows, persona_dir / f"{split_name}.jsonl")

        manifest["personas"][persona] = {
            "total": len(rows),
            "counts": {k: len(v) for k, v in splits.items()},
        }
        counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"{persona}: total={len(rows)} -> {counts}")

    manifest_path = DATA_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written to {manifest_path}")


if __name__ == "__main__":
    main()
