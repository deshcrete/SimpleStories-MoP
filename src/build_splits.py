"""Build per-persona train/val/test splits for the single-epoch experiment.

Reads the locally generated persona data (simple_stories_generate/data_raw/<persona>.jsonl,
produced by simple_stories_generate/generate_personas.py) and, for each persona,
shuffles with a fixed seed and partitions:

    test  (500)   -> held out from all training; part of the combined LoTP/test set
    val   (500)   -> held out; overfitting diagnostic (single-persona val; the union
                     of all 5 is the full-mix val)
    train (rest)  -> the specialist's training data AND this persona's contribution
                     to the mixture's training set (IDENTICAL split — see task_plan.md
                     "disjoint -> identical train"). With ~10k/persona this is ~9,000.

Outputs:
    data/<persona>/train.jsonl
    data/<persona>/val.jsonl
    data/<persona>/test.jsonl
    data/manifest.json   (source, seed, per-persona counts)

Re-running with the same SEED is bit-identical: rows are sorted by `id` before
shuffling, so any change in file ordering does not affect the result.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "simple_stories_generate" / "data_raw"
PERSONAS = [
    "noir_detective",
    "fairy_tale",
    "scientific_explainer",
    "absurdist",
    "epistolary",
]
SEED = 42

N_TEST = 500
N_VAL = 500
MIN_TRAIN = 1000  # fail loudly if a persona has too little data to train on

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_persona_rows(persona: str) -> list[dict]:
    """Read one persona's generated jsonl. Fails loudly if missing/empty."""
    path = SOURCE_DIR / f"{persona}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated data: {path}. Run "
            f"simple_stories_generate/generate_personas.py first."
        )
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    for required in ("id", "persona", "story"):
        if required not in rows[0]:
            raise RuntimeError(f"{path}: missing '{required}' column; got {list(rows[0].keys())}")
    bad = [r["persona"] for r in rows if r["persona"] != persona]
    if bad:
        raise RuntimeError(f"{path}: rows with wrong persona label, e.g. {bad[:3]}")
    return rows


def split_one_persona(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    need = N_TEST + N_VAL + MIN_TRAIN
    if len(rows) < need:
        raise RuntimeError(f"Persona pool has {len(rows)} rows, need at least {need}.")
    # Dedupe by id (generation can occasionally repeat params -> duplicate ids), then
    # sort by id so source ordering doesn't perturb the shuffle.
    by_id = {r["id"]: r for r in rows}
    ordered = sorted(by_id.values(), key=lambda r: r["id"])
    rng.shuffle(ordered)
    return {
        "test":  ordered[:N_TEST],
        "val":   ordered[N_TEST : N_TEST + N_VAL],
        "train": ordered[N_TEST + N_VAL :],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    # Single rng shared across personas; state advances deterministically through
    # the PERSONAS list. Re-running with the same SEED reproduces all splits.
    rng = random.Random(SEED)
    manifest: dict = {
        "source_dir": str(SOURCE_DIR),
        "seed": SEED,
        "splits": {"test": N_TEST, "val": N_VAL, "train": "rest"},
        "personas": {},
    }

    for persona in PERSONAS:
        rows = load_persona_rows(persona)
        splits = split_one_persona(rows, rng)

        persona_dir = DATA_DIR / persona
        for split_name, split_rows in splits.items():
            write_jsonl(split_rows, persona_dir / f"{split_name}.jsonl")

        manifest["personas"][persona] = {
            "total_deduped": sum(len(v) for v in splits.values()),
            "counts": {k: len(v) for k, v in splits.items()},
        }
        counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"{persona}: total={len(rows)} -> {counts}")

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written to {manifest_path}")


if __name__ == "__main__":
    main()
