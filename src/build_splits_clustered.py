"""Build per-persona train/val/test splits from the clustered SimpleStories dataset.

Counterpart to build_splits.py, but the personas come from the `cluster` column of
the HuggingFace dataset `desh2806/SimpleStories-clustered` (control/base_model_expr.md).

Decisions (control/base_model_expr.md, confirmed 2026-06-24):
    D1  Keep clusters 0,1,2,3 and DROP cluster 4 (only 8,351 rows). Truncate each
        kept cluster to N_CAP = 10,000 -> uniform N across personas.
    D3  Name cluster k -> "persona_k" (the dataset's own `persona` column is empty).

For each kept cluster we shuffle with a fixed seed and partition:

    test  (500)   -> held out from training (unused this run; kept for parity)
    val   (500)   -> per-checkpoint overfitting diagnostic in train.py
    train (rest)  -> the specialist's training data (9,000 = 10,000 - 500 - 500)

Outputs:
    data/persona_<k>/train.jsonl
    data/persona_<k>/val.jsonl
    data/persona_<k>/test.jsonl
    data/manifest.json   (source repo, seed, cap, per-persona counts)

The dataset has no `id` column; `generation_id` is unique across all 60k rows, so
we use it as `id`. Rows are deduped + sorted by `id` before shuffling, so the split
is invariant to source ordering and bit-identical across re-runs with the same SEED.
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HF_REPO = "desh2806/SimpleStories-clustered"
HF_FILE = "data/train-00000-of-00001.parquet"

# D1: keep these clusters, drop cluster 4. cluster k -> persona_k (D3).
CLUSTERS_TO_KEEP = [0, 1, 2, 3]
N_CAP = 10_000  # truncate each kept cluster to this many rows -> uniform N

SEED = 42
N_TEST = 500
N_VAL = 500

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def persona_name(cluster: int) -> str:
    return f"persona_{cluster}"


def load_clustered_frame() -> list[dict]:
    """Download the HF parquet and return the rows we need as dicts.

    Only the three columns the pipeline uses are loaded. (The pandas `hf://` path is
    broken in this env's fsspec, so we download the file and read it with pyarrow.)
    """
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    table = pq.read_table(path, columns=["generation_id", "cluster", "story"])
    cols = table.to_pydict()
    rows = [
        {"id": gid, "cluster": int(cl), "story": st}
        for gid, cl, st in zip(cols["generation_id"], cols["cluster"], cols["story"])
    ]
    # generation_id is the only unique key; fail loudly if that ever stops holding.
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("generation_id is not unique; cannot use it as the row id.")
    return rows


def split_one_cluster(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Cap to N_CAP then slice test/val/train. Fails loudly if too few rows."""
    if len(rows) < N_CAP:
        raise RuntimeError(
            f"cluster has {len(rows)} rows, need at least N_CAP={N_CAP}. "
            f"Adjust CLUSTERS_TO_KEEP / N_CAP in base_model_expr.md."
        )
    # Dedup by id (defensive) then sort so source ordering can't perturb the shuffle.
    by_id = {r["id"]: r for r in rows}
    ordered = sorted(by_id.values(), key=lambda r: r["id"])
    rng.shuffle(ordered)
    capped = ordered[:N_CAP]
    return {
        "test":  capped[:N_TEST],
        "val":   capped[N_TEST : N_TEST + N_VAL],
        "train": capped[N_TEST + N_VAL :],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            # Persist only the columns the loaders read, matching build_splits.py.
            f.write(json.dumps({"id": r["id"], "persona": r["persona"], "story": r["story"]}) + "\n")


def main() -> None:
    rows = load_clustered_frame()

    by_cluster: dict[int, list[dict]] = {}
    for r in rows:
        by_cluster.setdefault(r["cluster"], []).append(r)

    missing = [c for c in CLUSTERS_TO_KEEP if c not in by_cluster]
    if missing:
        raise RuntimeError(f"clusters {missing} absent from {HF_REPO}; present: {sorted(by_cluster)}")

    # Single rng shared across personas; state advances deterministically through
    # CLUSTERS_TO_KEEP. Re-running with the same SEED reproduces all splits.
    rng = random.Random(SEED)
    manifest: dict = {
        "source_repo": HF_REPO,
        "source_file": HF_FILE,
        "seed": SEED,
        "n_cap": N_CAP,
        "clusters_kept": CLUSTERS_TO_KEEP,
        "splits": {"test": N_TEST, "val": N_VAL, "train": N_CAP - N_TEST - N_VAL},
        "personas": {},
    }

    for cluster in CLUSTERS_TO_KEEP:
        persona = persona_name(cluster)
        cluster_rows = by_cluster[cluster]
        for r in cluster_rows:
            r["persona"] = persona  # label by persona_k, not the dataset's empty column
        splits = split_one_cluster(cluster_rows, rng)

        persona_dir = DATA_DIR / persona
        for split_name, split_rows in splits.items():
            write_jsonl(split_rows, persona_dir / f"{split_name}.jsonl")

        manifest["personas"][persona] = {
            "cluster": cluster,
            "available": len(cluster_rows),
            "counts": {k: len(v) for k, v in splits.items()},
        }
        counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"{persona} (cluster {cluster}): available={len(cluster_rows)} -> {counts}")

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest written to {manifest_path}")


if __name__ == "__main__":
    main()
