import json
import os
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
import pandas as pd


def load_validation_df(
    repo_id: str = "lennart-finke/SimpleStories",
    split: str = "test",
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """Load the SimpleStories validation split as a pandas DataFrame for EDA.

    Pass `hf_token` explicitly, or set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in the env
    for higher rate limits.
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir, token=token)
    return ds.to_pandas()


def build_theme_finetuning_dataset(
    n_themes: int = 6,
    text_column: str = "story",
    theme_column: str = "theme",
    output_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    hf_token: str | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Build a finetuning dataset from the top-`n_themes` most common themes.

    Loads the SimpleStories validation split (unless `df` is provided), keeps
    the `n_themes` largest theme groups, and downsamples each to the size of
    the smallest so all themes are equally represented. Returns a DataFrame
    with columns [text, theme]. If `output_path` is given, also writes the
    dataset as JSONL with one `{"text": ..., "theme": ...}` object per line.
    """
    if df is None:
        df = load_validation_df(hf_token=hf_token)

    counts = df[theme_column].value_counts().head(n_themes)
    top_themes = counts.index.tolist()
    per_theme = int(counts.min())

    subset = (
        df[df[theme_column].isin(top_themes)][[text_column, theme_column]]
        .groupby(theme_column, group_keys=False)
        .sample(n=per_theme, random_state=random_state)
        .rename(columns={text_column: "text", theme_column: "theme"})
        .reset_index(drop=True)
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in subset.itertuples(index=False):
                f.write(json.dumps({"text": row.text, "theme": row.theme}) + "\n")

    return subset


def _write_jsonl(df_part: pd.DataFrame, text_column: str, theme_column: str, path: Path) -> None:
    records = (
        df_part[[text_column, theme_column]]
        .rename(columns={text_column: "text", theme_column: "theme"})
        .to_dict("records")
    )
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_theme_splits(
    output_dir: str | Path,
    n_themes: int = 6,
    holdout_per_theme: int = 50,
    text_column: str = "story",
    theme_column: str = "theme",
    df: pd.DataFrame | None = None,
    hf_token: str | None = None,
    random_state: int = 0,
) -> dict[str, Path]:
    """Build per-theme + mixture + uniform holdout JSONL files.

    - Picks the top `n_themes` themes by count.
    - Reserves `holdout_per_theme` rows from each as a uniform held-out set.
    - Balances the remaining per-theme rows to the size of the smallest.
    - Writes <output_dir>/theme_{i}_{slug}.jsonl, mixture.jsonl, holdout.jsonl.

    Returns a dict mapping logical names ("theme_0"..., "mixture", "holdout")
    to file paths. The holdout is disjoint from every training file.
    """
    if df is None:
        df = load_validation_df(hf_token=hf_token)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = df[theme_column].value_counts().head(n_themes)
    top_themes = counts.index.tolist()
    pool = df[df[theme_column].isin(top_themes)].copy()

    holdout = (
        pool.groupby(theme_column, group_keys=False)
        .sample(n=holdout_per_theme, random_state=random_state)
    )
    remaining = pool.drop(holdout.index)
    per_theme_min = int(remaining[theme_column].value_counts().min())
    balanced = (
        remaining.groupby(theme_column, group_keys=False)
        .sample(n=per_theme_min, random_state=random_state)
    )

    paths: dict[str, Path] = {}
    for i, theme in enumerate(top_themes):
        slug = "".join(c if c.isalnum() else "_" for c in str(theme)).strip("_").lower()
        p = out / f"theme_{i}_{slug}.jsonl"
        _write_jsonl(balanced[balanced[theme_column] == theme], text_column, theme_column, p)
        paths[f"theme_{i}"] = p

    mixture_path = out / "mixture.jsonl"
    _write_jsonl(balanced, text_column, theme_column, mixture_path)
    paths["mixture"] = mixture_path

    holdout_path = out / "holdout.jsonl"
    _write_jsonl(holdout, text_column, theme_column, holdout_path)
    paths["holdout"] = holdout_path

    return paths


if __name__ == "__main__":
    paths = build_theme_splits("data")
    for k, v in paths.items():
        print(f"{k}: {v}")
