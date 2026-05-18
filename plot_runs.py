"""Plot holdout NLL curves from finetuning runs.

Reads `runs/<name>/holdout_metrics.jsonl` from every subdirectory and writes:
    plots/overall.png       one line per run, holdout/nll_overall vs step
    plots/per_theme.png     one subplot per theme, one line per run
    plots/own_vs_cross.png  one subplot per persona run, comparing the run's
                            NLL on its OWN theme vs the mean over the other
                            themes (forking = specialization; both rising =
                            overfitting/forgetting)

Usage:
    python plot_runs.py
    python plot_runs.py --runs_dir runs --output_dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_run(metrics_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def collect_runs(runs_dir: Path) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        metrics = run_dir / "holdout_metrics.jsonl"
        if metrics.exists():
            runs[run_dir.name] = load_run(metrics)
    return runs


def plot_overall(runs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in runs.items():
        if "holdout/nll_overall" not in df.columns:
            continue
        ax.plot(df["step"], df["holdout/nll_overall"], label=name, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("holdout NLL (overall)")
    ax.set_title("Holdout NLL by run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_theme(runs: dict[str, pd.DataFrame], out_path: Path) -> None:
    theme_cols: set[str] = set()
    for df in runs.values():
        theme_cols.update(c for c in df.columns if c.startswith("holdout/nll["))
    theme_cols_sorted = sorted(theme_cols)
    if not theme_cols_sorted:
        return

    n = len(theme_cols_sorted)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, col in zip(axes.flat, theme_cols_sorted):
        theme = col[len("holdout/nll[") : -1]
        for name, df in runs.items():
            if col not in df.columns:
                continue
            ax.plot(df["step"], df[col], label=name, marker="o", markersize=3)
        ax.set_title(theme)
        ax.set_xlabel("step")
        ax.set_ylabel("holdout NLL")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes.flat[len(theme_cols_sorted):]:
        ax.set_visible(False)

    fig.suptitle("Per-theme holdout NLL by run")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _slugify(theme: str) -> str:
    # Matches the slug convention used in personaDataset.build_theme_splits.
    return "".join(c if c.isalnum() else "_" for c in str(theme)).strip("_").lower()


def plot_own_vs_cross(runs: dict[str, pd.DataFrame], out_path: Path) -> None:
    persona_runs = {n: df for n, df in runs.items() if n.startswith("theme_")}
    if not persona_runs:
        return

    all_themes: set[str] = set()
    for df in runs.values():
        for c in df.columns:
            if c.startswith("holdout/nll["):
                all_themes.add(c[len("holdout/nll[") : -1])
    slug_to_theme = {_slugify(t): t for t in all_themes}

    mixture_df = runs.get("mixture")

    items = sorted(persona_runs.items())
    n = len(items)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, (name, df) in zip(axes.flat, items):
        # Run name format: "theme_<i>_<slug>".
        parts = name.split("_", 2)
        own_theme = slug_to_theme.get(parts[2]) if len(parts) == 3 else None
        own_col = f"holdout/nll[{own_theme}]" if own_theme else None
        if own_col is None or own_col not in df.columns:
            ax.set_title(f"{name} (unmatched)")
            continue

        cross_cols = [
            f"holdout/nll[{t}]"
            for t in all_themes
            if t != own_theme and f"holdout/nll[{t}]" in df.columns
        ]
        cross_mean = df[cross_cols].mean(axis=1)

        ax.plot(df["step"], df[own_col],
                label=f"own ({own_theme})", color="C0", marker="o", markersize=3)
        ax.plot(df["step"], cross_mean,
                label="cross (mean of others)", color="C3", marker="o", markersize=3, linestyle="--")
        if mixture_df is not None and own_col in mixture_df.columns:
            ax.plot(mixture_df["step"], mixture_df[own_col],
                    label="mixture on own theme", color="gray", linewidth=1.2, alpha=0.8)

        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("holdout NLL")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Own-theme vs cross-theme holdout NLL (persona runs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--output_dir", default="plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(runs_dir)
    if not runs:
        print(f"No holdout_metrics.jsonl files found under {runs_dir}/")
        return

    plot_overall(runs, out_dir / "overall.png")
    plot_per_theme(runs, out_dir / "per_theme.png")
    plot_own_vs_cross(runs, out_dir / "own_vs_cross.png")
    print(f"Wrote overall.png, per_theme.png, own_vs_cross.png in {out_dir}/ "
          f"for {len(runs)} runs.")


if __name__ == "__main__":
    main()
