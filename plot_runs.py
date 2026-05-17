"""Plot holdout NLL curves from finetuning runs.

Reads `runs/<name>/holdout_metrics.jsonl` from every subdirectory and writes:
    plots/overall.png      one line per run, holdout/nll_overall vs step
    plots/per_theme.png    one subplot per theme, one line per run

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
    print(f"Wrote {out_dir / 'overall.png'} and {out_dir / 'per_theme.png'} "
          f"for {len(runs)} runs.")


if __name__ == "__main__":
    main()
