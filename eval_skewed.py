"""Re-aggregate a run's per-theme holdout NLL under a skewed weighting.

The training run already wrote per-theme NLL into `<run>/holdout_metrics.jsonl`.
We just re-weight those columns with `exp(-decay * i)` weights (the same form
used to build the skewed train mixture) and overlay against the uniform-
weighted overall NLL.

The point: if the lack of a "productive learning phase" in the skewed-mixture
run was caused by train/eval distribution mismatch, then evaluating against
a *matching* skewed holdout should restore an initial decreasing period.

Usage:
    python eval_skewed.py --run runs/mixture_skew_d1 --decay 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def theme_order(data_dir: Path) -> list[str]:
    """Return theme names in the same order as `data/theme_<i>_*.jsonl`."""
    names: list[str] = []
    for p in sorted(data_dir.glob("theme_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            first = json.loads(next(line for line in f if line.strip()))
        names.append(first["theme"])
    if not names:
        raise FileNotFoundError(f"no theme_*.jsonl under {data_dir}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to run dir")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                    help="output png path; defaults to <run>/skewed_eval.png")
    args = ap.parse_args()

    run = Path(args.run)
    metrics_path = run / "holdout_metrics.jsonl"
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    themes = theme_order(Path(args.data_dir))
    raw = [math.exp(-args.decay * i) for i in range(len(themes))]
    z = sum(raw)
    weights = [w / z for w in raw]

    cols = [f"holdout/nll[{t}]" for t in themes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in metrics: {missing}")

    df["holdout/nll_skew"] = sum(w * df[c] for w, c in zip(weights, cols))

    out = Path(args.out) if args.out else run / "skewed_eval.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["step"], df["holdout/nll_overall"], label="uniform holdout",
            color="C0", marker="o", markersize=3)
    ax.plot(df["step"], df["holdout/nll_skew"],
            label=f"skew-weighted holdout (decay={args.decay})",
            color="C3", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("holdout NLL")
    ax.set_title(f"{run.name}: uniform vs skew-weighted holdout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Wrote {out}\n")
    print("Theme weights:")
    for t, w in zip(themes, weights):
        print(f"  {t:<20}  {w:.4f}")
    print()
    first, last = df.iloc[0], df.iloc[-1]
    print(f"step {int(first['step']):>4}:  uniform={first['holdout/nll_overall']:.4f}  "
          f"skew={first['holdout/nll_skew']:.4f}")
    print(f"step {int(last['step']):>4}:  uniform={last['holdout/nll_overall']:.4f}  "
          f"skew={last['holdout/nll_skew']:.4f}")
    print(f"Δuniform = {last['holdout/nll_overall'] - first['holdout/nll_overall']:+.4f}")
    print(f"Δskew    = {last['holdout/nll_skew'] - first['holdout/nll_skew']:+.4f}")
    print(f"min skew = {df['holdout/nll_skew'].min():.4f}  "
          f"@ step {int(df.loc[df['holdout/nll_skew'].idxmin(), 'step'])}")


if __name__ == "__main__":
    main()
