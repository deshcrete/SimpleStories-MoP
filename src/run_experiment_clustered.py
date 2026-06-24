"""Orchestrator for the clustered base-model experiment (control/base_model_expr.md).

Builds per-persona splits from desh2806/SimpleStories-clustered, then trains one
specialist per persona. NO mixture model, NO eval/LoTP analysis (D2, D4) — the
deliverable is just the 4 trained cluster-specialist models.

Each step is its own subprocess so a failure produces a clean traceback without
poisoning the rest of the run. Re-running is safe: build_splits_clustered overwrites
the JSONLs and train.py overwrites checkpoints under each run_name.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from data import PERSONAS

SRC = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SRC)
    if result.returncode != 0:
        sys.exit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    run([sys.executable, "build_splits_clustered.py"])

    # Specialists only — no "mixture" run (D2).
    for run_name in PERSONAS:
        run([sys.executable, "train.py", "--run", run_name])


if __name__ == "__main__":
    main()
