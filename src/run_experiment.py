"""End-to-end orchestrator: build splits, train all six models, evaluate every
checkpoint, fit LoTP π under both alignment modes.

Each step is its own subprocess so a failure in one stage produces a clean traceback
without poisoning the rest of the run. Re-running this script is safe: build_splits
overwrites existing JSONLs; train.py overwrites checkpoints under its run_name;
eval_checkpoints.py overwrites its metrics directory.
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
    run([sys.executable, "build_splits.py"])

    for run_name in PERSONAS + ["mixture"]:
        run([sys.executable, "train.py", "--run", run_name])

    for run_name in PERSONAS + ["mixture"]:
        run([sys.executable, "eval_checkpoints.py", "--run", run_name])

    run([sys.executable, "lotp.py"])


if __name__ == "__main__":
    main()
