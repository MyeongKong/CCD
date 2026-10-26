#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    corpus = ROOT / "runs" / "smoke" / "corpus"
    run_dir = ROOT / "runs" / "smoke" / "train"
    samples = ROOT / "runs" / "smoke" / "samples"
    run(
        [
            sys.executable,
            "scripts/preprocess_dag_qm9.py",
            "--graph_json",
            "data/toy_graphs.json",
            "--out_dir",
            str(corpus),
            "--max_nodes",
            "16",
        ]
    )
    run([sys.executable, "scripts/inspect_checkpoint_dag.py", "--corpus_dir", str(corpus), "--split", "train"])
    run(
        [
            sys.executable,
            "scripts/train_dag_qm9.py",
            "--config",
            "configs/smoke.yaml",
            "--corpus_dir",
            str(corpus),
            "--run_dir",
            str(run_dir),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/sample_dag_qm9.py",
            "--checkpoint",
            str(run_dir / "checkpoints" / "last.pt"),
            "--corpus_dir",
            str(corpus),
            "--out_dir",
            str(samples),
            "--num_samples",
            "8",
            "--device",
            "cpu",
        ]
    )


if __name__ == "__main__":
    main()
