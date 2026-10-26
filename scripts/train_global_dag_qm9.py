#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from checkpoint_dag_diffusion.global_train import train_global_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train global-code-conditioned checkpoint DAG denoiser.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--global_graph", type=str, required=True)
    parser.add_argument("--assignments", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    result = train_global_from_config(
        config_path=args.config,
        corpus_dir=args.corpus_dir,
        global_graph_path=args.global_graph,
        assignments_path=args.assignments,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
