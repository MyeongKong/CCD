#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from checkpoint_dag_diffusion.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train checkpoint-DAG edge transition denoiser.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()
    result = train_from_config(args.config, corpus_dir=args.corpus_dir, run_dir=args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
