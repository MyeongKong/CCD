#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from checkpoint_dag_diffusion.sampling import sample_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from checkpoint-DAG diffusion.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=1.3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--none_bias", type=float, default=2.0)
    parser.add_argument("--max_degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = sample_pipeline(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        corpus_dir=args.corpus_dir,
        device=args.device,
        guidance=args.guidance,
        temperature=args.temperature,
        none_bias=args.none_bias,
        max_degree=args.max_degree,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
