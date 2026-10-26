#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_dag_diffusion.global_codebook import assign_codebook
from checkpoint_dag_diffusion.global_graph import build_global_graph, build_hash_assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build global checkpoint transition graph.")
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--method", choices=["hash", "learned"], default="learned")
    parser.add_argument("--codebook", type=str, default=None)
    parser.add_argument("--assignments", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignments = args.assignments
    result: dict = {}
    if assignments is None:
        if args.method == "hash":
            built = build_hash_assignments(args.corpus_dir, out, split=args.split)
            assignments = built["assignments"]
            result["assignment"] = built
        else:
            if args.codebook is None:
                raise ValueError("--codebook is required for learned global graph unless --assignments is provided.")
            built = assign_codebook(
                corpus_dir=args.corpus_dir,
                codebook_path=args.codebook,
                out_dir=out,
                split=args.split,
                batch_size=args.batch_size,
                device=args.device,
            )
            assignments = built["assignments"]
            result["assignment"] = built
    graph = build_global_graph(args.corpus_dir, assignments, out, split=args.split)
    result["global_graph"] = graph
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
