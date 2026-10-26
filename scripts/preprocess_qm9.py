#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from checkpoint_dag_diffusion.preprocess import preprocess_to_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checkpoint-DAG diffusion corpus.")
    parser.add_argument("--smiles_txt", type=str, default=None)
    parser.add_argument("--graph_json", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_molecules", type=int, default=None)
    parser.add_argument("--max_nodes", type=int, default=32)
    parser.add_argument("--max_radius", type=int, default=None)
    parser.add_argument("--no_skip_edges", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = preprocess_to_corpus(
        out_dir=args.out_dir,
        smiles_txt=args.smiles_txt,
        graph_json=args.graph_json,
        max_molecules=args.max_molecules,
        max_nodes=args.max_nodes,
        max_radius=args.max_radius,
        include_skip_edges=not args.no_skip_edges,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
