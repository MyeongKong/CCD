#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_dag_diffusion.global_eval import evaluate_global_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global checkpoint samples.")
    parser.add_argument("--samples_json", type=str, required=True)
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--reference_split", type=str, default="test")
    parser.add_argument("--global_graph", type=str, default=None)
    parser.add_argument(
        "--require_rdkit",
        action="store_true",
        help="Fail if RDKit is unavailable instead of using structural validity as a smoke-test fallback.",
    )
    parser.add_argument("--compute_fcd", action="store_true")
    parser.add_argument("--fcd_device", type=str, default="cpu")
    parser.add_argument("--fcd_jobs", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    result = evaluate_global_samples(
        args.samples_json,
        corpus_dir=args.corpus_dir,
        reference_split=args.reference_split,
        global_graph_path=args.global_graph,
        require_rdkit=args.require_rdkit,
        compute_fcd=args.compute_fcd,
        fcd_device=args.fcd_device,
        fcd_jobs=args.fcd_jobs,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
