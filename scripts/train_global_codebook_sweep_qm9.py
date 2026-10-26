#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_dag_diffusion.global_codebook import assign_codebook, train_codebook
from checkpoint_dag_diffusion.global_graph import build_global_graph


def _parse_ks(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Provide at least one K value.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned global checkpoint codebook K-sweep.")
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ks", type=str, default="128,256,512,1024")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_global_graph", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for k in _parse_ks(args.ks):
        run_dir = root / f"K{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trained = train_codebook(
            corpus_dir=args.corpus_dir,
            out_dir=run_dir,
            split=args.split,
            codebook_size=k,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            device=args.device,
            seed=args.seed,
        )
        assigned = assign_codebook(
            corpus_dir=args.corpus_dir,
            codebook_path=trained["codebook"],
            out_dir=run_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
        graph_result = None
        if not args.skip_global_graph:
            graph_result = build_global_graph(args.corpus_dir, assigned["assignments"], run_dir, split=args.split)
        summary[str(k)] = {
            "run_dir": str(run_dir),
            "codebook": trained["codebook"],
            "last_epoch": trained["history"][-1] if trained["history"] else {},
            "assignment": assigned["summary"],
            "global_graph": None if graph_result is None else graph_result["manifest"],
        }
        (root / "k_sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
