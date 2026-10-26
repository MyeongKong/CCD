#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

from checkpoint_dag_diffusion.artifacts import load_corpus_manifest, load_corpus_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect checkpoint-DAG corpus statistics.")
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    corpus = Path(args.corpus_dir)
    manifest = load_corpus_manifest(corpus)
    records = load_corpus_records(corpus, args.split)
    checkpoint_counts = Counter(len(record["checkpoints"]) for record in records)
    edge_counts = Counter(len(record["dag_edges"]) for record in records)
    path_count_proxy = Counter()
    for record in records:
        for edge in record["dag_edges"]:
            path_count_proxy[(edge["target_level"], edge["guide_level"], edge["delta"])] += 1
    summary = {
        "manifest": manifest,
        "split": args.split,
        "num_split_records": len(records),
        "checkpoint_count_hist": dict(sorted(checkpoint_counts.items())),
        "dag_edge_count_hist": dict(sorted(edge_counts.items())),
        "top_relations": [
            {"target_level": key[0], "guide_level": key[1], "delta": key[2], "count": count}
            for key, count in path_count_proxy.most_common(args.top_k)
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
