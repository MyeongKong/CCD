#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter

from checkpoint_dag_diffusion.global_graph import load_global_graph, summarize_global_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect global checkpoint graph artifact.")
    parser.add_argument("--global_graph", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()
    graph = load_global_graph(args.global_graph)
    summary = summarize_global_graph(graph)
    summary["top_nodes"] = [
        {
            "global_id": int(node["global_id"]),
            "support_count": int(node["support_count"]),
            "level_hist": node.get("level_hist", {}),
            "ring_count_hist": node.get("ring_count_hist", {}),
            "hash_purity": float(node.get("hash_purity", 0.0)),
        }
        for node in sorted(graph["global_nodes"], key=lambda item: int(item["support_count"]), reverse=True)[: args.top_k]
    ]
    summary["top_edges"] = [
        {
            "global_edge_id": int(edge["global_edge_id"]),
            "guide_global_id": int(edge["guide_global_id"]),
            "target_global_id": int(edge["target_global_id"]),
            "operator_id": int(edge["operator_id"]),
            "count": int(edge["count"]),
        }
        for edge in sorted(graph["global_edges"], key=lambda item: int(item["count"]), reverse=True)[: args.top_k]
    ]
    path_starts = Counter(str(k) for k in graph.get("coarsest_prior", {}))
    summary["coarsest_prior_codes"] = len(path_starts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
