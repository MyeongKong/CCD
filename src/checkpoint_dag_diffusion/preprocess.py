from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Iterable

import torch

from .chemistry import DEFAULT_ATOM_VOCAB, DEFAULT_BOND_VOCAB, smiles_to_molecule_graph
from .dag_topology import build_checkpoint_dag
from .topology import MoleculeGraph
from .tensorize import TensorSpec


def load_smiles(path: str | Path, max_molecules: int | None = None) -> list[str]:
    smiles: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()[0] if line.strip() else ""
            if not value or value.startswith("#"):
                continue
            smiles.append(value)
            if max_molecules is not None and len(smiles) >= max_molecules:
                break
    return smiles


def load_graph_json(path: str | Path, max_molecules: int | None = None) -> list[MoleculeGraph]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "molecules" in data:
        data = data["molecules"]
    graphs = [MoleculeGraph.from_dict(item) for item in data]
    if max_molecules is not None:
        graphs = graphs[:max_molecules]
    return graphs


def build_records_from_graphs(
    graphs: Iterable[MoleculeGraph],
    atom_vocab: tuple[str, ...] = tuple(DEFAULT_ATOM_VOCAB),
    bond_vocab: tuple[str, ...] = tuple(DEFAULT_BOND_VOCAB),
    max_nodes: int = 32,
    max_radius: int | None = None,
    include_skip_edges: bool = True,
) -> list[dict]:
    records: list[dict] = []
    spec = TensorSpec(atom_vocab=atom_vocab, bond_vocab=bond_vocab, max_nodes=max_nodes)
    for idx, graph in enumerate(graphs):
        if graph.num_atoms == 0 or graph.num_atoms > max_nodes:
            continue
        dag = build_checkpoint_dag(
            graph,
            atom_vocab=atom_vocab,
            bond_vocab=bond_vocab,
            max_radius=max_radius,
            include_skip_edges=include_skip_edges,
        )
        if any(cp.state.num_nodes > max_nodes for cp in dag.checkpoints):
            continue
        record = dag.to_dict()
        record["index"] = idx
        record["source_graph"] = graph.to_dict()
        record["max_nodes"] = max_nodes
        record["node_feature_dim"] = spec.node_feature_dim
        records.append(record)
    return records


def build_records_from_smiles(
    smiles_values: Iterable[str],
    atom_vocab: tuple[str, ...] = tuple(DEFAULT_ATOM_VOCAB),
    bond_vocab: tuple[str, ...] = tuple(DEFAULT_BOND_VOCAB),
    max_nodes: int = 32,
    max_radius: int | None = None,
    include_skip_edges: bool = True,
) -> list[dict]:
    graphs = []
    for smi in smiles_values:
        try:
            graphs.append(smiles_to_molecule_graph(smi, source_id=smi))
        except Exception:
            continue
    return build_records_from_graphs(
        graphs,
        atom_vocab=atom_vocab,
        bond_vocab=bond_vocab,
        max_nodes=max_nodes,
        max_radius=max_radius,
        include_skip_edges=include_skip_edges,
    )


def split_records(records: list[dict], val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 0) -> dict[str, list[dict]]:
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n = len(indices)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_ids = set(indices[:n_test])
    val_ids = set(indices[n_test : n_test + n_val])
    splits = {"train": [], "val": [], "test": []}
    for idx, record in enumerate(records):
        if idx in test_ids:
            splits["test"].append(record)
        elif idx in val_ids:
            splits["val"].append(record)
        else:
            splits["train"].append(record)
    if not splits["train"] and records:
        splits["train"] = list(records)
    return splits


def summarize_records(records: list[dict], max_nodes: int, atom_vocab: tuple[str, ...], bond_vocab: tuple[str, ...]) -> dict:
    level_counts: Counter[int] = Counter()
    edge_delta_counts: Counter[int] = Counter()
    edge_operator_counts: Counter[int] = Counter()
    state_hash_counts: Counter[str] = Counter()
    path_edge_counts: Counter[int] = Counter()
    max_level = 0
    for record in records:
        for checkpoint in record["checkpoints"]:
            level = int(checkpoint["level"])
            level_counts[level] += 1
            max_level = max(max_level, level)
            state_hash_counts[str(checkpoint["hash"])] += 1
        for edge in record["dag_edges"]:
            edge_delta_counts[int(edge["delta"])] += 1
            edge_operator_counts[int(edge["operator_id"])] += 1
        path_edge_counts[len(record["dag_edges"])] += 1

    shared_hist = Counter(state_hash_counts.values())
    return {
        "num_records": len(records),
        "num_checkpoint_states": sum(level_counts.values()),
        "num_unique_checkpoint_states": len(state_hash_counts),
        "num_dag_edges": sum(edge_delta_counts.values()),
        "level_counts": dict(sorted(level_counts.items())),
        "edge_delta_counts": dict(sorted(edge_delta_counts.items())),
        "edge_operator_counts": dict(sorted(edge_operator_counts.items())),
        "shared_checkpoint_counts": dict(sorted(shared_hist.items())),
        "dag_edge_count_hist": dict(sorted(path_edge_counts.items())),
        "max_level": max_level,
        "atom_vocab": list(atom_vocab),
        "bond_vocab": list(bond_vocab),
        "max_nodes": max_nodes,
        "format": "checkpoint_dag_diffusion.corpus.v2",
        "certified_object_format": "checkpoint_dag_diffusion.certified_object.v2",
        "relation_order": ["guide_level", "target_level", "delta", "operator_id"],
        "correspondence_maps": True,
    }


def write_corpus(
    records: list[dict],
    out_dir: str | Path,
    atom_vocab: tuple[str, ...] = tuple(DEFAULT_ATOM_VOCAB),
    bond_vocab: tuple[str, ...] = tuple(DEFAULT_BOND_VOCAB),
    max_nodes: int = 32,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = split_records(records, seed=seed)
    for split, split_records_ in splits.items():
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(split_records_, split_dir / "records.pt")

    manifest = summarize_records(records, max_nodes=max_nodes, atom_vocab=atom_vocab, bond_vocab=bond_vocab)
    manifest["splits"] = {key: len(value) for key, value in splits.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def preprocess_to_corpus(
    out_dir: str | Path,
    smiles_txt: str | Path | None = None,
    graph_json: str | Path | None = None,
    max_molecules: int | None = None,
    max_nodes: int = 32,
    max_radius: int | None = None,
    include_skip_edges: bool = True,
    seed: int = 0,
) -> dict:
    if smiles_txt is None and graph_json is None:
        raise ValueError("Provide either smiles_txt or graph_json.")
    if graph_json is not None:
        graphs = load_graph_json(graph_json, max_molecules=max_molecules)
        records = build_records_from_graphs(
            graphs,
            max_nodes=max_nodes,
            max_radius=max_radius,
            include_skip_edges=include_skip_edges,
        )
    else:
        smiles = load_smiles(smiles_txt, max_molecules=max_molecules)
        records = build_records_from_smiles(
            smiles,
            max_nodes=max_nodes,
            max_radius=max_radius,
            include_skip_edges=include_skip_edges,
        )
    return write_corpus(records, out_dir=out_dir, max_nodes=max_nodes, seed=seed)
