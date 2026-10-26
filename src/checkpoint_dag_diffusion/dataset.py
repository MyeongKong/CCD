from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .artifacts import ArtifactVersionError, load_corpus_manifest, load_corpus_records
from .checkpoint_state import CheckpointState
from .tensorize import TensorSpec, checkpoint_to_tensors, collate_tensor_dicts, empty_condition, relation_to_tensor


class CheckpointDAGEdgeDataset(Dataset):
    def __init__(
        self,
        corpus_dir: str | Path,
        split: str = "train",
        spec: TensorSpec | None = None,
        include_noise_edges: bool = True,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.split = split
        manifest = load_corpus_manifest(self.corpus_dir)
        self.records: list[dict[str, Any]] = load_corpus_records(self.corpus_dir, split)
        corpus_spec = TensorSpec(
            atom_vocab=tuple(manifest["atom_vocab"]),
            bond_vocab=tuple(manifest["bond_vocab"]),
            max_nodes=int(manifest["max_nodes"]),
        )
        if spec is not None and spec != corpus_spec:
            raise ArtifactVersionError(
                "The requested TensorSpec does not match the v2 corpus manifest; "
                "regenerate or load a matching artifact chain."
            )
        self.spec = corpus_spec
        self.index: list[tuple[int, int]] = []
        for rec_idx, record in enumerate(self.records):
            for edge_idx, edge in enumerate(record["dag_edges"]):
                if edge.get("guide") is None and not include_noise_edges:
                    continue
                self.index.append((rec_idx, edge_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, edge_idx = self.index[idx]
        record = self.records[rec_idx]
        edge = record["dag_edges"][edge_idx]
        target_checkpoint = record["checkpoints"][int(edge["target"])]
        target = CheckpointState.from_dict(target_checkpoint["state"])
        if edge.get("guide") is None:
            guide = None
            guide_hash = ""
        else:
            guide_checkpoint = record["checkpoints"][int(edge["guide"])]
            guide = CheckpointState.from_dict(guide_checkpoint["state"])
            guide_hash = str(guide_checkpoint["hash"])

        target_t = checkpoint_to_tensors(target_checkpoint, self.spec)
        guide_t = checkpoint_to_tensors(guide_checkpoint, self.spec) if guide is not None else empty_condition(self.spec)
        correspondence = edge.get("correspondence", {})
        target_to_guide = torch.full((self.spec.max_nodes,), -1, dtype=torch.long)
        overlap_mask = torch.zeros(self.spec.max_nodes, dtype=torch.bool)
        correspondence_mask = torch.zeros(self.spec.max_nodes, dtype=torch.bool)
        boundary_expansion_mask = torch.zeros(self.spec.max_nodes, dtype=torch.bool)
        for target_idx, guide_idx in correspondence.get("target_to_guide", {}).items():
            target_i, guide_i = int(target_idx), int(guide_idx)
            if 0 <= target_i < self.spec.max_nodes and 0 <= guide_i < self.spec.max_nodes:
                target_to_guide[target_i] = guide_i
                correspondence_mask[target_i] = True
        for target_idx in correspondence.get("preserved_target_to_guide", {}):
            target_i = int(target_idx)
            if 0 <= target_i < self.spec.max_nodes:
                overlap_mask[target_i] = True
        for targets in correspondence.get("guide_boundary_to_target", {}).values():
            for target_idx in targets:
                target_i = int(target_idx)
                if 0 <= target_i < self.spec.max_nodes:
                    boundary_expansion_mask[target_i] = True
        return {
            "target": target_t,
            "guide": guide_t,
            "relation": relation_to_tensor(edge),
            "has_guide": torch.tensor(guide is not None, dtype=torch.bool),
            "record_index": torch.tensor(rec_idx, dtype=torch.long),
            "edge_index": torch.tensor(edge_idx, dtype=torch.long),
            "target_hash": str(target_checkpoint["hash"]),
            "guide_hash": guide_hash,
            "target_to_guide": target_to_guide,
            "overlap_mask": overlap_mask,
            "correspondence_mask": correspondence_mask,
            "boundary_expansion_mask": boundary_expansion_mask,
        }


def collate_edges(items: list[dict]) -> dict:
    return {
        "target": collate_tensor_dicts([item["target"] for item in items]),
        "guide": collate_tensor_dicts([item["guide"] for item in items]),
        "relation": torch.stack([item["relation"] for item in items], dim=0),
        "has_guide": torch.stack([item["has_guide"] for item in items], dim=0),
        "record_index": torch.stack([item["record_index"] for item in items], dim=0),
        "edge_index": torch.stack([item["edge_index"] for item in items], dim=0),
        "target_hash": [item["target_hash"] for item in items],
        "guide_hash": [item["guide_hash"] for item in items],
        "target_to_guide": torch.stack([item["target_to_guide"] for item in items], dim=0),
        "overlap_mask": torch.stack([item["overlap_mask"] for item in items], dim=0),
        "correspondence_mask": torch.stack([item["correspondence_mask"] for item in items], dim=0),
        "boundary_expansion_mask": torch.stack([item["boundary_expansion_mask"] for item in items], dim=0),
    }
