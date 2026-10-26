from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .artifacts import (
    ArtifactVersionError,
    load_assignments,
    load_corpus_manifest,
    validate_assignments_against_records,
)
from .checkpoint_state import CheckpointState
from .dataset import collate_edges
from .global_graph import load_corpus_records, load_global_graph
from .tensorize import TensorSpec, checkpoint_to_tensors, empty_condition, relation_to_tensor


class GlobalCheckpointEdgeDataset(Dataset):
    """Local DAG edge instances annotated with global checkpoint ids."""

    def __init__(
        self,
        corpus_dir: str | Path,
        global_graph_path: str | Path,
        assignments_path: str | Path,
        split: str = "train",
        spec: TensorSpec | None = None,
        include_noise_edges: bool = True,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.split = split
        self.records: list[dict[str, Any]] = load_corpus_records(corpus_dir, split)
        self.global_graph = load_global_graph(global_graph_path)
        self.assignments = load_assignments(assignments_path)
        validate_assignments_against_records(
            self.assignments,
            self.records,
            source=f"global assignments {Path(assignments_path)}",
        )
        if str(self.assignments["split"]) != str(split) or str(self.global_graph["split"]) != str(split):
            raise ValueError(
                "Corpus, global assignments, and global graph must use the same split; "
                "regenerate the v2 artifact chain together."
            )
        if str(self.global_graph["source_assignments_fingerprint"]) != str(
            self.assignments["artifact_fingerprint"]
        ):
            raise ArtifactVersionError(
                "Global graph and assignments come from different artifact chains; regenerate them together."
            )
        if str(self.global_graph["source_codebook_fingerprint"]) != str(
            self.assignments["source_codebook_fingerprint"]
        ):
            raise ArtifactVersionError(
                "Global graph and assignments reference different codebooks; regenerate them together."
            )
        spec_data = self.global_graph["spec"]
        graph_spec = TensorSpec(
            atom_vocab=tuple(spec_data["atom_vocab"]),
            bond_vocab=tuple(spec_data["bond_vocab"]),
            max_nodes=int(spec_data["max_nodes"]),
        )
        manifest = load_corpus_manifest(corpus_dir)
        corpus_spec = TensorSpec(
            atom_vocab=tuple(manifest["atom_vocab"]),
            bond_vocab=tuple(manifest["bond_vocab"]),
            max_nodes=int(manifest["max_nodes"]),
        )
        if graph_spec != corpus_spec or (spec is not None and spec != graph_spec):
            raise ArtifactVersionError(
                "Corpus, global graph, and requested TensorSpec do not match; "
                "regenerate the v2 artifact chain together."
            )
        self.spec = graph_spec
        self.assignment_map = {
            (int(item["record_index"]), int(item["checkpoint_index"])): int(item["global_id"])
            for item in self.assignments["assignments"]
        }
        self.planning_node_map = {
            (int(node["global_id"]), int(node["level"])): int(node["planning_id"])
            for node in self.global_graph.get("planning_nodes", [])
        }
        self.condition_id_map = {
            int(node["global_id"]): int(node.get("condition_id", node["global_id"]))
            for node in self.global_graph.get("global_entries", self.global_graph.get("global_nodes", []))
        }
        if self.global_graph.get("planning_edges") is not None:
            canonical_edges = list(self.global_graph.get("initialization_edges", [])) + list(
                self.global_graph.get("planning_edges", [])
            )
            self.global_edge_map = {
                (
                    int(edge["guide_global_id"]),
                    int(edge["guide_level"]),
                    int(edge["target_global_id"]),
                    int(edge["target_level"]),
                    int(edge["operator_id"]),
                ): int(edge["global_edge_id"])
                for edge in canonical_edges
            }
        else:
            self.global_edge_map = {
                (int(edge["guide_global_id"]), int(edge["target_global_id"]), int(edge["operator_id"])): int(edge["global_edge_id"])
                for edge in self.global_graph["global_edges"]
            }
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
        target_idx = int(edge["target"])
        target_checkpoint = record["checkpoints"][target_idx]
        target = CheckpointState.from_dict(target_checkpoint["state"])
        target_gid = self.assignment_map[(rec_idx, target_idx)]

        guide_idx = edge.get("guide")
        if guide_idx is None:
            guide = None
            guide_hash = ""
            guide_gid = -1
        else:
            guide_idx = int(guide_idx)
            guide_checkpoint = record["checkpoints"][guide_idx]
            guide = CheckpointState.from_dict(guide_checkpoint["state"])
            guide_hash = str(guide_checkpoint["hash"])
            guide_gid = self.assignment_map[(rec_idx, guide_idx)]

        if self.global_graph.get("planning_edges") is not None:
            edge_key = (
                guide_gid,
                int(edge["guide_level"]),
                target_gid,
                int(edge["target_level"]),
                int(edge["operator_id"]),
            )
            global_edge_id = self.global_edge_map[edge_key]
            target_planning_id = self.planning_node_map[(target_gid, int(edge["target_level"]))]
            guide_planning_id = (
                -1
                if guide_gid < 0
                else self.planning_node_map[(guide_gid, int(edge["guide_level"]))]
            )
        else:
            global_edge_id = self.global_edge_map[(guide_gid, target_gid, int(edge["operator_id"]))]
            target_planning_id = target_gid
            guide_planning_id = guide_gid
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
            "target_global_id": torch.tensor(target_gid, dtype=torch.long),
            "guide_global_id": torch.tensor(guide_gid, dtype=torch.long),
            "target_condition_id": torch.tensor(self.condition_id_map[target_gid], dtype=torch.long),
            "guide_condition_id": torch.tensor(
                -1 if guide_gid < 0 else self.condition_id_map[guide_gid], dtype=torch.long
            ),
            "global_edge_id": torch.tensor(global_edge_id, dtype=torch.long),
            "target_planning_id": torch.tensor(target_planning_id, dtype=torch.long),
            "guide_planning_id": torch.tensor(guide_planning_id, dtype=torch.long),
            "target_to_guide": target_to_guide,
            "overlap_mask": overlap_mask,
            "correspondence_mask": correspondence_mask,
            "boundary_expansion_mask": boundary_expansion_mask,
        }


def collate_global_edges(items: list[dict]) -> dict:
    batch = collate_edges(items)
    batch["target_global_id"] = torch.stack([item["target_global_id"] for item in items], dim=0)
    batch["guide_global_id"] = torch.stack([item["guide_global_id"] for item in items], dim=0)
    batch["target_condition_id"] = torch.stack([item["target_condition_id"] for item in items], dim=0)
    batch["guide_condition_id"] = torch.stack([item["guide_condition_id"] for item in items], dim=0)
    batch["global_edge_id"] = torch.stack([item["global_edge_id"] for item in items], dim=0)
    batch["target_planning_id"] = torch.stack([item["target_planning_id"] for item in items], dim=0)
    batch["guide_planning_id"] = torch.stack([item["guide_planning_id"] for item in items], dim=0)
    return batch
