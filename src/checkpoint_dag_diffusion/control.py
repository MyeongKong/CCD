from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactVersionError
from .chemistry import smiles_to_molecule_graph
from .dag_topology import DAGCheckpoint, build_checkpoint_dag
from .global_codebook import load_codebook_model
from .global_graph import load_global_graph, validate_level_augmented_dag
from .tensorize import certified_object_to_tensors


@dataclass(frozen=True)
class ScaffoldSelection:
    scaffold_smiles: str
    requested_level: int
    requested_mode: str
    selected_global_id: int
    selected_level: int
    planning_id: int
    embedding_distance: float
    requested_certificate_hash: str
    selected_certificate_hash: str
    compatibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _choose_local_checkpoint(
    checkpoints: tuple[DAGCheckpoint, ...],
    *,
    level: int | None,
    mode: str | None,
) -> DAGCheckpoint:
    candidates = list(checkpoints)
    if level is not None:
        candidates = [checkpoint for checkpoint in candidates if int(checkpoint.level) == int(level)]
    if mode is not None:
        candidates = [checkpoint for checkpoint in candidates if str(checkpoint.mode) == str(mode)]
    if not candidates:
        raise ValueError(f"No scaffold checkpoint matches level={level}, mode={mode!r}.")
    return max(candidates, key=lambda checkpoint: int(checkpoint.level))


def _boundary_signature(certified_object: dict[str, Any]) -> dict[str, Any]:
    slots = certified_object.get("boundary_slots", [])
    multiplicities = sorted(
        int(slot.get("multiplicity", slot.get("boundary_count", len(slot.get("boundary_atoms", [])) or 1)))
        for slot in slots
    )
    bond_labels = sorted(
        str(label)
        for slot in slots
        for label in slot.get("boundary_bond_labels", [slot.get("bond_label", "single")])
    )
    return {
        "slot_count": len(slots),
        "multiplicities": multiplicities,
        "bond_labels": bond_labels,
    }


def checkpoint_compatibility(
    requested: dict[str, Any],
    candidate: dict[str, Any],
    *,
    exact_quotient: bool = False,
    require_boundary: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Test the topology and attachment-interface filter from the paper."""

    requested_semantic = requested.get("semantic", {})
    candidate_semantic = candidate.get("semantic", {})
    topology_fields = ("topology_family", "ring_count", "ring_size_hist", "core_empty")
    topology_match = all(requested_semantic.get(key) == candidate_semantic.get(key) for key in topology_fields)
    quotient_match = requested.get("quotient_graph_hash") == candidate.get("quotient_graph_hash")
    requested_boundary = _boundary_signature(requested)
    candidate_boundary = _boundary_signature(candidate)
    boundary_match = requested_boundary == candidate_boundary
    compatible = topology_match and (quotient_match if exact_quotient else True)
    if require_boundary:
        compatible = compatible and boundary_match
    return compatible, {
        "topology_match": topology_match,
        "quotient_match": quotient_match,
        "boundary_match": boundary_match,
        "requested_boundary": requested_boundary,
        "candidate_boundary": candidate_boundary,
        "exact_quotient_required": bool(exact_quotient),
        "boundary_required": bool(require_boundary),
    }


@torch.no_grad()
def select_scaffold_checkpoint(
    *,
    scaffold_smiles: str,
    codebook_path: str | Path,
    global_graph_path: str | Path,
    level: int | None = None,
    mode: str | None = None,
    exact_quotient: bool = False,
    require_boundary: bool = True,
    device: str | None = None,
) -> ScaffoldSelection:
    """Contract a scaffold and select its nearest compatible active C_k."""

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, spec, codebook = load_codebook_model(codebook_path, dev)
    graph = load_global_graph(global_graph_path)
    if str(graph["source_codebook_fingerprint"]) != str(codebook["artifact_fingerprint"]):
        raise ArtifactVersionError(
            "Scaffold codebook and global graph come from different artifact chains; "
            "supply the matching v2 artifacts."
        )
    if graph.get("planning_nodes"):
        validate_level_augmented_dag(graph)
    molecule = smiles_to_molecule_graph(scaffold_smiles, source_id=f"scaffold:{scaffold_smiles}")
    local_dag = build_checkpoint_dag(
        molecule,
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )
    checkpoint = _choose_local_checkpoint(local_dag.checkpoints, level=level, mode=mode)
    requested = checkpoint.to_dict()["certified_object"]
    tensors = certified_object_to_tensors(requested, spec)
    state = {key: value.unsqueeze(0).to(dev) for key, value in tensors.items()}
    embedding = model.encode(state).squeeze(0)

    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for node in graph.get("planning_nodes", []):
        candidate = node.get("prototype_certificate", {})
        if not candidate:
            continue
        compatible, details = checkpoint_compatibility(
            requested,
            candidate,
            exact_quotient=exact_quotient,
            require_boundary=require_boundary,
        )
        if not compatible:
            continue
        code_id = int(node["global_id"])
        if code_id < 0 or code_id >= model.codebook.size(0):
            continue
        distance = float(torch.sum((embedding - model.codebook[code_id]) ** 2).cpu())
        candidates.append((distance, node, details))
    if not candidates:
        raise ValueError(
            "No active global checkpoint has a compatible quotient topology "
            "and attachment boundary for the supplied scaffold."
        )
    distance, selected, details = min(
        candidates,
        key=lambda row: (row[0], int(row[1]["global_id"]), int(row[1]["level"])),
    )
    selected_certificate = selected["prototype_certificate"]
    return ScaffoldSelection(
        scaffold_smiles=scaffold_smiles,
        requested_level=int(checkpoint.level),
        requested_mode=str(checkpoint.mode),
        selected_global_id=int(selected["global_id"]),
        selected_level=int(selected["level"]),
        planning_id=int(selected["planning_id"]),
        embedding_distance=distance,
        requested_certificate_hash=str(requested["certificate_hash"]),
        selected_certificate_hash=str(selected_certificate["certificate_hash"]),
        compatibility=details,
    )
