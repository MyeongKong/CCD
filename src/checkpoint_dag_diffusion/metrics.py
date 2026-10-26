from __future__ import annotations

from collections import Counter
from typing import Iterable

import networkx as nx

from .checkpoint_state import CheckpointState
from .chemistry import rdkit_available, state_to_canonical_smiles
from .topology import checkpoint_consistent, contract_z1_to_z2


def ring_count(state: CheckpointState) -> int:
    return len(nx.cycle_basis(state.to_networkx()))


def structural_validity(state: CheckpointState) -> bool:
    if state.num_nodes == 0:
        return False
    for (u, v), label in state.edges.items():
        if u < 0 or v < 0 or u >= state.num_nodes or v >= state.num_nodes or u == v:
            return False
        if label not in state.bond_vocab:
            return False
    if state.num_nodes > 1 and not nx.is_connected(state.to_networkx()):
        return False
    return True


def molecular_validity(state: CheckpointState, *, require_rdkit: bool = False) -> bool:
    """Paper-facing validity: RDKit sanitization when RDKit is available."""

    if rdkit_available():
        return state_to_canonical_smiles(state) is not None
    if require_rdkit:
        raise ImportError("RDKit is required for paper-facing molecular validity evaluation.")
    return structural_validity(state)


def atom_histogram(state: CheckpointState) -> tuple[int, ...]:
    if not state.node_summaries:
        return tuple()
    width = len(state.node_summaries[0].hist)
    return tuple(sum(ns.hist[i] for ns in state.node_summaries) for i in range(width))


def summarize_samples(samples: Iterable[CheckpointState]) -> dict:
    sample_list = list(samples)
    hashes = [sample.canonical_hash() for sample in sample_list]
    ring_counts = [ring_count(sample) for sample in sample_list]
    structural = [structural_validity(sample) for sample in sample_list]
    smiles = [state_to_canonical_smiles(sample) for sample in sample_list] if rdkit_available() else []
    valid = [value is not None for value in smiles] if smiles else structural
    valid_smiles = [value for value in smiles if value is not None]
    return {
        "num_samples": len(sample_list),
        "structural_validity": sum(structural) / max(1, len(structural)),
        "rdkit_validity": None if not smiles else sum(valid) / max(1, len(valid)),
        "validity": sum(valid) / max(1, len(valid)),
        "unique_fraction": (
            len(set(valid_smiles)) / max(1, len(valid_smiles))
            if valid_smiles
            else len(set(hashes)) / max(1, len(hashes))
        ),
        "ring_count_hist": dict(Counter(ring_counts)),
    }


def frechet_chemnet_distance(
    generated: Iterable[CheckpointState],
    reference: Iterable[CheckpointState],
    *,
    device: str = "cpu",
    n_jobs: int = 1,
) -> float:
    """Compute FCD on valid SMILES without deduplicating either collection."""

    if not rdkit_available():
        raise ImportError("RDKit is required to compute Fréchet ChemNet Distance.")
    try:
        from fcd_torch import FCD  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional evaluation dependency
        raise ImportError("Install the optional evaluation dependency with `pip install fcd_torch`.") from exc
    generated_smiles = [value for state in generated if (value := state_to_canonical_smiles(state)) is not None]
    reference_smiles = [value for state in reference if (value := state_to_canonical_smiles(state)) is not None]
    if len(generated_smiles) < 2 or len(reference_smiles) < 2:
        raise ValueError("FCD requires at least two valid generated and two valid reference molecules.")
    evaluator = FCD(device=device, n_jobs=int(n_jobs))
    return float(evaluator(ref=reference_smiles, gen=generated_smiles))


def checkpoint_consistency_rate(z1_samples: list[CheckpointState], z2_guides: list[CheckpointState]) -> float:
    if not z1_samples:
        return 0.0
    values = [checkpoint_consistent(child, guide) for child, guide in zip(z1_samples, z2_guides)]
    return sum(bool(x) for x in values) / len(values)


def project_and_compare_z2(z1: CheckpointState, z2: CheckpointState) -> dict:
    projected = contract_z1_to_z2(z1)
    return {
        "projected_hash": projected.canonical_hash(),
        "guide_hash": z2.canonical_hash(),
        "consistent": projected.canonical_hash() == z2.canonical_hash(),
    }
