from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from .certified_object import (
    certified_object_from_state,
    exact_certificate_match,
    normalized_summary_error,
    quotient_graph_match,
    semantic_certificate_match,
)
from .checkpoint_state import CheckpointState
from .dag_topology import build_checkpoint_dag
from .topology import MoleculeGraph


def checkpoint_state_to_molecule_graph(
    state: CheckpointState,
    *,
    source_id: str | None = None,
) -> MoleculeGraph:
    """Convert a decoded level-0 checkpoint into an atom-level graph.

    Recontraction is defined only for a concrete atom graph.  A decoded node
    that still represents multiple atoms is therefore a failed realization,
    not something that may be silently interpreted as one atom.
    """

    atom_symbols: list[str] = []
    charges: list[int] = []
    for index, summary in enumerate(state.node_summaries):
        if summary.atom_count != 1 or sum(summary.hist) != 1:
            raise ValueError(
                f"Decoded level-0 node {index} is unresolved: "
                f"atom_count={summary.atom_count}, hist_sum={sum(summary.hist)}."
            )
        active = [idx for idx, value in enumerate(summary.hist) if int(value) == 1]
        if len(active) != 1:
            raise ValueError(f"Decoded node {index} has no unique atom category.")
        atom_symbols.append(state.atom_vocab[active[0]])
        charges.append(int(summary.charge))

    graph = nx.Graph()
    graph.add_nodes_from(range(state.num_nodes))
    graph.add_edges_from(state.edges)
    cycle_edges: set[tuple[int, int]] = set()
    cycle_atoms: set[int] = set()
    for cycle in nx.cycle_basis(graph):
        cycle_atoms.update(cycle)
        for pos, left in enumerate(cycle):
            right = cycle[(pos + 1) % len(cycle)]
            cycle_edges.add((left, right) if left < right else (right, left))

    return MoleculeGraph(
        atom_symbols=tuple(atom_symbols),
        charges=tuple(charges),
        edges=dict(state.edges),
        ring_atoms=frozenset(cycle_atoms),
        ring_edges=frozenset(cycle_edges),
        original_atom_ids=tuple(range(state.num_nodes)),
        source_id=source_id,
    )


def _requested_contractor(certified_object: dict[str, Any]) -> tuple[int, str | None, int | None]:
    semantic = certified_object.get("semantic", {})
    level = semantic.get("level")
    if level is None:
        raise ValueError("Requested certified object does not specify a contraction level.")
    mode = semantic.get("mode")
    radius = semantic.get("radius")
    return (
        int(level),
        None if mode is None else str(mode),
        None if radius is None else int(radius),
    )


def recontract_and_compare(
    generated: CheckpointState,
    requested_objects: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    r"""Recontract ``generated`` and compare every requested path object.

    This is the implementation of
    :math:`\widehat O_m=O(C_{\ell_m}(\widehat G))`.  Missing levels, unresolved
    atom nodes, or contractor mismatches are returned as explicit failures.
    """

    requested = [dict(obj) for obj in requested_objects if isinstance(obj, dict)]
    if not requested:
        return {"recontractable": False, "error": "no_requested_objects", "comparisons": []}
    try:
        molecule = checkpoint_state_to_molecule_graph(generated, source_id="generated")
        dag = build_checkpoint_dag(
            molecule,
            atom_vocab=generated.atom_vocab,
            bond_vocab=generated.bond_vocab,
        )
    except Exception as exc:
        return {
            "recontractable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "comparisons": [
                {
                    "requested_level": obj.get("semantic", {}).get("level"),
                    "requested_mode": obj.get("semantic", {}).get("mode"),
                    "available": False,
                    "qmatch": False,
                    "semantic": False,
                    "exact": False,
                    "summary_error": None,
                }
                for obj in requested
            ],
        }

    by_level_mode = {(int(cp.level), str(cp.mode)): cp for cp in dag.checkpoints}
    by_contractor = {
        (str(cp.mode), None if cp.radius is None else int(cp.radius)): cp
        for cp in dag.checkpoints
    }
    by_level = {int(cp.level): cp for cp in dag.checkpoints}
    rows: list[dict[str, Any]] = []
    for target in requested:
        try:
            level, mode, radius = _requested_contractor(target)
        except ValueError as exc:
            rows.append(
                {
                    "requested_level": None,
                    "requested_mode": None,
                    "requested_radius": None,
                    "available": False,
                    "qmatch": False,
                    "semantic": False,
                    "exact": False,
                    "summary_error": None,
                    "error": str(exc),
                }
            )
            continue
        checkpoint = (
            by_contractor.get((mode, radius))
            if mode is not None and (radius is not None or mode in {"full", "pure_core"})
            else by_level_mode.get((level, mode))
            if mode is not None
            else by_level.get(level)
        )
        if checkpoint is None:
            rows.append(
                {
                    "requested_level": level,
                    "requested_mode": mode,
                    "requested_radius": radius,
                    "available": False,
                    "qmatch": False,
                    "semantic": False,
                    "exact": False,
                    "summary_error": None,
                    "error": "requested_level_not_recovered",
                }
            )
            continue
        recovered = certified_object_from_state(
            checkpoint.state,
            level=checkpoint.level,
            mode=checkpoint.mode,
            radius=checkpoint.radius,
        ).to_dict()
        rows.append(
            {
                "requested_level": level,
                "requested_mode": mode,
                "requested_radius": radius,
                "recovered_level": int(checkpoint.level),
                "recovered_mode": str(checkpoint.mode),
                "recovered_radius": None if checkpoint.radius is None else int(checkpoint.radius),
                "available": True,
                "qmatch": quotient_graph_match(recovered, target),
                "semantic": semantic_certificate_match(recovered, target),
                "exact": exact_certificate_match(recovered, target),
                "summary_error": normalized_summary_error(recovered, target),
                "requested_hashes": {
                    "quotient": str(target.get("quotient_graph_hash", "")),
                    "semantic": str(target.get("semantic_hash", "")),
                    "certificate": str(target.get("certificate_hash", "")),
                },
                "recovered_hashes": {
                    "quotient": str(recovered.get("quotient_graph_hash", "")),
                    "semantic": str(recovered.get("semantic_hash", "")),
                    "certificate": str(recovered.get("certificate_hash", "")),
                },
            }
        )
    return {
        "recontractable": True,
        "num_recovered_levels": len(dag.checkpoints),
        "comparisons": rows,
        "path_semantic": bool(rows) and all(bool(row["semantic"]) for row in rows),
        "path_exact": bool(rows) and all(bool(row["exact"]) for row in rows),
    }
