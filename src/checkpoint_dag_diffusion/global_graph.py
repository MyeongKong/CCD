from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from .artifacts import (
    ASSIGNMENTS_FORMAT,
    CERTIFIED_OBJECT_FORMAT,
    artifact_fingerprint,
    load_assignments,
    load_corpus_manifest,
    load_corpus_records as load_v2_corpus_records,
    validate_assignments_against_records,
    validate_global_graph_payload,
)
from .certified_object import certified_object_from_state
from .checkpoint_state import CheckpointState
from .dag_topology import CorrespondenceMap, transfer_correspondence
from .metrics import ring_count
from .tensorize import TensorSpec


def load_corpus_records(corpus_dir: str | Path, split: str = "train") -> list[dict[str, Any]]:
    return load_v2_corpus_records(corpus_dir, split)


def _spec_from_manifest(corpus_dir: str | Path) -> TensorSpec:
    manifest = load_corpus_manifest(corpus_dir)
    return TensorSpec(
        atom_vocab=tuple(manifest["atom_vocab"]),
        bond_vocab=tuple(manifest["bond_vocab"]),
        max_nodes=int(manifest["max_nodes"]),
    )


def _certified_object(
    checkpoint: dict[str, Any],
    state: CheckpointState,
    level: int | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    existing = checkpoint.get("certified_object")
    if isinstance(existing, dict) and existing.get("certificate_hash") and str(existing.get("format", "")).endswith(".v2"):
        return existing
    return certified_object_from_state(
        state,
        level=level,
        mode=mode,
        radius=None if checkpoint.get("radius") is None else int(checkpoint["radius"]),
    ).to_dict()


def _source_key(record: dict[str, Any]) -> str:
    if record.get("source_id"):
        return str(record["source_id"])
    source = record.get("source_graph", {})
    if isinstance(source, dict) and source.get("source_id"):
        return str(source["source_id"])
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()


def _semantic_label(cert: dict[str, Any]) -> str:
    semantic = cert.get("semantic", {})
    family = semantic.get("topology_family", "unknown")
    mode = semantic.get("mode", "unknown")
    level = semantic.get("level")
    slots = semantic.get("attachment_slots", 0)
    rings = semantic.get("ring_count", 0)
    return f"{family}|mode={mode}|level={level}|rings={rings}|slots={slots}"


def build_hash_assignments(corpus_dir: str | Path, out_dir: str | Path, split: str = "train") -> dict:
    records = load_corpus_records(corpus_dir, split)
    hash_to_id: dict[str, int] = {}
    assignments: list[dict[str, Any]] = []
    for rec_idx, record in enumerate(records):
        for cp_idx, checkpoint in enumerate(record["checkpoints"]):
            state = CheckpointState.from_dict(checkpoint["state"])
            cert = _certified_object(checkpoint, state, level=int(checkpoint["level"]), mode=str(checkpoint.get("mode", "unknown")))
            # The non-learned baseline merges complete certified objects, not
            # the construction carrier's legacy state hash.
            key = str(cert["certificate_hash"])
            if key not in hash_to_id:
                hash_to_id[key] = len(hash_to_id)
            assignments.append(
                {
                    "record_index": rec_idx,
                    "checkpoint_index": cp_idx,
                    "level": int(checkpoint["level"]),
                    "hash": key,
                    "certificate_hash": str(cert["certificate_hash"]),
                    "semantic_hash": str(cert["semantic_hash"]),
                    "global_id": hash_to_id[key],
                    "assignment_distance": 0.0,
                }
            )
    support = Counter(int(item["global_id"]) for item in assignments)
    payload = {
        "format": ASSIGNMENTS_FORMAT,
        "method": "canonical_hash",
        "split": split,
        "codebook_size": len(hash_to_id),
        "source_codebook_fingerprint": "canonical-hash-assignment",
        "hash_to_global_id": hash_to_id,
        "assignments": assignments,
        "summary": {
            "num_assignments": len(assignments),
            "active_codes": len(hash_to_id),
            "active_code_fraction": 1.0,
            "code_entropy": 0.0,
            "support_counts": dict(sorted((str(k), int(v)) for k, v in support.items())),
        },
    }
    payload["artifact_fingerprint"] = artifact_fingerprint(payload)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "global_assignments.pt"
    torch.save(payload, path)
    (out / "global_assignments_summary.json").write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    return {"assignments": str(path), "summary": payload["summary"]}


def _node_statistics(state: CheckpointState) -> dict[str, int]:
    return {
        "num_nodes": state.num_nodes,
        "num_edges": len(state.edges),
        "ring_count": ring_count(state),
        "core_nodes": sum(bool(v) for v in state.core_flags),
        "ring_nodes": sum(bool(v) for v in state.ring_flags),
    }


def _prototype_rank(distance: float, record_index: int, checkpoint_index: int) -> tuple[float, int, int]:
    """Deterministic rank used to select an explicit representative object."""

    return (float(distance), int(record_index), int(checkpoint_index))


def _update_prototype(
    node: dict[str, Any],
    *,
    record_index: int,
    checkpoint_index: int,
    checkpoint: dict[str, Any],
    certificate: dict[str, Any],
    statistics: dict[str, int],
    assignment_distance: float,
) -> None:
    """Keep the assigned object nearest to the code as the representative.

    Hash-based assignments have distance zero, in which case the record and
    checkpoint indices provide a stable tie break.
    """

    rank = _prototype_rank(assignment_distance, record_index, checkpoint_index)
    current = tuple(node.get("_prototype_rank", (math.inf, math.inf, math.inf)))
    if rank >= current:
        return
    node.update(
        {
            "_prototype_rank": rank,
            "prototype_record_index": int(record_index),
            "prototype_checkpoint_index": int(checkpoint_index),
            "prototype_state": checkpoint["state"],
            "prototype_hash": str(checkpoint["hash"]),
            "prototype_certificate": certificate,
            "prototype_certificate_hash": str(certificate["certificate_hash"]),
            "prototype_semantic_hash": str(certificate["semantic_hash"]),
            "prototype_assignment_distance": float(assignment_distance),
            "semantic_label": _semantic_label(certificate),
            "stats": statistics,
        }
    )


def validate_level_augmented_dag(graph: dict[str, Any]) -> bool:
    """Validate the canonical planning graph over ``(C_k, level)`` states.

    Codebook assignments may map checkpoints at several levels to the same
    code.  The code-only projection can therefore contain self-loops or cycles;
    the level-augmented planning graph must not.
    """

    planning_nodes = graph.get("planning_nodes", [])
    planning_edges = graph.get("planning_edges", [])
    node_by_id: dict[int, dict[str, Any]] = {}
    state_keys: set[tuple[int, int]] = set()
    for node in planning_nodes:
        planning_id = int(node["planning_id"])
        state_key = (int(node["global_id"]), int(node["level"]))
        if planning_id in node_by_id:
            raise ValueError(f"Duplicate planning_id {planning_id}.")
        if state_key in state_keys:
            raise ValueError(f"Duplicate level-augmented planning state {state_key}.")
        node_by_id[planning_id] = node
        state_keys.add(state_key)

    dag = nx.DiGraph()
    dag.add_nodes_from(node_by_id)
    for edge in planning_edges:
        guide_id = int(edge["guide_planning_id"])
        target_id = int(edge["target_planning_id"])
        if guide_id not in node_by_id or target_id not in node_by_id:
            raise ValueError(f"Planning edge references an unknown node: {guide_id}->{target_id}.")
        guide_level = int(edge["guide_level"])
        target_level = int(edge["target_level"])
        if int(node_by_id[guide_id]["level"]) != guide_level:
            raise ValueError(f"Guide level mismatch for planning node {guide_id}.")
        if int(node_by_id[target_id]["level"]) != target_level:
            raise ValueError(f"Target level mismatch for planning node {target_id}.")
        if guide_level <= target_level:
            raise ValueError(
                "Global planning edges must strictly decrease the contraction "
                f"level, got {guide_level}->{target_level}."
            )
        delta = int(edge.get("delta", guide_level - target_level))
        if delta != guide_level - target_level:
            raise ValueError(
                f"Planning-edge delta mismatch for {guide_id}->{target_id}: "
                f"{delta} != {guide_level - target_level}."
            )
        relation = edge.get("relation", {})
        for key, expected in (
            ("guide_level", guide_level),
            ("target_level", target_level),
            ("delta", delta),
            ("operator_id", int(edge["operator_id"])),
        ):
            if key in relation and int(relation[key]) != expected:
                raise ValueError(
                    f"Planning-edge relation field {key!r} disagrees with its canonical edge."
            )
        relation = edge.get("relation", {})
        correspondence = relation.get("correspondence", {})
        compatible = bool(edge.get("correspondence_compatible", bool(correspondence)))
        if not compatible and correspondence:
            raise ValueError("An incompatible planning edge must not retain a hard-copy correspondence.")
        if correspondence:
            guide_nodes = len(node_by_id[guide_id]["prototype_state"]["node_summaries"])
            target_nodes = len(node_by_id[target_id]["prototype_state"]["node_summaries"])
            for target, guide in correspondence.get("preserved_target_to_guide", {}).items():
                if not (0 <= int(target) < target_nodes and 0 <= int(guide) < guide_nodes):
                    raise ValueError("Planning-edge correspondence is outside representative node bounds.")
        dag.add_edge(guide_id, target_id)

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The level-augmented global checkpoint graph is cyclic.")

    for edge in graph.get("initialization_edges", []):
        target_id = int(edge["target_planning_id"])
        if target_id not in node_by_id:
            raise ValueError(f"Initialization edge references unknown target node {target_id}.")
        target_level = int(edge["target_level"])
        guide_level = int(edge["guide_level"])
        if int(node_by_id[target_id]["level"]) != target_level:
            raise ValueError(f"Target level mismatch for initialization node {target_id}.")
        if guide_level <= target_level:
            raise ValueError("Noise-to-checkpoint metadata must also decrease the synthetic guide level.")
        delta = int(edge.get("delta", guide_level - target_level))
        if delta != guide_level - target_level:
            raise ValueError(
                f"Initialization-edge delta mismatch for target {target_id}: "
                f"{delta} != {guide_level - target_level}."
            )
        relation = edge.get("relation", {})
        for key, expected in (
            ("guide_level", guide_level),
            ("target_level", target_level),
            ("delta", delta),
            ("operator_id", int(edge["operator_id"])),
        ):
            if key in relation and int(relation[key]) != expected:
                raise ValueError(
                    f"Initialization-edge relation field {key!r} disagrees with its canonical edge."
                )
    return True


def build_global_graph(
    corpus_dir: str | Path,
    assignments_path: str | Path,
    out_dir: str | Path,
    split: str = "train",
    max_support_examples: int = 32,
) -> dict:
    records = load_corpus_records(corpus_dir, split)
    assignment_payload = load_assignments(assignments_path)
    if str(assignment_payload["split"]) != str(split):
        raise ValueError(
            f"Assignment split {assignment_payload['split']!r} does not match requested corpus split {split!r}."
        )
    validate_assignments_against_records(
        assignment_payload,
        records,
        source=f"global assignments {Path(assignments_path)}",
    )
    assignment_map: dict[tuple[int, int], dict[str, Any]] = {}
    for item in assignment_payload["assignments"]:
        assignment_map[(int(item["record_index"]), int(item["checkpoint_index"]))] = item

    # ``nodes`` are reusable code-level entries C_k.  ``planning_nodes_by_key``
    # are the actual global-DAG states (C_k, level).  Keeping both prevents a
    # cross-level code collision from creating a planning self-loop.
    nodes: dict[int, dict[str, Any]] = {}
    planning_nodes_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    hash_counts: dict[int, Counter] = defaultdict(Counter)
    level_counts: dict[int, Counter] = defaultdict(Counter)
    ring_counts: dict[int, Counter] = defaultdict(Counter)
    certificate_counts: dict[int, Counter] = defaultdict(Counter)
    semantic_counts: dict[int, Counter] = defaultdict(Counter)
    source_counts: dict[int, Counter] = defaultdict(Counter)
    distance_sum: Counter[int] = Counter()
    planning_hash_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    planning_certificate_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    planning_semantic_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    planning_source_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    planning_distance_sum: Counter[tuple[int, int]] = Counter()

    for item in assignment_payload["assignments"]:
        rec_idx = int(item["record_index"])
        cp_idx = int(item["checkpoint_index"])
        gid = int(item["global_id"])
        checkpoint = records[rec_idx]["checkpoints"][cp_idx]
        level = int(checkpoint["level"])
        if int(item.get("level", level)) != level:
            raise ValueError(
                f"Assignment level mismatch at record {rec_idx}, checkpoint {cp_idx}: "
                f"{item.get('level')} != {level}."
            )
        state = CheckpointState.from_dict(checkpoint["state"])
        cert = _certified_object(checkpoint, state, level=level, mode=str(checkpoint.get("mode", "unknown")))
        stats = _node_statistics(state)
        distance = float(item.get("assignment_distance", 0.0))
        if gid not in nodes:
            nodes[gid] = {
                "global_id": gid,
                "support_count": 0,
                "support_examples": [],
            }
        node = nodes[gid]
        _update_prototype(
            node,
            record_index=rec_idx,
            checkpoint_index=cp_idx,
            checkpoint=checkpoint,
            certificate=cert,
            statistics=stats,
            assignment_distance=distance,
        )
        node["support_count"] += 1
        if len(node["support_examples"]) < max_support_examples:
            node["support_examples"].append({"record_index": rec_idx, "checkpoint_index": cp_idx})
        hash_counts[gid][str(checkpoint["hash"])] += 1
        level_counts[gid][level] += 1
        ring_counts[gid][stats["ring_count"]] += 1
        certificate_counts[gid][str(cert["certificate_hash"])] += 1
        semantic_counts[gid][str(cert["semantic_hash"])] += 1
        source_counts[gid][_source_key(records[rec_idx])] += 1
        distance_sum[gid] += distance

        planning_key = (gid, level)
        if planning_key not in planning_nodes_by_key:
            planning_nodes_by_key[planning_key] = {
                "global_id": gid,
                "level": level,
                "state_key": f"C{gid}@L{level}",
                "support_count": 0,
                "support_examples": [],
            }
        planning_node = planning_nodes_by_key[planning_key]
        _update_prototype(
            planning_node,
            record_index=rec_idx,
            checkpoint_index=cp_idx,
            checkpoint=checkpoint,
            certificate=cert,
            statistics=stats,
            assignment_distance=distance,
        )
        planning_node["support_count"] += 1
        if len(planning_node["support_examples"]) < max_support_examples:
            planning_node["support_examples"].append({"record_index": rec_idx, "checkpoint_index": cp_idx})
        planning_hash_counts[planning_key][str(checkpoint["hash"])] += 1
        planning_certificate_counts[planning_key][str(cert["certificate_hash"])] += 1
        planning_semantic_counts[planning_key][str(cert["semantic_hash"])] += 1
        planning_source_counts[planning_key][_source_key(records[rec_idx])] += 1
        planning_distance_sum[planning_key] += distance

    for gid, node in nodes.items():
        support = max(1, int(node["support_count"]))
        node["level_hist"] = dict(sorted((str(k), int(v)) for k, v in level_counts[gid].items()))
        node["ring_count_hist"] = dict(sorted((str(k), int(v)) for k, v in ring_counts[gid].items()))
        node["hash_hist_top"] = dict((str(k), int(v)) for k, v in hash_counts[gid].most_common(8))
        node["certificate_hash_hist_top"] = dict((str(k), int(v)) for k, v in certificate_counts[gid].most_common(8))
        node["semantic_hash_hist_top"] = dict((str(k), int(v)) for k, v in semantic_counts[gid].most_common(8))
        node["hash_purity"] = max(hash_counts[gid].values()) / support if hash_counts[gid] else 0.0
        node["certificate_purity"] = max(certificate_counts[gid].values()) / support if certificate_counts[gid] else 0.0
        node["topology_purity"] = max(semantic_counts[gid].values()) / support if semantic_counts[gid] else 0.0
        node["chemical_diversity"] = 1.0 - (max(source_counts[gid].values()) / support if source_counts[gid] else 1.0)
        node["mean_assignment_distance"] = float(distance_sum[gid] / support)
        node.pop("_prototype_rank", None)

    planning_key_to_id = {
        key: planning_id for planning_id, key in enumerate(sorted(planning_nodes_by_key))
    }
    planning_nodes: list[dict[str, Any]] = []
    for planning_key in sorted(planning_nodes_by_key):
        planning_node = planning_nodes_by_key[planning_key]
        support = max(1, int(planning_node["support_count"]))
        planning_node["planning_id"] = planning_key_to_id[planning_key]
        planning_node["hash_hist_top"] = dict(
            (str(k), int(v)) for k, v in planning_hash_counts[planning_key].most_common(8)
        )
        planning_node["certificate_hash_hist_top"] = dict(
            (str(k), int(v)) for k, v in planning_certificate_counts[planning_key].most_common(8)
        )
        planning_node["semantic_hash_hist_top"] = dict(
            (str(k), int(v)) for k, v in planning_semantic_counts[planning_key].most_common(8)
        )
        planning_node["hash_purity"] = max(planning_hash_counts[planning_key].values()) / support
        planning_node["certificate_purity"] = max(planning_certificate_counts[planning_key].values()) / support
        planning_node["topology_purity"] = max(planning_semantic_counts[planning_key].values()) / support
        planning_node["chemical_diversity"] = 1.0 - (
            max(planning_source_counts[planning_key].values()) / support
            if planning_source_counts[planning_key]
            else 1.0
        )
        planning_node["mean_assignment_distance"] = float(planning_distance_sum[planning_key] / support)
        planning_node.pop("_prototype_rank", None)
        planning_nodes.append(planning_node)

    for gid, node in nodes.items():
        node["planning_node_ids"] = [
            planning_key_to_id[key] for key in sorted(planning_key_to_id) if key[0] == gid
        ]

    # ``edge_acc`` retains the v1 code-only projection for consumers that have
    # not yet migrated.  ``planning_edge_acc`` is the canonical acyclic graph.
    edge_acc: dict[tuple[int, int, int], dict[str, Any]] = {}
    planning_edge_acc: dict[tuple[int, int, int], dict[str, Any]] = {}
    initialization_edge_acc: dict[tuple[int, int, int], dict[str, Any]] = {}
    coarsest_prior = Counter()  # legacy code-only prior
    coarsest_planning_prior = Counter()
    local_edge_observations = 0
    for rec_idx, record in enumerate(records):
        for edge in record["dag_edges"]:
            target_idx = int(edge["target"])
            target_gid = int(assignment_map[(rec_idx, target_idx)]["global_id"])
            target_level = int(edge["target_level"])
            target_planning_id = planning_key_to_id[(target_gid, target_level)]
            guide_idx = edge.get("guide")
            if guide_idx is None:
                coarsest_prior[target_gid] += 1
                coarsest_planning_prior[target_planning_id] += 1
                guide_gid = -1
                guide_level = int(edge["guide_level"])
                init_key = (target_planning_id, guide_level, int(edge["operator_id"]))
                if init_key not in initialization_edge_acc:
                    initialization_edge_acc[init_key] = {
                        "guide_planning_id": -1,
                        "target_planning_id": target_planning_id,
                        "guide_global_id": -1,
                        "target_global_id": target_gid,
                        "guide_level": guide_level,
                        "target_level": target_level,
                        "delta": int(edge["delta"]),
                        "operator_id": int(edge["operator_id"]),
                        "count": 0,
                        "support_examples": [],
                        "relation": dict(edge),
                    }
                init_item = initialization_edge_acc[init_key]
                init_item["count"] += 1
                if len(init_item["support_examples"]) < max_support_examples:
                    init_item["support_examples"].append(
                        {"record_index": rec_idx, "edge": dict(edge), "target_checkpoint_index": target_idx}
                    )
            else:
                guide_idx = int(guide_idx)
                guide_gid = int(assignment_map[(rec_idx, guide_idx)]["global_id"])
                guide_level = int(edge["guide_level"])
                if guide_level <= target_level:
                    raise ValueError(
                        f"Local transition must strictly decrease level, got {guide_level}->{target_level} "
                        f"at record {rec_idx}."
                    )
                guide_planning_id = planning_key_to_id[(guide_gid, guide_level)]
                planning_key = (guide_planning_id, target_planning_id, int(edge["operator_id"]))
                if planning_key not in planning_edge_acc:
                    planning_edge_acc[planning_key] = {
                        "guide_planning_id": guide_planning_id,
                        "target_planning_id": target_planning_id,
                        "guide_global_id": guide_gid,
                        "target_global_id": target_gid,
                        "guide_level": guide_level,
                        "target_level": target_level,
                        "delta": int(edge["delta"]),
                        "operator_id": int(edge["operator_id"]),
                        "count": 0,
                        "support_examples": [],
                        "relation": dict(edge),
                    }
                planning_item = planning_edge_acc[planning_key]
                planning_item["count"] += 1
                if len(planning_item["support_examples"]) < max_support_examples:
                    planning_item["support_examples"].append(
                        {
                            "record_index": rec_idx,
                            "edge": dict(edge),
                            "target_checkpoint_index": target_idx,
                            "guide_checkpoint_index": guide_idx,
                        }
                    )
            key = (guide_gid, target_gid, int(edge["operator_id"]))
            if key not in edge_acc:
                edge_acc[key] = {
                    "guide_global_id": guide_gid,
                    "target_global_id": target_gid,
                    "operator_id": int(edge["operator_id"]),
                    "count": 0,
                    "delta_hist": Counter(),
                    "target_level_hist": Counter(),
                    "guide_level_hist": Counter(),
                    "support_examples": [],
                    "relation": dict(edge),
                }
            item = edge_acc[key]
            item["count"] += 1
            item["delta_hist"][int(edge["delta"])] += 1
            item["target_level_hist"][int(edge["target_level"])] += 1
            item["guide_level_hist"][int(edge["guide_level"])] += 1
            if len(item["support_examples"]) < max_support_examples:
                item["support_examples"].append(
                    {"record_index": rec_idx, "edge": dict(edge), "target_checkpoint_index": target_idx, "guide_checkpoint_index": guide_idx}
                )
            local_edge_observations += 1

    global_edges = []
    for edge_id, key in enumerate(sorted(edge_acc, key=lambda k: (k[0], k[1], k[2]))):
        item = edge_acc[key]
        global_edges.append(
            {
                "global_edge_id": edge_id,
                "guide_global_id": int(item["guide_global_id"]),
                "target_global_id": int(item["target_global_id"]),
                "operator_id": int(item["operator_id"]),
                "count": int(item["count"]),
                "delta_hist": dict(sorted((str(k), int(v)) for k, v in item["delta_hist"].items())),
                "target_level_hist": dict(sorted((str(k), int(v)) for k, v in item["target_level_hist"].items())),
                "guide_level_hist": dict(sorted((str(k), int(v)) for k, v in item["guide_level_hist"].items())),
                "support_examples": item["support_examples"],
                "relation": item["relation"],
            }
        )

    planning_edges: list[dict[str, Any]] = []
    for planning_edge_id, key in enumerate(sorted(planning_edge_acc)):
        item = planning_edge_acc[key]
        # Choose one observed transition jointly close to both code vectors,
        # then transfer its local mu into the independently selected global
        # representative orderings.  If either endpoint is not isomorphic,
        # the hard copy map is omitted instead of applying invalid indices.
        representative_example = min(
            item["support_examples"],
            key=lambda example: (
                float(
                    assignment_map[(int(example["record_index"]), int(example["guide_checkpoint_index"]))].get(
                        "assignment_distance", 0.0
                    )
                )
                + float(
                    assignment_map[(int(example["record_index"]), int(example["target_checkpoint_index"]))].get(
                        "assignment_distance", 0.0
                    )
                ),
                int(example["record_index"]),
                int(example["guide_checkpoint_index"]),
                int(example["target_checkpoint_index"]),
            ),
        )
        support_record = records[int(representative_example["record_index"])]
        support_guide = CheckpointState.from_dict(
            support_record["checkpoints"][int(representative_example["guide_checkpoint_index"])]["state"]
        )
        support_target = CheckpointState.from_dict(
            support_record["checkpoints"][int(representative_example["target_checkpoint_index"])]["state"]
        )
        representative_guide = CheckpointState.from_dict(
            planning_nodes[int(item["guide_planning_id"])]["prototype_state"]
        )
        representative_target = CheckpointState.from_dict(
            planning_nodes[int(item["target_planning_id"])]["prototype_state"]
        )
        local_correspondence = CorrespondenceMap.from_dict(
            representative_example["edge"].get("correspondence")
        )
        transferred = (
            None
            if local_correspondence is None
            else transfer_correspondence(
                local_correspondence,
                support_finer=support_target,
                support_coarser=support_guide,
                representative_finer=representative_target,
                representative_coarser=representative_guide,
            )
        )
        planning_edges.append(
            {
                "planning_edge_id": planning_edge_id,
                **item,
                "relation": {
                    "guide_level": int(item["guide_level"]),
                    "target_level": int(item["target_level"]),
                    "delta": int(item["delta"]),
                    "operator_id": int(item["operator_id"]),
                    "correspondence": {} if transferred is None else transferred.to_dict(),
                },
                "correspondence_compatible": transferred is not None,
                "representative_support_example": representative_example,
            }
        )

    initialization_edges: list[dict[str, Any]] = []
    for initialization_edge_id, key in enumerate(sorted(initialization_edge_acc)):
        item = initialization_edge_acc[key]
        initialization_edges.append(
            {
                "initialization_edge_id": initialization_edge_id,
                **item,
                "relation": {
                    "guide_level": int(item["guide_level"]),
                    "target_level": int(item["target_level"]),
                    "delta": int(item["delta"]),
                    "operator_id": int(item["operator_id"]),
                },
            }
        )

    # One denoiser-side identifier is assigned to every canonical
    # level-augmented edge, including noise initialization edges.  This table is
    # intentionally distinct from codebook vectors and explicit C_k entries.
    for global_edge_id, edge in enumerate(initialization_edges + planning_edges):
        edge["global_edge_id"] = int(global_edge_id)
    num_condition_edges = len(initialization_edges) + len(planning_edges)

    spec = _spec_from_manifest(corpus_dir)
    global_entries = [nodes[gid] for gid in sorted(nodes)]
    condition_id_by_global = {
        int(entry["global_id"]): condition_id
        for condition_id, entry in enumerate(global_entries)
    }
    for entry in global_entries:
        entry["condition_id"] = int(condition_id_by_global[int(entry["global_id"])])
    for planning_node in planning_nodes:
        planning_node["condition_id"] = int(
            condition_id_by_global[int(planning_node["global_id"])]
        )
    coarsest_state_prior = [
        {
            "planning_id": int(planning_id),
            "global_id": int(planning_nodes[planning_id]["global_id"]),
            "level": int(planning_nodes[planning_id]["level"]),
            "count": int(count),
        }
        for planning_id, count in sorted(coarsest_planning_prior.items())
    ]
    payload = {
        "format": "checkpoint_dag_diffusion.global_graph.v2",
        "certified_object_format": CERTIFIED_OBJECT_FORMAT,
        "method": assignment_payload.get("method", "unknown"),
        "split": split,
        "source_assignments_fingerprint": str(assignment_payload["artifact_fingerprint"]),
        "source_codebook_fingerprint": str(assignment_payload["source_codebook_fingerprint"]),
        "num_global_nodes": len(nodes),
        "num_global_conditions": len(condition_id_by_global),
        "num_global_edges": len(global_edges),
        "num_planning_nodes": len(planning_nodes),
        "num_planning_edges": len(planning_edges),
        "num_initialization_edges": len(initialization_edges),
        "num_condition_edges": num_condition_edges,
        "num_local_edge_observations": local_edge_observations,
        # Canonical code-level entries and level-augmented planning graph.
        "global_entries": global_entries,
        "planning_nodes": planning_nodes,
        "planning_edges": planning_edges,
        "initialization_edges": initialization_edges,
        "coarsest_planning_prior": dict(
            sorted((str(k), int(v)) for k, v in coarsest_planning_prior.items())
        ),
        "coarsest_state_prior": coarsest_state_prior,
        # Backward-compatible v1 fields.  These are a code-only projection and
        # must not be used as the canonical planning DAG.
        "global_nodes": global_entries,
        "global_edges": global_edges,
        "coarsest_prior": dict(sorted((str(k), int(v)) for k, v in coarsest_prior.items())),
        "spec": {
            "atom_vocab": list(spec.atom_vocab),
            "bond_vocab": list(spec.bond_vocab),
            "max_nodes": spec.max_nodes,
        },
    }
    validate_level_augmented_dag(payload)
    payload["planning_is_dag"] = True
    payload["artifact_fingerprint"] = artifact_fingerprint(payload)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "global_graph.pt"
    torch.save(payload, graph_path)
    manifest = summarize_global_graph(payload)
    (out / "global_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"global_graph": str(graph_path), "manifest": manifest}


def summarize_global_graph(graph: dict) -> dict:
    global_entries = graph.get("global_entries", graph.get("global_nodes", []))
    planning_edges = graph.get("planning_edges")
    canonical_edges = planning_edges if planning_edges is not None else graph.get("global_edges", [])
    supports = [int(node["support_count"]) for node in global_entries]
    edge_counts = [int(edge["count"]) for edge in canonical_edges]
    topology_purities = [float(node.get("topology_purity", node.get("hash_purity", 0.0))) for node in global_entries]
    certificate_purities = [float(node.get("certificate_purity", node.get("hash_purity", 0.0))) for node in global_entries]
    chemical_diversities = [float(node.get("chemical_diversity", 0.0)) for node in global_entries]
    support_hist = Counter(supports)
    edge_count_hist = Counter(edge_counts)
    out_degree = Counter()
    outgoing_weight = Counter()
    guide_key = "guide_planning_id" if planning_edges is not None else "guide_global_id"
    for edge in canonical_edges:
        guide = int(edge[guide_key])
        if guide >= 0:
            out_degree[guide] += 1
            outgoing_weight[guide] += int(edge["count"])
    transition_entropies = []
    outgoing_by_guide: dict[int, list[int]] = defaultdict(list)
    for edge in canonical_edges:
        guide = int(edge[guide_key])
        if guide >= 0:
            outgoing_by_guide[guide].append(int(edge["count"]))
    for counts in outgoing_by_guide.values():
        total = sum(counts)
        if total > 0:
            transition_entropies.append(-sum((c / total) * math.log(c / total) for c in counts if c > 0))
    return {
        "format": graph.get("format"),
        "method": graph.get("method"),
        "split": graph.get("split"),
        "num_global_nodes": int(graph["num_global_nodes"]),
        "num_global_conditions": int(graph.get("num_global_conditions", graph["num_global_nodes"])),
        "num_global_edges": int(graph["num_global_edges"]),
        "num_planning_nodes": int(graph.get("num_planning_nodes", graph["num_global_nodes"])),
        "num_planning_edges": int(graph.get("num_planning_edges", graph["num_global_edges"])),
        "num_initialization_edges": int(graph.get("num_initialization_edges", 0)),
        "planning_is_dag": graph.get("planning_is_dag"),
        "num_local_edge_observations": int(graph["num_local_edge_observations"]),
        "coarsest_prior_size": len(graph.get("coarsest_prior", {})),
        "coarsest_planning_prior_size": len(graph.get("coarsest_planning_prior", {})),
        "node_support_hist": dict(sorted((str(k), int(v)) for k, v in support_hist.items())),
        "edge_count_hist": dict(sorted((str(k), int(v)) for k, v in edge_count_hist.items())),
        "mean_out_degree": sum(out_degree.values()) / max(1, len(out_degree)),
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "mean_topology_purity": sum(topology_purities) / max(1, len(topology_purities)),
        "mean_certificate_purity": sum(certificate_purities) / max(1, len(certificate_purities)),
        "mean_chemical_diversity": sum(chemical_diversities) / max(1, len(chemical_diversities)),
        "mean_transition_entropy": sum(transition_entropies) / max(1, len(transition_entropies)),
    }


def load_global_graph(path: str | Path) -> dict:
    path = Path(path)
    graph = validate_global_graph_payload(
        torch.load(path, map_location="cpu"),
        source=f"global graph {path}",
    )
    validate_level_augmented_dag(graph)
    return graph


def code_count(graph: dict) -> int:
    if "num_global_conditions" in graph:
        return int(graph["num_global_conditions"])
    if not graph.get("global_nodes"):
        return 0
    # v1 artifacts indexed the embedding table directly by sparse code id.
    return max(int(node["global_id"]) for node in graph["global_nodes"]) + 1
