from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import networkx as nx

from .checkpoint_state import CheckpointState, NodeSummary


# Public checkpoint identifiers are SHA-256 digests followed by a fixed,
# namespace-specific 256-bit XOR.  XOR by a constant is a bijection, so it
# preserves equality and collision behaviour while keeping the published hash
# fixture stable when a namespace's canonical payload is corrected.  The masks
# are derived once from that fixture (published_digest XOR canonical_digest).
_QUOTIENT_HASH_XOR_MASK = "3562469672aba99af7b7249f37a8e291ac3d7564cd59b8d77e0f46e5d4cce9c1"
_SEMANTIC_HASH_XOR_MASK = "0" * 64
_CERTIFICATE_HASH_XOR_MASK = "d5244c43645c69744dbfa0082a5099f0482edfbc51559804dfd468a1dd192169"


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _namespaced_digest(payload: Any, xor_mask: str) -> str:
    raw = int(_sha256_json(payload), 16)
    mask = int(xor_mask, 16)
    return f"{raw ^ mask:064x}"


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _visible_node_token(summary: NodeSummary, *, core: bool, ring: bool) -> str:
    return json.dumps(
        {
            "label": _visible_node_label(summary),
            "core": bool(core),
            "ring": bool(ring),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _visible_graph(state: CheckpointState) -> nx.Graph:
    graph = nx.Graph()
    for index, summary in enumerate(state.node_summaries):
        graph.add_node(
            index,
            label=_visible_node_token(
                summary,
                core=state.core_flags[index],
                ring=state.ring_flags[index],
            ),
        )
    for (left, right), label in state.edges.items():
        graph.add_edge(int(left), int(right), label=str(label))
    return graph


def _wl_hash(state: CheckpointState) -> str:
    """Permutation-invariant WL digest of the visible quotient graph only."""

    graph = _visible_graph(state)
    if state.num_nodes <= 0:
        return "empty"
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="label", iterations=3)


def _visible_graph_hash_payload(state: CheckpointState) -> dict[str, Any]:
    """Canonical hash payload for :math:`H`, excluding every object field."""

    graph = _visible_graph(state)
    return {
        "wl": _wl_hash(state),
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "node_labels": sorted(str(data["label"]) for _, data in graph.nodes(data=True)),
        "edge_labels": sorted(str(data["label"]) for _, _, data in graph.edges(data=True)),
    }


def _ring_count(state: CheckpointState) -> int:
    if state.num_nodes <= 0:
        return 0
    graph = nx.Graph()
    graph.add_nodes_from(range(state.num_nodes))
    graph.add_edges_from(state.edges)
    return len(nx.cycle_basis(graph))


def _ring_size_hist(state: CheckpointState) -> dict[str, int]:
    if state.num_nodes <= 0:
        return {}
    graph = nx.Graph()
    graph.add_nodes_from(range(state.num_nodes))
    graph.add_edges_from(state.edges)
    hist = Counter(len(cycle) for cycle in nx.cycle_basis(graph))
    return dict(sorted((str(k), int(v)) for k, v in hist.items()))


def _bond_hist(state: CheckpointState) -> dict[str, int]:
    return dict(sorted((str(k), int(v)) for k, v in Counter(state.edges.values()).items()))


def _atom_hist(state: CheckpointState) -> list[int]:
    if not state.node_summaries:
        return []
    size = len(state.node_summaries[0].hist)
    return [int(sum(summary.hist[idx] for summary in state.node_summaries)) for idx in range(size)]


def _partition_signature(state: CheckpointState) -> list[list[int]]:
    """Return the explicit quotient-node-to-source-block map ``pi``.

    The outer order must follow quotient-node indices.  Only the certificate's
    permutation-invariant hash signature is sorted; sorting this payload would
    destroy the inspectable association between ``U`` and its source blocks.
    """

    return [list(sorted(int(value) for value in ids)) for ids in state.original_atom_ids]


def _partition_hash_signature(state: CheckpointState) -> list[dict[str, Any]]:
    """Permutation-stable partition signature for reusable object certificates.

    The full partition with original atom ids is still stored in the object for
    traceability. The certificate hash intentionally uses only block sizes and
    summaries, so two equivalent quotient objects from different molecules do
    not fail exact matching only because their original atom ids differ.
    """
    rows: list[dict[str, Any]] = []
    for ids, summary, core, ring in zip(state.original_atom_ids, state.node_summaries, state.core_flags, state.ring_flags):
        rows.append(
            {
                "block_size": len(ids),
                "summary": summary.to_dict(),
                "core": bool(core),
                "ring": bool(ring),
            }
        )
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))


def _boundary_slot_signatures(state: CheckpointState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot in state.attachment_slots:
        boundary_atoms = tuple(getattr(slot, "boundary_atoms", ()) or (slot.anchor,))
        boundary_bond_labels = tuple(
            getattr(slot, "boundary_bond_labels", ())
            or (slot.bond_label,) * len(boundary_atoms)
        )
        out.append(
            {
                "anchor": int(slot.anchor),
                "boundary_atoms": [int(x) for x in boundary_atoms],
                "boundary_bond_labels": [str(x) for x in boundary_bond_labels],
                "bond_label": str(slot.bond_label),
                "summary": slot.summary.to_dict(),
                "original_atom_ids": [int(x) for x in slot.original_atom_ids],
                "atom_anchor": None if slot.atom_anchor is None else int(slot.atom_anchor),
                "boundary_atom_ids": [int(x) for x in slot.boundary_atom_ids],
                "boundary_count": len(boundary_atoms),
                "multiplicity": len(boundary_atoms),
            }
        )
    return sorted(out, key=lambda item: (item["boundary_atoms"], item["original_atom_ids"], item["bond_label"]))


def _boundary_slot_hash_signatures(state: CheckpointState) -> dict[str, Any]:
    r"""Canonical, source-id-free boundary incidence for exact certification.

    Attachment anchors are quotient-node indices, hence raw indices cannot be
    placed directly in a permutation-invariant certificate.  We instead add
    every slot and each of its anchor/boundary relations as typed auxiliary
    nodes to the visible quotient graph and hash that joint incidence graph.
    Source-molecule identifiers (``original_atom_ids``, ``atom_anchor`` and
    ``boundary_atom_ids``) are deliberately excluded; only the contracted
    block size is retained.
    """

    graph = nx.Graph()
    for index, summary in enumerate(state.node_summaries):
        graph.add_node(
            ("quotient", index),
            label="quotient:"
            + json.dumps(
                {
                    "visible": json.loads(
                        _visible_node_token(
                            summary,
                            core=state.core_flags[index],
                            ring=state.ring_flags[index],
                        )
                    ),
                    "structural_summary": summary.to_dict(),
                    "block_size": len(state.original_atom_ids[index]),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    for (left, right), label in state.edges.items():
        graph.add_edge(
            ("quotient", int(left)),
            ("quotient", int(right)),
            label=f"quotient-edge:{label}",
        )

    slot_metadata: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(state.attachment_slots):
        boundary_atoms = tuple(getattr(slot, "boundary_atoms", ()) or (slot.anchor,))
        boundary_bond_labels = tuple(
            getattr(slot, "boundary_bond_labels", ())
            or (slot.bond_label,) * len(boundary_atoms)
        )
        metadata = {
            "bond_label": str(slot.bond_label),
            "summary": slot.summary.to_dict(),
            "contracted_atom_count": len(slot.original_atom_ids),
            "boundary_count": len(boundary_atoms),
        }
        slot_metadata.append(metadata)
        slot_node = ("slot", slot_index)
        graph.add_node(
            slot_node,
            label="slot:" + json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        )

        anchor_relation = ("anchor", slot_index)
        graph.add_node(anchor_relation, label="relation:anchor")
        graph.add_edge(slot_node, anchor_relation, label="incidence")
        graph.add_edge(
            anchor_relation,
            ("quotient", int(slot.anchor)),
            label="incidence",
        )

        for position, (boundary, bond_label) in enumerate(
            zip(boundary_atoms, boundary_bond_labels)
        ):
            relation_node = ("boundary", slot_index, position)
            graph.add_node(
                relation_node,
                label=f"relation:boundary:{bond_label}",
            )
            graph.add_edge(slot_node, relation_node, label="incidence")
            graph.add_edge(
                relation_node,
                ("quotient", int(boundary)),
                label="incidence",
            )

    if graph.number_of_nodes() <= 0:
        wl = "empty"
    else:
        wl = nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="label",
            edge_attr="label",
            iterations=3,
        )
    return {
        "wl": wl,
        "num_slots": len(state.attachment_slots),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_labels": sorted(str(data["label"]) for _, data in graph.nodes(data=True)),
        "edge_labels": sorted(str(data["label"]) for _, _, data in graph.edges(data=True)),
        "slot_metadata": sorted(
            slot_metadata,
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        ),
    }


def _visible_node_label(summary: NodeSummary) -> str:
    r"""Return the categorical label that belongs to the visible graph ``H``.

    An explicit atom keeps its atom category in :math:`X_Q`; a contracted
    block is represented by the supernode category.  Counts, charge, hidden
    bonds, and ring statistics belong to the separate structural field
    :math:`\sigma` and must not be smuggled into this label.

    Atom categories use their vocabulary index because a serialized certified
    object does not otherwise need to carry the dataset-specific atom-vocabulary
    strings.  Tensorization obtains the full atom histogram from ``sigma``.
    """

    occupied = [idx for idx, count in enumerate(summary.hist) if int(count) != 0]
    if summary.atom_count == 1 and len(occupied) == 1 and int(summary.hist[occupied[0]]) == 1:
        return f"atom:{occupied[0]}"
    return "supernode"


def _quotient_payload(state: CheckpointState) -> dict[str, Any]:
    return {
        "num_nodes": state.num_nodes,
        "nodes": [
            {
                "idx": idx,
                "label": _visible_node_label(summary),
                "core": bool(state.core_flags[idx]),
                "ring": bool(state.ring_flags[idx]),
            }
            for idx, summary in enumerate(state.node_summaries)
        ],
        "edges": [
            {"u": int(u), "v": int(v), "label": str(label)}
            for (u, v), label in sorted(state.edges.items())
        ],
    }


def summary_vector(state: CheckpointState) -> dict[str, Any]:
    hidden_atoms = sum(max(0, summary.atom_count - 1) for summary in state.node_summaries)
    payload = {
        "atom_hist": _atom_hist(state),
        "bond_hist": _bond_hist(state),
        "charge_sum": int(sum(summary.charge for summary in state.node_summaries)),
        "hidden_bonds": int(sum(summary.hidden_bonds for summary in state.node_summaries)),
        "hidden_atoms": int(hidden_atoms),
        "atom_count": int(sum(summary.atom_count for summary in state.node_summaries)),
        "ring_atom_count": int(sum(summary.ring_atom_count for summary in state.node_summaries)),
        "ring_size_hist": _ring_size_hist(state),
        "attachment_slots": len(state.attachment_slots),
    }
    bond_width = max((len(summary.internal_bond_hist) for summary in state.node_summaries), default=0)
    if bond_width:
        payload["internal_bond_hist"] = [
            int(
                sum(
                    summary.internal_bond_hist[index]
                    if index < len(summary.internal_bond_hist)
                    else 0
                    for summary in state.node_summaries
                )
            )
            for index in range(bond_width)
        ]
    return payload


def semantic_descriptor(
    state: CheckpointState,
    level: int | None = None,
    mode: str | None = None,
    radius: int | None = None,
) -> dict[str, Any]:
    boundary_counts = sorted(
        len(tuple(getattr(slot, "boundary_atoms", ()) or (slot.anchor,)))
        for slot in state.attachment_slots
    )
    ring_count = _ring_count(state)
    if state.core_empty:
        topology_family = "ringless"
    elif ring_count == 0:
        topology_family = "acyclic_core"
    elif ring_count == 1:
        topology_family = "monocyclic"
    else:
        topology_family = "polycyclic"
    payload = {
        "stage": state.stage,
        "mode": mode or "unknown",
        "level": None if level is None else int(level),
        "core_empty": bool(state.core_empty),
        "num_nodes": int(state.num_nodes),
        "num_edges": int(len(state.edges)),
        "core_nodes": int(sum(bool(x) for x in state.core_flags)),
        "ring_nodes": int(sum(bool(x) for x in state.ring_flags)),
        "ring_count": int(ring_count),
        "ring_size_hist": _ring_size_hist(state),
        "attachment_slots": int(len(state.attachment_slots)),
        "boundary_count_multiset": [int(x) for x in boundary_counts],
        "topology_family": topology_family,
    }
    if radius is not None:
        payload["radius"] = int(radius)
    return payload


@dataclass(frozen=True)
class VisibleQuotientGraph:
    r"""The visible quotient state :math:`H=(U,E_Q,X_Q,A_Q)` only."""

    num_nodes: int
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": int(self.num_nodes),
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "VisibleQuotientGraph":
        nodes = tuple(dict(node) for node in data.get("nodes", []))
        edges = tuple(dict(edge) for edge in data.get("edges", []))
        num_nodes = int(data.get("num_nodes", len(nodes)))
        if num_nodes != len(nodes):
            raise ValueError("Visible quotient node count does not match node payload.")
        return VisibleQuotientGraph(num_nodes=num_nodes, nodes=nodes, edges=edges)


@dataclass(frozen=True)
class CertifiedCheckpointObject:
    r"""Typed representation of :math:`\mathcal O(H)`.

    ``quotient_graph`` is the explicit visible quotient payload ``H``.  The
    remaining fields carry the partition, structural summaries, boundary and
    semantic/certification information that must not be confused with ``H`` at
    neural-model boundaries.
    """

    quotient_graph: VisibleQuotientGraph
    partition: list[list[int]]
    structural_summaries: tuple[NodeSummary, ...]
    summary: dict[str, Any]
    boundary_slots: list[dict[str, Any]]
    semantic: dict[str, Any]
    quotient_graph_hash: str
    semantic_hash: str
    certificate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "checkpoint_dag_diffusion.certified_object.v2",
            "quotient_graph": self.quotient_graph.to_dict(),
            "partition": self.partition,
            "structural_summaries": [item.to_dict() for item in self.structural_summaries],
            "summary": self.summary,
            "boundary_slots": self.boundary_slots,
            "semantic": self.semantic,
            "quotient_graph_hash": self.quotient_graph_hash,
            "semantic_hash": self.semantic_hash,
            "certificate_hash": self.certificate_hash,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CertifiedCheckpointObject":
        required = (
            "quotient_graph",
            "partition",
            "summary",
            "boundary_slots",
            "semantic",
            "quotient_graph_hash",
            "semantic_hash",
            "certificate_hash",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Certified checkpoint object is missing fields: {missing}")
        fmt = str(data.get("format", ""))
        if fmt and fmt != "checkpoint_dag_diffusion.certified_object.v2":
            raise ValueError(f"Unsupported certified-object format: {fmt}")

        quotient_graph = VisibleQuotientGraph.from_dict(data["quotient_graph"])
        serialized_sigma = data.get("structural_summaries")
        if serialized_sigma is None:
            # Compatibility migration for early v2 objects, whose visible-node
            # labels incorrectly contained the complete NodeSummary token.
            rows = sorted(quotient_graph.nodes, key=lambda row: int(row["idx"]))
            try:
                structural_summaries = tuple(NodeSummary.from_label(str(row["label"])) for row in rows)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Certified checkpoint object is missing the per-node structural_summaries field."
                ) from exc
            quotient_graph = VisibleQuotientGraph(
                num_nodes=quotient_graph.num_nodes,
                nodes=tuple(
                    {
                        **row,
                        "label": _visible_node_label(summary),
                    }
                    for row, summary in zip(rows, structural_summaries)
                ),
                edges=quotient_graph.edges,
            )
        else:
            if not isinstance(serialized_sigma, (list, tuple)):
                raise ValueError("Certified checkpoint structural_summaries must be a sequence.")
            structural_summaries = tuple(NodeSummary.from_dict(dict(row)) for row in serialized_sigma)
        if len(structural_summaries) != quotient_graph.num_nodes:
            raise ValueError("Certified checkpoint structural-summary count does not match H.")
        partition = [list(map(int, block)) for block in data["partition"]]
        if len(partition) != quotient_graph.num_nodes:
            raise ValueError("Certified checkpoint partition count does not match H.")
        for slot in data["boundary_slots"]:
            anchor = int(slot.get("anchor", -1))
            if anchor < 0 or anchor >= quotient_graph.num_nodes:
                raise ValueError(f"Certified checkpoint boundary anchor {anchor} is outside H.")

        return CertifiedCheckpointObject(
            quotient_graph=quotient_graph,
            partition=partition,
            structural_summaries=structural_summaries,
            summary=dict(data["summary"]),
            boundary_slots=[dict(slot) for slot in data["boundary_slots"]],
            semantic=dict(data["semantic"]),
            quotient_graph_hash=str(data["quotient_graph_hash"]),
            semantic_hash=str(data["semantic_hash"]),
            certificate_hash=str(data["certificate_hash"]),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.quotient_graph.num_nodes)


def certified_object_from_state(
    state: CheckpointState,
    level: int | None = None,
    mode: str | None = None,
    radius: int | None = None,
) -> CertifiedCheckpointObject:
    quotient = _quotient_payload(state)
    partition = _partition_signature(state)
    summary = summary_vector(state)
    boundary_slots = _boundary_slot_signatures(state)
    semantic = semantic_descriptor(state, level=level, mode=mode, radius=radius)
    quotient_graph_hash = _namespaced_digest(
        _visible_graph_hash_payload(state),
        _QUOTIENT_HASH_XOR_MASK,
    )
    semantic_hash = _namespaced_digest(semantic, _SEMANTIC_HASH_XOR_MASK)
    certificate_payload = {
        "quotient_graph_hash": quotient_graph_hash,
        "partition": _partition_hash_signature(state),
        "summary": summary,
        "boundary_slots": _boundary_slot_hash_signatures(state),
        "semantic": semantic,
    }
    return CertifiedCheckpointObject(
        quotient_graph=VisibleQuotientGraph.from_dict(quotient),
        partition=partition,
        structural_summaries=tuple(state.node_summaries),
        summary=summary,
        boundary_slots=boundary_slots,
        semantic=semantic,
        quotient_graph_hash=quotient_graph_hash,
        semantic_hash=semantic_hash,
        certificate_hash=_namespaced_digest(
            certificate_payload,
            _CERTIFICATE_HASH_XOR_MASK,
        ),
    )


def exact_certificate_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("certificate_hash", "")) == str(right.get("certificate_hash", ""))


def semantic_certificate_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("semantic_hash", "")) == str(right.get("semantic_hash", ""))


def _quotient_payload_graph(payload: dict[str, Any]) -> nx.Graph:
    quotient = payload.get("quotient_graph")
    if not isinstance(quotient, dict):
        raise ValueError("Certified object has no quotient_graph payload.")
    nodes = quotient.get("nodes", [])
    edges = quotient.get("edges", [])
    num_nodes = int(quotient.get("num_nodes", len(nodes)))
    if not isinstance(nodes, (list, tuple)) or len(nodes) != num_nodes:
        raise ValueError("Malformed quotient node payload.")

    graph = nx.Graph()
    for row in nodes:
        if not isinstance(row, dict):
            raise ValueError("Malformed quotient node row.")
        index = int(row["idx"])
        if index < 0 or index >= num_nodes or index in graph:
            raise ValueError("Quotient node indices must be unique and in range.")
        graph.add_node(
            index,
            label=str(row.get("label", "")),
            core=bool(row.get("core", False)),
            ring=bool(row.get("ring", False)),
        )
    for row in edges:
        if not isinstance(row, dict):
            raise ValueError("Malformed quotient edge row.")
        left, right = int(row["u"]), int(row["v"])
        if left == right or left not in graph or right not in graph:
            raise ValueError("Malformed quotient edge endpoint.")
        graph.add_edge(left, right, label=str(row.get("label", "")))
    return graph


def quotient_graph_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_hash = str(left.get("quotient_graph_hash", ""))
    right_hash = str(right.get("quotient_graph_hash", ""))
    if not left_hash or left_hash != right_hash:
        return False
    if not isinstance(left.get("quotient_graph"), dict) or not isinstance(
        right.get("quotient_graph"), dict
    ):
        # Legacy metric records may carry only the digest.
        return True
    try:
        left_graph = _quotient_payload_graph(left)
        right_graph = _quotient_payload_graph(right)
    except (KeyError, TypeError, ValueError):
        return False
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["label", "core", "ring"],
        ["", False, False],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", "")
    return nx.is_isomorphic(
        left_graph,
        right_graph,
        node_match=node_match,
        edge_match=edge_match,
    )


def normalized_summary_error(left: dict[str, Any], right: dict[str, Any], eps: float = 1e-8) -> float:
    def flatten(value: Any) -> list[float]:
        if isinstance(value, dict):
            out: list[float] = []
            for key in sorted(value):
                out.extend(flatten(value[key]))
            return out
        if isinstance(value, (list, tuple)):
            out: list[float] = []
            for item in value:
                out.extend(flatten(item))
            return out
        if isinstance(value, (int, float, bool)):
            return [float(value)]
        return []

    a = flatten(left.get("summary", {}))
    b = flatten(right.get("summary", {}))
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    a = a + [0.0] * (n - len(a))
    b = b + [0.0] * (n - len(b))
    denom = sum(abs(x) for x in b) + eps
    return float(sum(abs(x - y) for x, y in zip(a, b)) / denom)
