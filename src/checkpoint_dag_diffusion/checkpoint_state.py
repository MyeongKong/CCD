from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Sequence

import networkx as nx

from .chemistry import DEFAULT_ATOM_VOCAB, DEFAULT_BOND_VOCAB


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class NodeSummary:
    hist: tuple[int, ...]
    charge: int = 0
    hidden_bonds: int = 0
    atom_count: int = 1
    ring_atom_count: int = 0
    internal_bond_hist: tuple[int, ...] = field(default_factory=tuple)

    @staticmethod
    def empty(atom_vocab: Sequence[str] = DEFAULT_ATOM_VOCAB) -> "NodeSummary":
        return NodeSummary(tuple(0 for _ in atom_vocab), 0, 0, 0, 0)

    @staticmethod
    def from_atom(
        atom_symbol: str,
        charge: int = 0,
        in_ring: bool = False,
        atom_vocab: Sequence[str] = DEFAULT_ATOM_VOCAB,
    ) -> "NodeSummary":
        hist = [0 for _ in atom_vocab]
        idx = atom_vocab.index(atom_symbol) if atom_symbol in atom_vocab else atom_vocab.index("other")
        hist[idx] = 1
        return NodeSummary(tuple(hist), int(charge), 0, 1, int(in_ring))

    @staticmethod
    def merge(
        summaries: Iterable["NodeSummary"],
        extra_hidden_bonds: int = 0,
        extra_internal_bond_hist: Sequence[int] | None = None,
    ) -> "NodeSummary":
        items = list(summaries)
        if not items:
            return NodeSummary.empty()
        hist = tuple(sum(ns.hist[i] for ns in items) for i in range(len(items[0].hist)))
        bond_width = max(
            [len(ns.internal_bond_hist) for ns in items]
            + [0 if extra_internal_bond_hist is None else len(extra_internal_bond_hist)]
        )
        internal_bond_hist = tuple(
            sum(ns.internal_bond_hist[i] if i < len(ns.internal_bond_hist) else 0 for ns in items)
            + (
                int(extra_internal_bond_hist[i])
                if extra_internal_bond_hist is not None and i < len(extra_internal_bond_hist)
                else 0
            )
            for i in range(bond_width)
        )
        return NodeSummary(
            hist=hist,
            charge=sum(ns.charge for ns in items),
            hidden_bonds=sum(ns.hidden_bonds for ns in items) + int(extra_hidden_bonds),
            atom_count=sum(ns.atom_count for ns in items),
            ring_atom_count=sum(ns.ring_atom_count for ns in items),
            internal_bond_hist=internal_bond_hist,
        )

    def as_feature(self) -> list[float]:
        unresolved = 1.0 if self.atom_count > 1 else 0.0
        return [float(x) for x in self.hist] + [
            float(self.charge),
            float(self.hidden_bonds),
            float(self.atom_count),
            float(self.ring_atom_count),
            unresolved,
        ]

    def label(self) -> str:
        hist = ",".join(str(x) for x in self.hist)
        label = f"h={hist}|q={self.charge}|hb={self.hidden_bonds}|n={self.atom_count}|r={self.ring_atom_count}"
        if self.internal_bond_hist:
            label += "|ibh=" + ",".join(str(value) for value in self.internal_bond_hist)
        return label

    @staticmethod
    def from_label(label: str) -> "NodeSummary":
        r"""Parse the legacy canonical node-summary token.

        Early v2 certified objects placed this token in the visible quotient
        label.  New objects serialize per-node summaries as the separate
        structural field :math:`\sigma`; this parser remains for migration of
        existing artifacts.
        """

        fields: dict[str, str] = {}
        for part in str(label).split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        required = {"h", "q", "hb", "n", "r"}
        if not required.issubset(fields):
            raise ValueError(f"Invalid canonical node-summary label: {label!r}")
        return NodeSummary(
            hist=tuple(int(value) for value in fields["h"].split(",")),
            charge=int(fields["q"]),
            hidden_bonds=int(fields["hb"]),
            atom_count=int(fields["n"]),
            ring_atom_count=int(fields["r"]),
            internal_bond_hist=tuple(int(value) for value in fields.get("ibh", "").split(",") if value != ""),
        )

    def to_dict(self) -> dict:
        payload = {
            "hist": list(self.hist),
            "charge": self.charge,
            "hidden_bonds": self.hidden_bonds,
            "atom_count": self.atom_count,
            "ring_atom_count": self.ring_atom_count,
        }
        if self.internal_bond_hist:
            payload["internal_bond_hist"] = list(self.internal_bond_hist)
        return payload

    @staticmethod
    def from_dict(data: dict) -> "NodeSummary":
        return NodeSummary(
            hist=tuple(int(x) for x in data["hist"]),
            charge=int(data.get("charge", 0)),
            hidden_bonds=int(data.get("hidden_bonds", data.get("hidden_single_bonds", 0))),
            atom_count=int(data.get("atom_count", sum(data["hist"]))),
            ring_atom_count=int(data.get("ring_atom_count", 0)),
            internal_bond_hist=tuple(int(value) for value in data.get("internal_bond_hist", [])),
        )


@dataclass(frozen=True)
class AttachmentSlot:
    anchor: int
    summary: NodeSummary
    bond_label: str = "single"
    original_atom_ids: tuple[int, ...] = field(default_factory=tuple)
    atom_anchor: int | None = None
    z1_node: int | None = None
    boundary_atoms: tuple[int, ...] = field(default_factory=tuple)
    boundary_atom_ids: tuple[int, ...] = field(default_factory=tuple)
    boundary_bond_labels: tuple[str, ...] = field(default_factory=tuple)
    representative_slot_id: int | None = None

    @property
    def multiplicity(self) -> int:
        return len(self.boundary_atoms or (self.anchor,))

    def to_dict(self) -> dict:
        boundary_atoms = self.boundary_atoms or (self.anchor,)
        boundary_bond_labels = self.boundary_bond_labels or (self.bond_label,)
        return {
            "anchor": self.anchor,
            "multiplicity": self.multiplicity,
            "summary": self.summary.to_dict(),
            "bond_label": self.bond_label,
            "original_atom_ids": list(self.original_atom_ids),
            "atom_anchor": self.atom_anchor,
            "z1_node": self.z1_node,
            "boundary_atoms": [int(x) for x in boundary_atoms],
            "boundary_atom_ids": [int(x) for x in self.boundary_atom_ids],
            "boundary_bond_labels": [str(x) for x in boundary_bond_labels],
            "representative_slot_id": self.representative_slot_id,
        }

    @staticmethod
    def from_dict(data: dict) -> "AttachmentSlot":
        anchor = int(data["anchor"])
        bond_label = str(data.get("bond_label", "single"))
        return AttachmentSlot(
            anchor=anchor,
            summary=NodeSummary.from_dict(data["summary"]),
            bond_label=bond_label,
            original_atom_ids=tuple(int(x) for x in data.get("original_atom_ids", [])),
            atom_anchor=None if data.get("atom_anchor") is None else int(data["atom_anchor"]),
            z1_node=None if data.get("z1_node") is None else int(data["z1_node"]),
            boundary_atoms=tuple(int(x) for x in data.get("boundary_atoms", [anchor])),
            boundary_atom_ids=tuple(int(x) for x in data.get("boundary_atom_ids", [])),
            boundary_bond_labels=tuple(str(x) for x in data.get("boundary_bond_labels", [bond_label])),
            representative_slot_id=(
                None
                if data.get("representative_slot_id") is None
                else int(data["representative_slot_id"])
            ),
        )


@dataclass
class CheckpointState:
    r"""Construction-time carrier for a local contracted checkpoint.

    This class contains the information needed while constructing
    :math:`O(H)` (including source blocks and boundary slots); it must not be
    interpreted as the bare visible quotient graph ``H`` at model boundaries.
    Neural code should consume a serialized ``CertifiedCheckpointObject`` via
    ``certified_object_to_tensors``.
    """
    stage: str
    node_summaries: tuple[NodeSummary, ...]
    edges: dict[tuple[int, int], str]
    core_flags: tuple[bool, ...]
    ring_flags: tuple[bool, ...]
    original_atom_ids: tuple[tuple[int, ...], ...]
    attachment_slots: tuple[AttachmentSlot, ...] = field(default_factory=tuple)
    tensor_slot_ids: tuple[int, ...] = field(default_factory=tuple)
    core_empty: bool = False
    atom_vocab: tuple[str, ...] = tuple(DEFAULT_ATOM_VOCAB)
    bond_vocab: tuple[str, ...] = tuple(DEFAULT_BOND_VOCAB)

    def __post_init__(self) -> None:
        normalized: dict[tuple[int, int], str] = {}
        for (u, v), label in self.edges.items():
            if u == v:
                continue
            normalized[_edge_key(int(u), int(v))] = str(label)
        self.edges = normalized
        n = len(self.node_summaries)
        if len(self.core_flags) != n or len(self.ring_flags) != n or len(self.original_atom_ids) != n:
            raise ValueError("CheckpointState metadata lengths must match node count.")
        if self.tensor_slot_ids and len(self.tensor_slot_ids) != n:
            raise ValueError("tensor_slot_ids must be empty or match node count.")
        if self.tensor_slot_ids:
            known_slots = [int(value) for value in self.tensor_slot_ids if int(value) >= 0]
            if any(int(value) < -1 for value in self.tensor_slot_ids) or len(set(known_slots)) != len(known_slots):
                raise ValueError(
                    "tensor_slot_ids must contain unique representative slots or -1 for unknown alignment."
                )
        seen_atoms: set[int] = set()
        for index, (summary, block) in enumerate(zip(self.node_summaries, self.original_atom_ids)):
            if int(summary.atom_count) <= 0 or sum(int(value) for value in summary.hist) != int(summary.atom_count):
                raise ValueError(f"Checkpoint node {index} has an inconsistent atom-count summary.")
            if summary.internal_bond_hist:
                if len(summary.internal_bond_hist) != len(self.bond_vocab):
                    raise ValueError(f"Checkpoint node {index} has an incompatible internal bond histogram.")
                if sum(int(value) for value in summary.internal_bond_hist) != int(summary.hidden_bonds):
                    raise ValueError(f"Checkpoint node {index} has an inconsistent hidden-bond summary.")
            if len(block) != int(summary.atom_count):
                raise ValueError(f"Checkpoint partition block {index} does not match its structural summary.")
            block_set = {int(value) for value in block}
            if len(block_set) != len(block) or seen_atoms & block_set:
                raise ValueError("Checkpoint partition blocks must be internally unique and disjoint.")
            seen_atoms.update(block_set)
        for (u, v), label in self.edges.items():
            if u < 0 or v < 0 or u >= n or v >= n:
                raise ValueError(f"Checkpoint edge ({u}, {v}) is outside the quotient node set.")
            if label not in self.bond_vocab:
                raise ValueError(f"Unknown checkpoint bond label: {label!r}.")
        for slot in self.attachment_slots:
            if int(slot.anchor) < 0 or int(slot.anchor) >= n:
                raise ValueError(f"Attachment anchor {slot.anchor} is outside the quotient node set.")
            # ``z1_node`` may intentionally refer to the corresponding node in
            # a finer checkpoint rather than to a node of the current H.
            if any(int(value) < 0 or int(value) >= n for value in slot.boundary_atoms):
                raise ValueError("Attachment boundary_atoms must index quotient nodes.")
            if slot.boundary_bond_labels and len(slot.boundary_bond_labels) != len(slot.boundary_atoms):
                raise ValueError("Attachment boundary atoms and bond labels must have equal multiplicity.")
            if slot.boundary_atom_ids and len(slot.boundary_atom_ids) != len(slot.boundary_atoms):
                raise ValueError("Attachment quotient boundaries and atom-level anchors must have equal multiplicity.")
            if slot.representative_slot_id is not None and int(slot.representative_slot_id) < -1:
                raise ValueError("Attachment representative_slot_id must be non-negative or -1 for unknown.")
        known_boundary_slots = [
            int(slot.representative_slot_id)
            for slot in self.attachment_slots
            if slot.representative_slot_id is not None and int(slot.representative_slot_id) >= 0
        ]
        if len(set(known_boundary_slots)) != len(known_boundary_slots):
            raise ValueError("Known attachment representative-slot aliases must be unique.")

    @property
    def num_nodes(self) -> int:
        return len(self.node_summaries)

    @property
    def node_feature_dim(self) -> int:
        return len(self.atom_vocab) + 5

    def neighbors(self, node: int) -> list[int]:
        out: list[int] = []
        for u, v in self.edges:
            if u == node:
                out.append(v)
            elif v == node:
                out.append(u)
        return sorted(out)

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        for idx, summary in enumerate(self.node_summaries):
            graph.add_node(
                idx,
                label=summary.label(),
                core=int(self.core_flags[idx]),
                ring=int(self.ring_flags[idx]),
                stage=self.stage,
            )
        for (u, v), label in self.edges.items():
            graph.add_edge(u, v, label=label)
        return graph

    def canonical_hash(self) -> str:
        graph = self.to_networkx()
        if self.num_nodes > 0:
            wl = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="label", iterations=3)
        else:
            wl = "empty"
        payload = {
            "stage": self.stage,
            "wl": wl,
            "core_empty": self.core_empty,
            "node_labels": sorted(ns.label() for ns in self.node_summaries),
            "edge_labels": sorted(str(label) for label in self.edges.values()),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "node_summaries": [ns.to_dict() for ns in self.node_summaries],
            "edges": [{"u": u, "v": v, "label": label} for (u, v), label in sorted(self.edges.items())],
            "core_flags": [bool(x) for x in self.core_flags],
            "ring_flags": [bool(x) for x in self.ring_flags],
            "original_atom_ids": [list(ids) for ids in self.original_atom_ids],
            "attachment_slots": [slot.to_dict() for slot in self.attachment_slots],
            "tensor_slot_ids": [int(value) for value in self.tensor_slot_ids],
            "core_empty": self.core_empty,
            "atom_vocab": list(self.atom_vocab),
            "bond_vocab": list(self.bond_vocab),
            "canonical_hash": self.canonical_hash(),
        }

    @staticmethod
    def from_dict(data: dict) -> "CheckpointState":
        return CheckpointState(
            stage=str(data["stage"]),
            node_summaries=tuple(NodeSummary.from_dict(x) for x in data["node_summaries"]),
            edges={(int(e["u"]), int(e["v"])): str(e["label"]) for e in data.get("edges", [])},
            core_flags=tuple(bool(x) for x in data["core_flags"]),
            ring_flags=tuple(bool(x) for x in data["ring_flags"]),
            original_atom_ids=tuple(tuple(int(v) for v in ids) for ids in data["original_atom_ids"]),
            attachment_slots=tuple(AttachmentSlot.from_dict(x) for x in data.get("attachment_slots", [])),
            tensor_slot_ids=tuple(int(value) for value in data.get("tensor_slot_ids", [])),
            core_empty=bool(data.get("core_empty", False)),
            atom_vocab=tuple(data.get("atom_vocab", DEFAULT_ATOM_VOCAB)),
            bond_vocab=tuple(data.get("bond_vocab", DEFAULT_BOND_VOCAB)),
        )
