from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx

from .checkpoint_state import AttachmentSlot, CheckpointState, NodeSummary
from .chemistry import DEFAULT_ATOM_VOCAB, DEFAULT_BOND_VOCAB


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class MoleculeGraph:
    atom_symbols: tuple[str, ...]
    charges: tuple[int, ...]
    edges: dict[tuple[int, int], str]
    ring_atoms: frozenset[int] = frozenset()
    ring_edges: frozenset[tuple[int, int]] = frozenset()
    original_atom_ids: tuple[int, ...] | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        normalized = {_edge_key(int(u), int(v)): str(label) for (u, v), label in self.edges.items() if u != v}
        object.__setattr__(self, "edges", normalized)
        if len(self.atom_symbols) != len(self.charges):
            raise ValueError("atom_symbols and charges must have the same length.")
        if self.original_atom_ids is None:
            object.__setattr__(self, "original_atom_ids", tuple(range(len(self.atom_symbols))))

    @property
    def num_atoms(self) -> int:
        return len(self.atom_symbols)

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_atoms))
        for (u, v), label in self.edges.items():
            graph.add_edge(u, v, label=label)
        return graph

    @staticmethod
    def from_dict(data: dict) -> "MoleculeGraph":
        return MoleculeGraph(
            atom_symbols=tuple(str(x) for x in data["atom_symbols"]),
            charges=tuple(int(x) for x in data.get("charges", [0] * len(data["atom_symbols"]))),
            edges={(int(e["u"]), int(e["v"])): str(e.get("label", "single")) for e in data.get("edges", [])},
            ring_atoms=frozenset(int(x) for x in data.get("ring_atoms", [])),
            ring_edges=frozenset(_edge_key(int(e["u"]), int(e["v"])) for e in data.get("ring_edges", [])),
            original_atom_ids=tuple(int(x) for x in data.get("original_atom_ids", range(len(data["atom_symbols"])))),
            source_id=data.get("source_id"),
        )

    def to_dict(self) -> dict:
        return {
            "atom_symbols": list(self.atom_symbols),
            "charges": list(self.charges),
            "edges": [{"u": u, "v": v, "label": label} for (u, v), label in sorted(self.edges.items())],
            "ring_atoms": sorted(self.ring_atoms),
            "ring_edges": [{"u": u, "v": v} for (u, v) in sorted(self.ring_edges)],
            "original_atom_ids": list(self.original_atom_ids or range(self.num_atoms)),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class CheckpointLadder:
    z0: CheckpointState
    z1: CheckpointState
    z2: CheckpointState
    source_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "z0": self.z0.to_dict(),
            "z1": self.z1.to_dict(),
            "z2": self.z2.to_dict(),
            "hashes": {
                "z0": self.z0.canonical_hash(),
                "z1": self.z1.canonical_hash(),
                "z2": self.z2.canonical_hash(),
            },
        }


def _atom_summary(graph: MoleculeGraph, idx: int, atom_vocab: Sequence[str]) -> NodeSummary:
    return NodeSummary.from_atom(
        graph.atom_symbols[idx],
        charge=graph.charges[idx],
        in_ring=idx in graph.ring_atoms,
        atom_vocab=atom_vocab,
    )


def _state_from_full_graph(
    graph: MoleculeGraph,
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> CheckpointState:
    summaries = tuple(_atom_summary(graph, i, atom_vocab) for i in range(graph.num_atoms))
    original_ids = tuple((int((graph.original_atom_ids or tuple(range(graph.num_atoms)))[i]),) for i in range(graph.num_atoms))
    return CheckpointState(
        stage="z0",
        node_summaries=summaries,
        edges=dict(graph.edges),
        core_flags=tuple(False for _ in summaries),
        ring_flags=tuple(i in graph.ring_atoms for i in range(graph.num_atoms)),
        original_atom_ids=original_ids,
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )


def _cycle_core_atoms(graph: MoleculeGraph) -> set[int]:
    nx_graph = graph.to_networkx()
    two_core = set(nx.k_core(nx_graph, k=2).nodes()) if graph.num_atoms > 0 else set()
    if graph.ring_atoms:
        # RDKit ring atoms are the primary signal; the 2-core check keeps the
        # graph-theoretic interpretation explicit and catches unusual inputs.
        return set(graph.ring_atoms) | (set(graph.ring_atoms) & two_core)
    return two_core


def _core_edges(graph: MoleculeGraph, core_atoms: set[int]) -> dict[tuple[int, int], str]:
    if not core_atoms:
        return {}
    out: dict[tuple[int, int], str] = {}
    for (u, v), label in graph.edges.items():
        if u in core_atoms and v in core_atoms:
            if not graph.ring_edges or (u, v) in graph.ring_edges or label == "aromatic":
                out[(u, v)] = label
    return out


def _component_summary(
    graph: MoleculeGraph,
    component: Iterable[int],
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> NodeSummary:
    nodes = set(int(value) for value in component)
    summaries = [_atom_summary(graph, idx, atom_vocab) for idx in sorted(nodes)]
    bond_hist = [0 for _ in bond_vocab]
    for (left, right), label in graph.edges.items():
        if left in nodes and right in nodes:
            index = bond_vocab.index(label) if label in bond_vocab else bond_vocab.index("single")
            bond_hist[index] += 1
    return NodeSummary.merge(
        summaries,
        extra_hidden_bonds=sum(bond_hist),
        extra_internal_bond_hist=bond_hist,
    )


def _single_bond_hist(label: str, bond_vocab: Sequence[str]) -> tuple[int, ...]:
    values = [0 for _ in bond_vocab]
    if label != "none":
        index = bond_vocab.index(label) if label in bond_vocab else bond_vocab.index("single")
        values[index] = 1
    return tuple(values)


def _off_core_components(graph: MoleculeGraph, core_atoms: set[int]) -> list[tuple[set[int], list[tuple[int, int, str]]]]:
    off_atoms = set(range(graph.num_atoms)) - set(core_atoms)
    if not off_atoms:
        return []
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(off_atoms)
    for (u, v), label in graph.edges.items():
        if u in off_atoms and v in off_atoms:
            nx_graph.add_edge(u, v, label=label)

    components: list[tuple[set[int], list[tuple[int, int, str]]]] = []
    for comp_nodes in nx.connected_components(nx_graph):
        attachments: list[tuple[int, int, str]] = []
        for (u, v), label in graph.edges.items():
            if u in comp_nodes and v in core_atoms:
                attachments.append((v, u, label))
            elif v in comp_nodes and u in core_atoms:
                attachments.append((u, v, label))
        components.append((set(comp_nodes), sorted(attachments)))
    return components


def _build_ringless_z1_z2(
    graph: MoleculeGraph,
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> tuple[CheckpointState, CheckpointState]:
    summary = _component_summary(graph, range(graph.num_atoms), atom_vocab, bond_vocab)
    original_ids = tuple(int((graph.original_atom_ids or tuple(range(graph.num_atoms)))[i]) for i in range(graph.num_atoms))
    slot = AttachmentSlot(
        anchor=0,
        summary=summary,
        bond_label="none",
        original_atom_ids=original_ids,
        atom_anchor=original_ids[0],
        z1_node=0,
        boundary_atoms=(0,),
        boundary_atom_ids=(original_ids[0],),
        boundary_bond_labels=("none",),
    )
    common = dict(
        node_summaries=(summary,),
        edges={},
        core_flags=(True,),
        ring_flags=(False,),
        original_atom_ids=(original_ids,),
        attachment_slots=(slot,),
        core_empty=True,
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )
    return CheckpointState(stage="z1", **common), CheckpointState(stage="z2", **common)


def build_checkpoint_ladder(
    graph: MoleculeGraph,
    atom_vocab: Sequence[str] = DEFAULT_ATOM_VOCAB,
    bond_vocab: Sequence[str] = DEFAULT_BOND_VOCAB,
) -> CheckpointLadder:
    z0 = _state_from_full_graph(graph, atom_vocab=atom_vocab, bond_vocab=bond_vocab)
    core_atoms = _cycle_core_atoms(graph)
    if not core_atoms:
        z1, z2 = _build_ringless_z1_z2(graph, atom_vocab=atom_vocab, bond_vocab=bond_vocab)
        return CheckpointLadder(z0=z0, z1=z1, z2=z2, source_id=graph.source_id)

    core_order = sorted(core_atoms)
    atom_to_core_node = {atom_idx: node_idx for node_idx, atom_idx in enumerate(core_order)}
    original_atom_ids = graph.original_atom_ids or tuple(range(graph.num_atoms))

    z1_nodes: list[NodeSummary] = [_atom_summary(graph, atom_idx, atom_vocab) for atom_idx in core_order]
    z1_originals: list[tuple[int, ...]] = [(int(original_atom_ids[atom_idx]),) for atom_idx in core_order]
    z1_core_flags: list[bool] = [True for _ in core_order]
    z1_ring_flags: list[bool] = [True for _ in core_order]
    z1_edges: dict[tuple[int, int], str] = {}
    for (u, v), label in _core_edges(graph, core_atoms).items():
        z1_edges[_edge_key(atom_to_core_node[u], atom_to_core_node[v])] = label

    z2_nodes: list[NodeSummary] = list(z1_nodes)
    z2_originals: list[tuple[int, ...]] = list(z1_originals)
    slots: list[AttachmentSlot] = []

    for comp_nodes, attachments in _off_core_components(graph, core_atoms):
        summary = _component_summary(graph, comp_nodes, atom_vocab=atom_vocab, bond_vocab=bond_vocab)
        comp_originals = tuple(int(original_atom_ids[i]) for i in sorted(comp_nodes))
        if attachments:
            anchor_atom, _, bond_label = attachments[0]
            anchor = atom_to_core_node[anchor_atom]
        else:
            anchor_atom = core_order[0]
            anchor = 0
            bond_label = "single"
        boundary_source_atoms = tuple(a for a, _, _ in attachments) or (anchor_atom,)
        boundary_atoms = tuple(atom_to_core_node[a] for a in boundary_source_atoms)
        boundary_atom_ids = tuple(int(original_atom_ids[a]) for a in boundary_source_atoms)
        boundary_bond_labels = tuple(label for _, _, label in attachments) or (bond_label,)

        z1_node = len(z1_nodes)
        z1_nodes.append(summary)
        z1_originals.append(comp_originals)
        z1_core_flags.append(False)
        z1_ring_flags.append(False)
        z1_edges[_edge_key(anchor, z1_node)] = bond_label

        z2_nodes[anchor] = NodeSummary.merge(
            [z2_nodes[anchor], summary],
            extra_hidden_bonds=1 if bond_label != "none" else 0,
            extra_internal_bond_hist=_single_bond_hist(bond_label, bond_vocab),
        )
        z2_originals[anchor] = tuple(sorted(z2_originals[anchor] + comp_originals))
        slots.append(
            AttachmentSlot(
                anchor=anchor,
                summary=summary,
                bond_label=bond_label,
                original_atom_ids=comp_originals,
                atom_anchor=int(original_atom_ids[anchor_atom]),
                z1_node=z1_node,
                boundary_atoms=boundary_atoms,
                boundary_atom_ids=boundary_atom_ids,
                boundary_bond_labels=boundary_bond_labels,
            )
        )

    z2_edges = dict(z1_edges)
    for slot in slots:
        if slot.z1_node is not None:
            z2_edges.pop(_edge_key(slot.anchor, slot.z1_node), None)

    z1 = CheckpointState(
        stage="z1",
        node_summaries=tuple(z1_nodes),
        edges=z1_edges,
        core_flags=tuple(z1_core_flags),
        ring_flags=tuple(z1_ring_flags),
        original_atom_ids=tuple(z1_originals),
        attachment_slots=tuple(slots),
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )
    z2 = CheckpointState(
        stage="z2",
        node_summaries=tuple(z2_nodes),
        edges=z2_edges,
        core_flags=tuple(True for _ in z2_nodes),
        ring_flags=tuple(True for _ in z2_nodes),
        original_atom_ids=tuple(z2_originals),
        attachment_slots=tuple(slots),
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )
    return CheckpointLadder(z0=z0, z1=z1, z2=z2, source_id=graph.source_id)


def contract_z1_to_z2(state: CheckpointState) -> CheckpointState:
    if state.stage == "z2":
        return state
    if state.core_empty:
        data = state.to_dict()
        data["stage"] = "z2"
        return CheckpointState.from_dict(data)
    core_nodes = [idx for idx, is_core in enumerate(state.core_flags) if is_core]
    old_to_new = {old: new for new, old in enumerate(core_nodes)}
    nodes = [state.node_summaries[idx] for idx in core_nodes]
    originals = [state.original_atom_ids[idx] for idx in core_nodes]
    slots_by_node = {slot.z1_node: slot for slot in state.attachment_slots if slot.z1_node is not None}
    for old_idx, slot in slots_by_node.items():
        if old_idx >= state.num_nodes or slot.anchor not in old_to_new:
            continue
        anchor = old_to_new[slot.anchor]
        nodes[anchor] = NodeSummary.merge(
            [nodes[anchor], state.node_summaries[old_idx]],
            extra_hidden_bonds=1,
            extra_internal_bond_hist=_single_bond_hist(slot.bond_label, state.bond_vocab),
        )
        originals[anchor] = tuple(sorted(originals[anchor] + state.original_atom_ids[old_idx]))
    edges = {
        _edge_key(old_to_new[u], old_to_new[v]): label
        for (u, v), label in state.edges.items()
        if u in old_to_new and v in old_to_new
    }
    return CheckpointState(
        stage="z2",
        node_summaries=tuple(nodes),
        edges=edges,
        core_flags=tuple(True for _ in nodes),
        ring_flags=tuple(True for _ in nodes),
        original_atom_ids=tuple(originals),
        attachment_slots=state.attachment_slots,
        core_empty=False,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )


def _subtract_summaries(total: NodeSummary, parts: list[NodeSummary], hidden_edges: int) -> NodeSummary | None:
    hist = list(total.hist)
    for part in parts:
        hist = [a - b for a, b in zip(hist, part.hist)]
    charge = total.charge - sum(part.charge for part in parts)
    hidden = total.hidden_bonds - sum(part.hidden_bonds for part in parts) - hidden_edges
    atom_count = total.atom_count - sum(part.atom_count for part in parts)
    ring_atom_count = total.ring_atom_count - sum(part.ring_atom_count for part in parts)
    if any(value < 0 for value in hist) or hidden < 0 or atom_count <= 0 or ring_atom_count < 0:
        return None
    return NodeSummary(
        hist=tuple(hist),
        charge=charge,
        hidden_bonds=hidden,
        atom_count=atom_count,
        ring_atom_count=ring_atom_count,
    )


def project_z1_to_guide_z2(candidate: CheckpointState, guide_z2: CheckpointState) -> CheckpointState:
    """Repair a generated Z1 so its contraction is exactly the guided Z2 when feasible.

    The sampler uses this as the hard boundary projection between the
    `Z2 -> Z1` and `Z1 -> Z0` segments. Extra generated nodes are treated as
    attachment summaries; each guide core node receives a residual summary so
    that residual + attachments contracts back to the guide summary.
    """

    if guide_z2.core_empty:
        data = guide_z2.to_dict()
        data["stage"] = "z1"
        return CheckpointState.from_dict(data)
    core_n = guide_z2.num_nodes
    if core_n == 0:
        return candidate

    attachment_by_anchor: dict[int, list[tuple[int, NodeSummary, str, tuple[int, ...]]]] = {idx: [] for idx in range(core_n)}
    for node_idx, summary in enumerate(candidate.node_summaries):
        if node_idx < core_n:
            continue
        anchor = None
        for nbr in candidate.neighbors(node_idx):
            if nbr < core_n:
                anchor = nbr
                break
        if anchor is None:
            anchor = node_idx % core_n
        edge_label = candidate.edges.get(_edge_key(anchor, node_idx), "single")
        attachment_by_anchor[anchor].append((node_idx, summary, edge_label, candidate.original_atom_ids[node_idx]))

    nodes: list[NodeSummary] = []
    originals: list[tuple[int, ...]] = []
    kept_attachments: list[tuple[int, NodeSummary, str, tuple[int, ...]]] = []
    for anchor in range(core_n):
        parts = [item[1] for item in attachment_by_anchor[anchor]]
        residual = _subtract_summaries(guide_z2.node_summaries[anchor], parts, hidden_edges=len(parts))
        if residual is None:
            residual = guide_z2.node_summaries[anchor]
            attachment_by_anchor[anchor] = []
        nodes.append(residual)
        originals.append(guide_z2.original_atom_ids[anchor])
        kept_attachments.extend((anchor, item[1], item[2], item[3]) for item in attachment_by_anchor[anchor])

    edges = dict(guide_z2.edges)
    slots: list[AttachmentSlot] = []
    for anchor, summary, bond_label, original_ids in kept_attachments:
        z1_node = len(nodes)
        nodes.append(summary)
        originals.append(original_ids)
        edges[_edge_key(anchor, z1_node)] = bond_label if bond_label != "none" else "single"
        slots.append(
            AttachmentSlot(
                anchor=anchor,
                summary=summary,
                bond_label=bond_label if bond_label != "none" else "single",
                original_atom_ids=original_ids,
                z1_node=z1_node,
            )
        )

    return CheckpointState(
        stage="z1",
        node_summaries=tuple(nodes),
        edges=edges,
        core_flags=tuple([True] * core_n + [False] * len(kept_attachments)),
        ring_flags=tuple([True] * core_n + [False] * len(kept_attachments)),
        original_atom_ids=tuple(originals),
        attachment_slots=tuple(slots),
        core_empty=False,
        atom_vocab=guide_z2.atom_vocab,
        bond_vocab=guide_z2.bond_vocab,
    )


def checkpoint_consistent(child: CheckpointState, guide: CheckpointState) -> bool:
    if guide.stage == "z2" and child.stage == "z1":
        return contract_z1_to_z2(child).canonical_hash() == guide.canonical_hash()
    return child.canonical_hash() == guide.canonical_hash()
