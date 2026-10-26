from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from .certified_object import certified_object_from_state
from .checkpoint_state import AttachmentSlot, CheckpointState, NodeSummary
from .chemistry import DEFAULT_ATOM_VOCAB, DEFAULT_BOND_VOCAB
from .topology import MoleculeGraph


OP_NOISE_TO_CHECKPOINT = 0
OP_ADJACENT_REFINEMENT = 1
OP_SKIP_REFINEMENT = 2


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _atom_summary(graph: MoleculeGraph, idx: int, atom_vocab: Sequence[str]) -> NodeSummary:
    return NodeSummary.from_atom(
        graph.atom_symbols[idx],
        charge=graph.charges[idx],
        in_ring=idx in graph.ring_atoms,
        atom_vocab=atom_vocab,
    )


@dataclass(frozen=True)
class DAGCheckpoint:
    state: CheckpointState
    level: int
    radius: int | None
    mode: str
    hash: str

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "level": self.level,
            "radius": self.radius,
            "mode": self.mode,
            "hash": self.hash,
            "certified_object": certified_object_from_state(
                self.state,
                level=self.level,
                mode=self.mode,
                radius=self.radius,
            ).to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "DAGCheckpoint":
        state = CheckpointState.from_dict(data["state"])
        return DAGCheckpoint(
            state=state,
            level=int(data["level"]),
            radius=None if data.get("radius") is None else int(data["radius"]),
            mode=str(data["mode"]),
            hash=str(data.get("hash", state.canonical_hash())),
        )


@dataclass(frozen=True)
class CorrespondenceMap:
    r"""Nested-partition supervision :math:`\mu_{g\rightarrow f}`.

    Indices refer to quotient-node positions in the stored guide (coarse) and
    target (fine) checkpoint objects. ``preserved_target_to_guide`` contains
    exact one-to-one blocks that can be copied safely during refinement;
    expanded guide blocks remain in ``guide_to_target`` but are never copied by
    positional prefix.
    """

    guide_to_target: dict[int, tuple[int, ...]]
    target_to_guide: dict[int, int]
    preserved_target_to_guide: dict[int, int]
    guide_boundary_to_target: dict[int, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guide_to_target": {str(k): [int(v) for v in values] for k, values in sorted(self.guide_to_target.items())},
            "target_to_guide": {str(k): int(v) for k, v in sorted(self.target_to_guide.items())},
            "preserved_target_to_guide": {
                str(k): int(v) for k, v in sorted(self.preserved_target_to_guide.items())
            },
            "guide_boundary_to_target": {
                str(k): [int(v) for v in values]
                for k, values in sorted(self.guide_boundary_to_target.items())
            },
        }

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "CorrespondenceMap | None":
        if not data:
            return None
        return CorrespondenceMap(
            guide_to_target={int(k): tuple(int(v) for v in values) for k, values in data.get("guide_to_target", {}).items()},
            target_to_guide={int(k): int(v) for k, v in data.get("target_to_guide", {}).items()},
            preserved_target_to_guide={
                int(k): int(v) for k, v in data.get("preserved_target_to_guide", {}).items()
            },
            guide_boundary_to_target={
                int(k): tuple(int(v) for v in values)
                for k, values in data.get("guide_boundary_to_target", {}).items()
            },
        )


def refinement_correspondence(finer: CheckpointState, coarser: CheckpointState) -> CorrespondenceMap:
    """Construct the explicit coarse-to-fine map from nested atom blocks."""

    fine_blocks = [set(int(v) for v in block) for block in finer.original_atom_ids]
    coarse_blocks = [set(int(v) for v in block) for block in coarser.original_atom_ids]
    guide_to_target: dict[int, tuple[int, ...]] = {}
    target_to_guide: dict[int, int] = {}
    preserved: dict[int, int] = {}
    for guide_idx, coarse_block in enumerate(coarse_blocks):
        target_indices = tuple(
            target_idx
            for target_idx, fine_block in enumerate(fine_blocks)
            if fine_block and fine_block.issubset(coarse_block)
        )
        if not target_indices:
            raise ValueError(f"Guide block {guide_idx} has no finer correspondence.")
        covered: set[int] = set()
        for target_idx in target_indices:
            if target_idx in target_to_guide:
                raise ValueError(f"Fine block {target_idx} maps to multiple guide blocks.")
            target_to_guide[target_idx] = guide_idx
            covered.update(fine_blocks[target_idx])
        if covered != coarse_block:
            raise ValueError(f"Nested partitions do not exactly cover guide block {guide_idx}.")
        guide_to_target[guide_idx] = target_indices
        if len(target_indices) == 1 and fine_blocks[target_indices[0]] == coarse_block:
            preserved[target_indices[0]] = guide_idx

    boundary_map: dict[int, tuple[int, ...]] = {}
    for slot_idx, slot in enumerate(coarser.attachment_slots):
        hidden = set(int(v) for v in slot.original_atom_ids)
        matches = tuple(
            target_idx
            for target_idx, fine_block in enumerate(fine_blocks)
            if hidden.intersection(fine_block)
        )
        boundary_map[slot_idx] = matches
    return CorrespondenceMap(
        guide_to_target=guide_to_target,
        target_to_guide=target_to_guide,
        preserved_target_to_guide=preserved,
        guide_boundary_to_target=boundary_map,
    )


def checkpoint_state_isomorphism(
    source: CheckpointState,
    target: CheckpointState,
) -> dict[int, int] | None:
    """Return an exact labeled-quotient isomorphism ``source -> target``."""

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["label", "core", "ring"],
        ["", 0, 0],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", "none")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source.to_networkx(),
        target.to_networkx(),
        node_match=node_match,
        edge_match=edge_match,
    )
    if not matcher.is_isomorphic():
        return None
    return {int(left): int(right) for left, right in matcher.mapping.items()}


def transfer_correspondence(
    correspondence: CorrespondenceMap,
    *,
    support_finer: CheckpointState,
    support_coarser: CheckpointState,
    representative_finer: CheckpointState,
    representative_coarser: CheckpointState,
) -> CorrespondenceMap | None:
    """Transfer a local ``mu`` into the two global representative orderings.

    Global entries choose representatives independently of any one supporting
    transition.  A local index map may therefore be reused only when both
    endpoints are exactly isomorphic to those representatives.
    """

    fine_map = checkpoint_state_isomorphism(support_finer, representative_finer)
    coarse_map = checkpoint_state_isomorphism(support_coarser, representative_coarser)
    if fine_map is None or coarse_map is None:
        return None

    def slot_signature(slot: AttachmentSlot, mapped_anchor: int) -> tuple[Any, ...]:
        boundary_labels = tuple(sorted(slot.boundary_bond_labels or (slot.bond_label,)))
        return (
            int(mapped_anchor),
            str(slot.bond_label),
            slot.summary.label(),
            len(slot.boundary_atoms or (slot.anchor,)),
            boundary_labels,
        )

    representative_slots: dict[tuple[Any, ...], list[int]] = {}
    for index, slot in enumerate(representative_coarser.attachment_slots):
        representative_slots.setdefault(slot_signature(slot, slot.anchor), []).append(index)
    slot_map: dict[int, int] = {}
    for index, slot in enumerate(support_coarser.attachment_slots):
        signature = slot_signature(slot, coarse_map[int(slot.anchor)])
        matches = representative_slots.get(signature, [])
        if len(matches) != 1:
            return None
        slot_map[index] = int(matches[0])

    guide_to_target = {
        coarse_map[int(guide)]: tuple(fine_map[int(target)] for target in targets)
        for guide, targets in correspondence.guide_to_target.items()
    }
    target_to_guide = {
        fine_map[int(target)]: coarse_map[int(guide)]
        for target, guide in correspondence.target_to_guide.items()
    }
    preserved = {
        fine_map[int(target)]: coarse_map[int(guide)]
        for target, guide in correspondence.preserved_target_to_guide.items()
    }
    boundary = {
        slot_map[int(slot)]: tuple(fine_map[int(target)] for target in targets)
        for slot, targets in correspondence.guide_boundary_to_target.items()
    }
    return CorrespondenceMap(
        guide_to_target=guide_to_target,
        target_to_guide=target_to_guide,
        preserved_target_to_guide=preserved,
        guide_boundary_to_target=boundary,
    )


@dataclass(frozen=True)
class DAGEdge:
    target: int
    guide: int | None
    target_level: int
    guide_level: int
    delta: int
    operator_id: int
    correspondence: CorrespondenceMap | None = None

    def to_dict(self) -> dict:
        payload = {
            "target": self.target,
            "guide": self.guide,
            "target_level": self.target_level,
            "guide_level": self.guide_level,
            "delta": self.delta,
            "operator_id": self.operator_id,
        }
        if self.correspondence is not None:
            payload["correspondence"] = self.correspondence.to_dict()
        return payload

    @staticmethod
    def from_dict(data: dict) -> "DAGEdge":
        return DAGEdge(
            target=int(data["target"]),
            guide=None if data.get("guide") is None else int(data["guide"]),
            target_level=int(data["target_level"]),
            guide_level=int(data["guide_level"]),
            delta=int(data["delta"]),
            operator_id=int(data["operator_id"]),
            correspondence=CorrespondenceMap.from_dict(data.get("correspondence")),
        )


@dataclass(frozen=True)
class CheckpointDAG:
    checkpoints: tuple[DAGCheckpoint, ...]
    edges: tuple[DAGEdge, ...]
    source_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "dag_edges": [edge.to_dict() for edge in self.edges],
            "num_levels": len(self.checkpoints),
            "coarsest": len(self.checkpoints) - 1,
        }


def anchor_atoms(graph: MoleculeGraph) -> set[int]:
    if graph.num_atoms == 0:
        return set()
    if graph.ring_atoms:
        return set(graph.ring_atoms)
    nx_graph = graph.to_networkx()
    two_core = set(nx.k_core(nx_graph, k=2).nodes())
    if two_core:
        return two_core

    anchors: set[int] = set()
    for component in nx.connected_components(nx_graph):
        sub = nx_graph.subgraph(component).copy()
        if sub.number_of_nodes() == 1:
            anchors.update(sub.nodes())
        else:
            anchors.update(nx.center(sub))
    return anchors


def distances_to_anchors(graph: MoleculeGraph, anchors: set[int]) -> dict[int, int]:
    if not anchors:
        return {idx: 0 for idx in range(graph.num_atoms)}
    nx_graph = graph.to_networkx()
    distances = nx.multi_source_dijkstra_path_length(nx_graph, sorted(anchors))
    return {idx: int(distances.get(idx, 0)) for idx in range(graph.num_atoms)}


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
            bond_index = bond_vocab.index(label) if label in bond_vocab else bond_vocab.index("single")
            bond_hist[bond_index] += 1
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


def _off_explicit_components(graph: MoleculeGraph, explicit_atoms: set[int]) -> list[tuple[set[int], list[tuple[int, int, str]]]]:
    off_atoms = set(range(graph.num_atoms)) - set(explicit_atoms)
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
            if u in explicit_atoms and v in comp_nodes:
                attachments.append((u, v, label))
            elif v in explicit_atoms and u in comp_nodes:
                attachments.append((v, u, label))
        components.append((set(comp_nodes), sorted(attachments)))
    return components


def _full_state(
    graph: MoleculeGraph,
    anchors: set[int],
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> CheckpointState:
    original_ids = graph.original_atom_ids or tuple(range(graph.num_atoms))
    return CheckpointState(
        stage="h",
        node_summaries=tuple(_atom_summary(graph, idx, atom_vocab) for idx in range(graph.num_atoms)),
        edges=dict(graph.edges),
        core_flags=tuple(idx in anchors for idx in range(graph.num_atoms)),
        ring_flags=tuple(idx in graph.ring_atoms for idx in range(graph.num_atoms)),
        original_atom_ids=tuple((int(original_ids[idx]),) for idx in range(graph.num_atoms)),
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )


def _radius_state(
    graph: MoleculeGraph,
    anchors: set[int],
    explicit_atoms: set[int],
    radius: int,
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> CheckpointState:
    if not explicit_atoms:
        explicit_atoms = set(anchors) if anchors else {0}
    original_ids = graph.original_atom_ids or tuple(range(graph.num_atoms))
    explicit_order = sorted(explicit_atoms)
    atom_to_node = {atom_idx: node_idx for node_idx, atom_idx in enumerate(explicit_order)}

    nodes: list[NodeSummary] = [_atom_summary(graph, atom_idx, atom_vocab) for atom_idx in explicit_order]
    originals: list[tuple[int, ...]] = [(int(original_ids[atom_idx]),) for atom_idx in explicit_order]
    core_flags: list[bool] = [atom_idx in anchors for atom_idx in explicit_order]
    ring_flags: list[bool] = [atom_idx in graph.ring_atoms for atom_idx in explicit_order]
    edges: dict[tuple[int, int], str] = {}
    slots: list[AttachmentSlot] = []

    for (u, v), label in graph.edges.items():
        if u in explicit_atoms and v in explicit_atoms:
            edges[_edge_key(atom_to_node[u], atom_to_node[v])] = label

    for comp_nodes, attachments in _off_explicit_components(graph, explicit_atoms):
        summary = _component_summary(graph, comp_nodes, atom_vocab, bond_vocab)
        comp_originals = tuple(int(original_ids[idx]) for idx in sorted(comp_nodes))
        if attachments:
            anchor_atom, _, bond_label = attachments[0]
            anchor_node = atom_to_node[anchor_atom]
        else:
            anchor_atom = explicit_order[0]
            anchor_node = 0
            bond_label = "single"
        boundary_source_atoms = tuple(a for a, _, _ in attachments if a in atom_to_node) or (anchor_atom,)
        boundary_atoms = tuple(atom_to_node[a] for a in boundary_source_atoms)
        boundary_atom_ids = tuple(int(original_ids[a]) for a in boundary_source_atoms)
        boundary_bond_labels = tuple(label for _, _, label in attachments) or (bond_label,)
        summary_node = len(nodes)
        nodes.append(summary)
        originals.append(comp_originals)
        core_flags.append(False)
        ring_flags.append(False)
        edges[_edge_key(anchor_node, summary_node)] = bond_label
        slots.append(
            AttachmentSlot(
                anchor=anchor_node,
                summary=summary,
                bond_label=bond_label,
                original_atom_ids=comp_originals,
                atom_anchor=int(original_ids[anchor_atom]),
                z1_node=summary_node,
                boundary_atoms=boundary_atoms,
                boundary_atom_ids=boundary_atom_ids,
                boundary_bond_labels=boundary_bond_labels,
            )
        )

    return CheckpointState(
        stage="h",
        node_summaries=tuple(nodes),
        edges=edges,
        core_flags=tuple(core_flags),
        ring_flags=tuple(ring_flags),
        original_atom_ids=tuple(originals),
        attachment_slots=tuple(slots),
        core_empty=False,
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )


def _pure_core_state(
    graph: MoleculeGraph,
    anchors: set[int],
    atom_vocab: Sequence[str],
    bond_vocab: Sequence[str],
) -> CheckpointState:
    if not anchors:
        anchors = {0}
    original_ids = graph.original_atom_ids or tuple(range(graph.num_atoms))
    anchor_order = sorted(anchors)
    atom_to_node = {atom_idx: node_idx for node_idx, atom_idx in enumerate(anchor_order)}
    nodes = [_atom_summary(graph, atom_idx, atom_vocab) for atom_idx in anchor_order]
    originals: list[tuple[int, ...]] = [(int(original_ids[atom_idx]),) for atom_idx in anchor_order]
    edges: dict[tuple[int, int], str] = {}
    slots: list[AttachmentSlot] = []

    for (u, v), label in graph.edges.items():
        if u in anchors and v in anchors:
            edges[_edge_key(atom_to_node[u], atom_to_node[v])] = label

    for comp_nodes, attachments in _off_explicit_components(graph, anchors):
        summary = _component_summary(graph, comp_nodes, atom_vocab, bond_vocab)
        comp_originals = tuple(int(original_ids[idx]) for idx in sorted(comp_nodes))
        if attachments:
            anchor_atom, _, bond_label = attachments[0]
            anchor_node = atom_to_node[anchor_atom]
        else:
            anchor_atom = anchor_order[0]
            anchor_node = 0
            bond_label = "single"
        boundary_source_atoms = tuple(a for a, _, _ in attachments if a in atom_to_node) or (anchor_atom,)
        boundary_atoms = tuple(atom_to_node[a] for a in boundary_source_atoms)
        boundary_atom_ids = tuple(int(original_ids[a]) for a in boundary_source_atoms)
        boundary_bond_labels = tuple(label for _, _, label in attachments) or (bond_label,)
        nodes[anchor_node] = NodeSummary.merge(
            [nodes[anchor_node], summary],
            extra_hidden_bonds=1 if bond_label != "none" else 0,
            extra_internal_bond_hist=_single_bond_hist(bond_label, bond_vocab),
        )
        originals[anchor_node] = tuple(sorted(originals[anchor_node] + comp_originals))
        slots.append(
            AttachmentSlot(
                anchor=anchor_node,
                summary=summary,
                bond_label=bond_label,
                original_atom_ids=comp_originals,
                atom_anchor=int(original_ids[anchor_atom]),
                z1_node=None,
                boundary_atoms=boundary_atoms,
                boundary_atom_ids=boundary_atom_ids,
                boundary_bond_labels=boundary_bond_labels,
            )
        )

    return CheckpointState(
        stage="h",
        node_summaries=tuple(nodes),
        edges=edges,
        core_flags=tuple(True for _ in nodes),
        # ``originals`` stores source-molecule identifiers, whereas
        # ``graph.ring_atoms`` is indexed in the current heavy-atom graph.
        # Compare in the local indexing domain so non-contiguous source ids do
        # not silently erase ring membership from the certified object.
        ring_flags=tuple(anchor in graph.ring_atoms for anchor in anchor_order),
        original_atom_ids=tuple(originals),
        attachment_slots=tuple(slots),
        core_empty=not bool(graph.ring_atoms),
        atom_vocab=tuple(atom_vocab),
        bond_vocab=tuple(bond_vocab),
    )


def checkpoint_contractible_to(finer: CheckpointState, coarser: CheckpointState) -> bool:
    finer_parts = {frozenset(ids) for ids in finer.original_atom_ids}
    if not finer_parts:
        return False
    finer_union = set().union(*finer_parts)
    coarser_parts = [frozenset(ids) for ids in coarser.original_atom_ids]
    if set().union(*coarser_parts) != finer_union:
        return False
    for coarse_part in coarser_parts:
        remaining = set(coarse_part)
        for fine_part in finer_parts:
            if fine_part <= coarse_part:
                remaining -= set(fine_part)
        if remaining:
            return False
    return True


def build_checkpoint_dag(
    graph: MoleculeGraph,
    atom_vocab: Sequence[str] = DEFAULT_ATOM_VOCAB,
    bond_vocab: Sequence[str] = DEFAULT_BOND_VOCAB,
    max_radius: int | None = None,
    include_skip_edges: bool = True,
) -> CheckpointDAG:
    if graph.num_atoms == 0:
        raise ValueError("Cannot build a checkpoint DAG for an empty graph.")
    anchors = anchor_atoms(graph)
    distances = distances_to_anchors(graph, anchors)
    max_dist = max(distances.values()) if distances else 0
    if max_radius is not None:
        max_dist = min(max_dist, int(max_radius))

    candidates: list[tuple[CheckpointState, int | None, str]] = []
    candidates.append((_full_state(graph, anchors, atom_vocab, bond_vocab), None, "full"))
    for radius in range(max_dist, -1, -1):
        explicit = {idx for idx, dist in distances.items() if dist <= radius}
        candidates.append((_radius_state(graph, anchors, explicit, radius, atom_vocab, bond_vocab), radius, "radius"))
    candidates.append((_pure_core_state(graph, anchors, atom_vocab, bond_vocab), None, "pure_core"))

    checkpoints: list[DAGCheckpoint] = []
    seen: set[str] = set()
    for state, radius, mode in candidates:
        state_hash = state.canonical_hash()
        if state_hash in seen:
            continue
        seen.add(state_hash)
        checkpoints.append(DAGCheckpoint(state=state, level=len(checkpoints), radius=radius, mode=mode, hash=state_hash))

    if len(checkpoints) == 1:
        # Degenerate one-atom case: create one noise edge only.
        guide_level = 1
    else:
        guide_level = len(checkpoints)

    edges: list[DAGEdge] = [
        DAGEdge(
            target=len(checkpoints) - 1,
            guide=None,
            target_level=len(checkpoints) - 1,
            guide_level=guide_level,
            delta=1,
            operator_id=OP_NOISE_TO_CHECKPOINT,
            correspondence=None,
        )
    ]
    for target_idx, target in enumerate(checkpoints):
        for guide_idx, guide in enumerate(checkpoints[target_idx + 1 :], start=target_idx + 1):
            if not include_skip_edges and guide_idx != target_idx + 1:
                continue
            if not checkpoint_contractible_to(target.state, guide.state):
                continue
            delta = guide.level - target.level
            edges.append(
                DAGEdge(
                    target=target_idx,
                    guide=guide_idx,
                    target_level=target.level,
                    guide_level=guide.level,
                    delta=delta,
                    operator_id=OP_ADJACENT_REFINEMENT if delta == 1 else OP_SKIP_REFINEMENT,
                    correspondence=refinement_correspondence(target.state, guide.state),
                )
            )

    return CheckpointDAG(checkpoints=tuple(checkpoints), edges=tuple(edges), source_id=graph.source_id)
