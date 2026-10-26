from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import torch

from .checkpoint_state import AttachmentSlot, CheckpointState, NodeSummary
from .chemistry import DEFAULT_ATOM_VOCAB, DEFAULT_BOND_VOCAB
from .certified_object import CertifiedCheckpointObject, certified_object_from_state


NUM_OPERATORS = 3


def _visible_label(summary: NodeSummary) -> str:
    occupied = [idx for idx, count in enumerate(summary.hist) if int(count) != 0]
    if summary.atom_count == 1 and len(occupied) == 1 and int(summary.hist[occupied[0]]) == 1:
        return f"atom:{occupied[0]}"
    return "supernode"


@dataclass(frozen=True)
class TensorSpec:
    atom_vocab: tuple[str, ...] = tuple(DEFAULT_ATOM_VOCAB)
    bond_vocab: tuple[str, ...] = tuple(DEFAULT_BOND_VOCAB)
    max_nodes: int = 32

    @property
    def node_feature_dim(self) -> int:
        return len(self.atom_vocab) + 5 + 4

    @property
    def edge_dim(self) -> int:
        return len(self.bond_vocab)

    @property
    def categorical_node_indices(self) -> tuple[int, ...]:
        """Binary object fields corrupted in categorical state space.

        The atom histogram, charge and block counts are numerical summaries.
        The unresolved, core, ring and empty-core indicators are categorical.
        """

        start = len(self.atom_vocab) + 4
        return (start, start + 1, start + 2, start + 3)

    @property
    def atom_feature_dim(self) -> int:
        return len(self.atom_vocab)

    @property
    def reserved_node_indices(self) -> tuple[int, ...]:
        return (self.node_feature_dim - 1,)


def certified_object_to_tensors(
    certified_object: CertifiedCheckpointObject | dict[str, Any],
    spec: TensorSpec,
) -> dict[str, torch.Tensor]:
    r"""Implement :math:`\mathsf T(\mathcal O(H))=(X^{obj},A_Q,m)`.

    Only the selected structural/semantic descriptors used by the published
    21-channel representation are placed in the neural tensors.  Partition,
    boundary and certificate fields remain inspectable on ``certified_object``;
    boundary participation is returned separately as an auxiliary supervision
    target and is not folded into the visible quotient state ``H``.
    """

    obj = (
        certified_object
        if isinstance(certified_object, CertifiedCheckpointObject)
        else CertifiedCheckpointObject.from_dict(certified_object)
    )
    quotient = obj.quotient_graph
    nodes = sorted(quotient.nodes, key=lambda row: int(row["idx"]))
    n_total = int(quotient.num_nodes)
    if n_total != len(nodes):
        raise ValueError("Certified object quotient node count does not match its node payload.")
    if n_total > spec.max_nodes:
        raise ValueError(f"Certified object has {n_total} nodes but max_nodes={spec.max_nodes}.")
    n = n_total
    x = torch.zeros(spec.max_nodes, spec.node_feature_dim, dtype=torch.float32)
    none_idx = spec.bond_vocab.index("none")
    edge = torch.full((spec.max_nodes, spec.max_nodes), none_idx, dtype=torch.long)
    mask = torch.zeros(spec.max_nodes, dtype=torch.bool)
    core = torch.zeros(spec.max_nodes, dtype=torch.float32)
    ring = torch.zeros(spec.max_nodes, dtype=torch.float32)
    attachment = torch.zeros(spec.max_nodes, dtype=torch.float32)
    core_empty = bool(obj.semantic.get("core_empty", False))

    structural_summaries = obj.structural_summaries
    if len(structural_summaries) != n:
        raise ValueError("Certified object structural-summary count does not match H.")
    for row, summary in zip(nodes, structural_summaries):
        idx = int(row["idx"])
        if idx < 0 or idx >= n:
            raise ValueError(f"Non-contiguous quotient node index {idx} in certified object.")
        if len(summary.hist) != len(spec.atom_vocab):
            raise ValueError("Certified object atom histogram is incompatible with TensorSpec.")
        visible_label = str(row.get("label", ""))
        expected_label = _visible_label(summary)
        if visible_label != expected_label:
            raise ValueError(
                "Visible quotient label and structural summary disagree at "
                f"node {idx}: {visible_label!r} != {expected_label!r}."
            )
        base = summary.as_feature()
        flags = [
            float(bool(row.get("core", False))),
            float(bool(row.get("ring", False))),
            float(core_empty),
            0.0,
        ]
        x[idx, : len(base) + len(flags)] = torch.tensor(base + flags, dtype=torch.float32)
        mask[idx] = True
        core[idx] = float(bool(row.get("core", False)))
        ring[idx] = float(bool(row.get("ring", False)))

    for row in quotient.edges:
        u, v, label = int(row["u"]), int(row["v"]), str(row["label"])
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError(f"Invalid quotient edge ({u}, {v}) in certified object.")
        label_idx = spec.bond_vocab.index(label) if label in spec.bond_vocab else spec.bond_vocab.index("single")
        edge[u, v] = label_idx
        edge[v, u] = label_idx

    for slot in obj.boundary_slots:
        anchor = int(slot.get("anchor", -1))
        if 0 <= anchor < n:
            attachment[anchor] = 1.0

    return {
        "x": x,
        "edge": edge,
        "mask": mask,
        "core": core,
        "ring": ring,
        "attachment": attachment,
        "num_nodes": torch.tensor(n, dtype=torch.long),
    }


def state_to_tensors(
    state: CheckpointState,
    spec: TensorSpec,
    *,
    level: int | None = None,
    mode: str | None = None,
) -> dict[str, torch.Tensor]:
    r"""Compatibility adapter that first materializes :math:`\mathcal O(H)`.

    New model/data code should pass the stored certified object directly to
    :func:`certified_object_to_tensors`.
    """

    obj = certified_object_from_state(state, level=level, mode=mode)
    return certified_object_to_tensors(obj, spec)


def checkpoint_to_tensors(checkpoint: dict[str, Any], spec: TensorSpec) -> dict[str, torch.Tensor]:
    r"""Tensorize a serialized local checkpoint through its stored object.

    Legacy corpora without ``certified_object`` are supported by materializing
    the object once from the construction carrier, but all newly preprocessed
    corpora take the explicit :math:`\mathcal O(H)` path.
    """

    existing = checkpoint.get("certified_object")
    if isinstance(existing, dict):
        return certified_object_to_tensors(existing, spec)
    state = CheckpointState.from_dict(checkpoint["state"])
    return state_to_tensors(
        state,
        spec,
        level=None if checkpoint.get("level") is None else int(checkpoint["level"]),
        mode=None if checkpoint.get("mode") is None else str(checkpoint["mode"]),
    )


def empty_condition(spec: TensorSpec) -> dict[str, torch.Tensor]:
    none_idx = spec.bond_vocab.index("none")
    return {
        "x": torch.zeros(spec.max_nodes, spec.node_feature_dim, dtype=torch.float32),
        "edge": torch.full((spec.max_nodes, spec.max_nodes), none_idx, dtype=torch.long),
        "mask": torch.zeros(spec.max_nodes, dtype=torch.bool),
        "core": torch.zeros(spec.max_nodes, dtype=torch.float32),
        "ring": torch.zeros(spec.max_nodes, dtype=torch.float32),
        "attachment": torch.zeros(spec.max_nodes, dtype=torch.float32),
        "num_nodes": torch.tensor(0, dtype=torch.long),
    }


def relation_to_tensor(edge: dict) -> torch.Tensor:
    """Return ``(guide level, fine/target level, delta, operator)``."""

    return torch.tensor(
        [
            int(edge["guide_level"]),
            int(edge["target_level"]),
            int(edge["delta"]),
            int(edge["operator_id"]),
        ],
        dtype=torch.long,
    )


def _representative_slots(state: CheckpointState) -> tuple[int, ...]:
    """Return local-node -> representative-node aliases.

    Stored construction/prototype states have an empty alias table and use
    their canonical local ordering.  Sampled states carry an explicit entry
    per local node; ``-1`` means that no semantic alignment was established.
    """

    return state.tensor_slot_ids or tuple(range(state.num_nodes))


def _unique_visible_alignment(
    summaries: Sequence[NodeSummary],
    edges: dict[tuple[int, int], str],
    core_flags: Sequence[bool],
    ring_flags: Sequence[bool],
    template: CheckpointState | None,
    hard_generated_to_representative: dict[int, int],
) -> tuple[dict[int, int], bool]:
    """Align generated local nodes to a selected representative's visible H.

    The target Transformer has no positional embeddings.  Consequently a
    padded tensor position is not a checkpoint-node identity.  We accept a
    generated-to-template map only when the labeled visible-graph
    isomorphism is unique after applying the explicit hard-copy anchors.
    Ambiguous or mismatching nodes remain unaligned and cannot participate in
    a later hard correspondence projection.
    """

    if template is None or len(summaries) != template.num_nodes:
        return {}, False
    template_representatives = _representative_slots(template)
    template_local_by_representative = {
        int(rep): local
        for local, rep in enumerate(template_representatives)
        if int(rep) >= 0
    }
    if any(rep not in template_local_by_representative for rep in hard_generated_to_representative.values()):
        return {}, False

    generated = nx.Graph()
    for local, summary in enumerate(summaries):
        representative = hard_generated_to_representative.get(local)
        generated.add_node(
            local,
            label=_visible_label(summary),
            core=bool(core_flags[local]),
            ring=bool(ring_flags[local]),
            hard="free" if representative is None else f"rep:{representative}",
        )
    for (left, right), label in edges.items():
        generated.add_edge(int(left), int(right), label=str(label))

    hard_representatives = set(hard_generated_to_representative.values())
    target = nx.Graph()
    for local, summary in enumerate(template.node_summaries):
        representative = int(template_representatives[local])
        target.add_node(
            local,
            label=_visible_label(summary),
            core=bool(template.core_flags[local]),
            ring=bool(template.ring_flags[local]),
            hard=(f"rep:{representative}" if representative in hard_representatives else "free"),
        )
    for (left, right), label in template.edges.items():
        target.add_edge(int(left), int(right), label=str(label))

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["label", "core", "ring", "hard"],
        ["", False, False, "free"],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", "none")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        generated,
        target,
        node_match=node_match,
        edge_match=edge_match,
    )
    mappings = matcher.isomorphisms_iter()
    try:
        first = {int(left): int(right) for left, right in next(mappings).items()}
    except StopIteration:
        return {}, False
    try:
        next(mappings)
    except StopIteration:
        return first, True
    # An automorphism leaves the non-anchored representative identity
    # underdetermined.  Retain only the externally established hard anchors.
    return {
        generated_local: template_local_by_representative[representative]
        for generated_local, representative in hard_generated_to_representative.items()
    }, False


def _positive_allocation(total: int, weights: Sequence[float], minimum: int = 0) -> list[int] | None:
    count = len(weights)
    if count == 0 or total < count * minimum:
        return None
    allocation = [minimum for _ in weights]
    remaining = total - count * minimum
    clean = [max(0.0, float(weight)) for weight in weights]
    if sum(clean) <= 0.0:
        clean = [1.0 for _ in weights]
    scaled = [remaining * weight / sum(clean) for weight in clean]
    floors = [int(value) for value in scaled]
    allocation = [base + extra for base, extra in zip(allocation, floors)]
    left = total - sum(allocation)
    order = sorted(range(count), key=lambda idx: (scaled[idx] - floors[idx], -idx), reverse=True)
    for idx in order[:left]:
        allocation[idx] += 1
    return allocation


def _histogram_allocation(
    total_hist: Sequence[int],
    capacities: Sequence[int],
    preferences: Sequence[Sequence[int]],
) -> list[list[int]]:
    rows = [[0 for _ in total_hist] for _ in capacities]
    remaining = [int(value) for value in capacities]
    for category, amount in enumerate(total_hist):
        for _ in range(int(amount)):
            candidates = [idx for idx, capacity in enumerate(remaining) if capacity > 0]
            if not candidates:
                raise ValueError("Histogram allocation exceeds requested child capacities.")
            chosen = max(
                candidates,
                key=lambda idx: (
                    float(preferences[idx][category]) / max(1, sum(int(v) for v in preferences[idx])),
                    remaining[idx],
                    -idx,
                ),
            )
            rows[chosen][category] += 1
            remaining[chosen] -= 1
    if any(remaining):
        raise ValueError("Histogram allocation did not fill every child block.")
    return rows


def _lineage_id_allocation(
    guide_ids: Sequence[int],
    capacities: Sequence[int],
    target_representatives: Sequence[int],
    guide_state: CheckpointState,
    guide_boundary_to_target: dict[int, tuple[int, ...]],
) -> list[tuple[int, ...]] | None:
    """Partition one generated guide block while respecting boundary mu.

    ``guide_boundary_to_target`` is expressed in representative slot/node
    domains.  Generated boundary slots carry their representative alias, so
    hidden-component lineage is assigned only to the target children declared
    by the observed refinement correspondence.
    """

    if sum(int(value) for value in capacities) != len(guide_ids):
        return None
    target_position = {int(rep): pos for pos, rep in enumerate(target_representatives)}
    assigned: list[list[int]] = [[] for _ in capacities]
    owner: dict[int, int] = {}
    constraints: list[tuple[set[int], tuple[int, ...]]] = []
    for local_slot, slot in enumerate(guide_state.attachment_slots):
        representative_slot = (
            int(slot.representative_slot_id)
            if slot.representative_slot_id is not None
            else int(local_slot)
        )
        mapped_targets = guide_boundary_to_target.get(representative_slot)
        if not mapped_targets:
            continue
        allowed = {
            target_position[int(target)]
            for target in mapped_targets
            if int(target) in target_position
        }
        hidden_ids = tuple(int(value) for value in slot.original_atom_ids if int(value) in guide_ids)
        if hidden_ids and allowed:
            constraints.append((allowed, hidden_ids))
    constraints.sort(key=lambda item: (len(item[0]), sorted(item[0]), item[1]))

    for allowed, values in constraints:
        for value in values:
            if value in owner:
                if owner[value] not in allowed:
                    return None
                continue
            candidates = [
                position
                for position in sorted(allowed)
                if len(assigned[position]) < int(capacities[position])
            ]
            if not candidates:
                return None
            chosen = max(candidates, key=lambda position: (int(capacities[position]) - len(assigned[position]), -position))
            assigned[chosen].append(value)
            owner[value] = chosen

    for value in sorted(int(item) for item in guide_ids):
        if value in owner:
            continue
        candidates = [
            position
            for position in range(len(capacities))
            if len(assigned[position]) < int(capacities[position])
        ]
        if not candidates:
            return None
        chosen = max(candidates, key=lambda position: (int(capacities[position]) - len(assigned[position]), -position))
        assigned[chosen].append(value)
    return [tuple(sorted(values)) for values in assigned]


def tensors_to_state(
    x: torch.Tensor,
    edge_logits_or_ids: torch.Tensor,
    active_logits_or_mask: torch.Tensor,
    spec: TensorSpec,
    threshold: float = 0.0,
    *,
    split_logits: torch.Tensor | None = None,
    attachment_logits: torch.Tensor | None = None,
    template_state: CheckpointState | None = None,
    target_level: int | None = None,
    preserved_target_slots: Sequence[int] | set[int] | None = None,
    guide_state: CheckpointState | None = None,
    correspondence: dict[str, Any] | None = None,
) -> CheckpointState:
    """Materialize a sampled tensor as a complete checkpoint carrier.

    ``tensor_slot_ids`` stores a validated generated-node -> representative-
    node alignment, not a padded tensor position.  A value of ``-1`` means
    that the visible realization was mismatched or permutation-ambiguous.
    Hard-preserved nodes keep their complete guide summary and partition
    lineage.  For expanded blocks, the explicit correspondence is used to
    partition the guide lineage whenever both endpoint alignments are known.
    Boundary descriptors are transferred only through a validated template
    alignment and are remapped into the generated atom-id domain.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected an unbatched node tensor, got shape {tuple(x.shape)}.")
    if active_logits_or_mask.dtype == torch.bool:
        active = active_logits_or_mask
    else:
        active = active_logits_or_mask > threshold
    if active.ndim != 1:
        active = active.reshape(-1)
    active_indices = [int(i) for i in active.nonzero(as_tuple=False).flatten().tolist()]
    if not active_indices:
        active_indices = [0]

    old_to_new = {old: new for new, old in enumerate(active_indices)}
    preserved = {
        int(value)
        for value in (preserved_target_slots or ())
        if int(value) in old_to_new
    }
    relation_correspondence = dict(correspondence or {})
    preserved_target_to_guide = {
        int(target): int(guide)
        for target, guide in relation_correspondence.get("preserved_target_to_guide", {}).items()
    }
    guide_representative_to_local: dict[int, int] = {}
    if guide_state is not None:
        guide_representative_to_local = {
            int(representative): local
            for local, representative in enumerate(_representative_slots(guide_state))
            if int(representative) >= 0
        }

    summaries: list[NodeSummary] = []
    core_flags: list[bool] = []
    ring_flags: list[bool] = []
    atom_dim = len(spec.atom_vocab)
    resolved_level = target_level is not None and int(target_level) == 0

    def _finite_scalar(value: torch.Tensor, default: float = 0.0) -> float:
        number = float(value.detach().cpu().item())
        return number if torch.isfinite(value.detach()).all().item() else float(default)

    def _project_histogram(raw: torch.Tensor, count: int) -> tuple[int, ...]:
        values = torch.nan_to_num(raw.detach().float().cpu(), nan=0.0, posinf=16.0, neginf=0.0)
        values = values.clamp(min=0.0, max=16.0)
        if resolved_level:
            index = int(values.argmax().item()) if values.numel() else 0
            return tuple(1 if idx == index else 0 for idx in range(atom_dim))
        if float(values.sum().item()) <= 0.0:
            values = torch.zeros(atom_dim, dtype=torch.float32)
            values[0] = 1.0
        scaled = values / values.sum().clamp_min(1e-8) * int(count)
        projected = torch.floor(scaled).long()
        remaining = int(count) - int(projected.sum().item())
        if remaining > 0:
            order = torch.argsort(scaled - projected.float(), descending=True)
            for index in order[:remaining].tolist():
                projected[int(index)] += 1
        return tuple(int(value) for value in projected.tolist())

    for old in active_indices:
        feat = x[old].detach().cpu()
        unresolved = (
            bool(_finite_scalar(feat[atom_dim + 4]) > 0.5)
            if old in preserved
            else bool(split_logits[old].detach().cpu().item() > threshold)
            if split_logits is not None
            else bool(_finite_scalar(feat[atom_dim + 4]) > 0.5)
        )
        if resolved_level:
            unresolved = False
            atom_count = 1
        else:
            predicted_count = int(round(_finite_scalar(feat[atom_dim + 2], default=1.0)))
            atom_count = max(2 if unresolved else 1, predicted_count)
            if not unresolved:
                atom_count = 1
            atom_count = min(atom_count, 64)
        hist_values = _project_histogram(feat[:atom_dim], atom_count)
        ring_flag = bool(_finite_scalar(feat[atom_dim + 6]) > 0.5)
        ring_atom_count = int(round(_finite_scalar(feat[atom_dim + 3])))
        ring_atom_count = max(0, min(atom_count, ring_atom_count))
        if ring_flag:
            ring_atom_count = max(1, ring_atom_count)
        else:
            ring_atom_count = 0
        predicted_hidden_bonds = (
            0
            if resolved_level
            else max(0, min(4 * atom_count, int(round(_finite_scalar(feat[atom_dim + 1])))))
        )
        summary = NodeSummary(
            hist=hist_values,
            charge=int(round(_finite_scalar(feat[atom_dim]))),
            hidden_bonds=predicted_hidden_bonds,
            atom_count=atom_count,
            ring_atom_count=ring_atom_count,
            internal_bond_hist=tuple(),
        )
        guide_representative = preserved_target_to_guide.get(old)
        guide_local = (
            None
            if guide_representative is None
            else guide_representative_to_local.get(int(guide_representative))
        )
        if old in preserved and guide_state is not None and guide_local is not None:
            # Preservation is an object-level one-to-one correspondence.  The
            # final auxiliary logits must not undo the copied clean x state or
            # erase the hidden structure that is not present in the 21-vector.
            summary = guide_state.node_summaries[guide_local]
        summaries.append(summary)
        core_flags.append(bool(_finite_scalar(feat[atom_dim + 5]) > 0.5))
        ring_flags.append(ring_flag)

    if edge_logits_or_ids.ndim == 3:
        edge_ids = torch.argmax(edge_logits_or_ids, dim=-1)
    else:
        edge_ids = edge_logits_or_ids
    edges: dict[tuple[int, int], str] = {}
    none_idx = spec.bond_vocab.index("none")
    for i, old_i in enumerate(active_indices):
        for old_j in active_indices[i + 1 :]:
            label_idx = int(edge_ids[old_i, old_j].item())
            if label_idx == none_idx:
                continue
            label_idx = max(0, min(label_idx, len(spec.bond_vocab) - 1))
            edges[(old_to_new[old_i], old_to_new[old_j])] = spec.bond_vocab[label_idx]

    # The graph-level empty-core indicator is duplicated across active rows by
    # tensorization, so majority decoding is stable to individual node errors.
    core_empty_index = atom_dim + 7
    core_empty = bool(
        torch.nan_to_num(x[active_indices, core_empty_index].detach().float(), nan=0.0).mean().item() > 0.5
    )

    hard_generated_to_representative = {
        old_to_new[old]: int(old)
        for old in preserved
    }
    generated_to_template, unique_alignment = _unique_visible_alignment(
        summaries,
        edges,
        core_flags,
        ring_flags,
        template_state,
        hard_generated_to_representative,
    )
    template_representatives = (
        tuple() if template_state is None else _representative_slots(template_state)
    )
    aliases = [-1 for _ in summaries]
    for generated_local, template_local in generated_to_template.items():
        representative = int(template_representatives[template_local])
        if representative >= 0:
            aliases[generated_local] = representative
    # Hard-copy anchors remain valid even when the rest of H is ambiguous.
    for generated_local, representative in hard_generated_to_representative.items():
        aliases[generated_local] = representative

    # Propagate a nested generated-atom lineage through the complete mu map.
    # This is deliberately independent of prototype source atom identifiers.
    lineage_blocks: dict[int, tuple[int, ...]] = {}
    if guide_state is not None:
        target_local_by_representative = {
            int(representative): local
            for local, representative in enumerate(aliases)
            if int(representative) >= 0
        }
        guide_to_target = {
            int(guide): tuple(int(target) for target in targets)
            for guide, targets in relation_correspondence.get("guide_to_target", {}).items()
        }
        guide_boundary_to_target = {
            int(slot): tuple(int(target) for target in targets)
            for slot, targets in relation_correspondence.get("guide_boundary_to_target", {}).items()
        }
        for guide_representative, target_representatives in guide_to_target.items():
            guide_local = guide_representative_to_local.get(guide_representative)
            target_locals = [target_local_by_representative.get(target) for target in target_representatives]
            if guide_local is None or any(local is None for local in target_locals):
                continue
            children = [int(local) for local in target_locals if local is not None]
            if len(set(children)) != len(children):
                continue
            guide_ids = tuple(sorted(int(value) for value in guide_state.original_atom_ids[guide_local]))
            counts = _positive_allocation(
                len(guide_ids),
                [float(summaries[local].atom_count) for local in children],
                minimum=1,
            )
            if counts is None:
                continue
            allocated_ids = _lineage_id_allocation(
                guide_ids,
                counts,
                target_representatives,
                guide_state,
                guide_boundary_to_target,
            )
            if allocated_ids is None:
                continue
            guide_summary = guide_state.node_summaries[guide_local]
            histograms = _histogram_allocation(
                guide_summary.hist,
                counts,
                [summaries[local].hist for local in children],
            )
            ring_counts = _positive_allocation(
                int(guide_summary.ring_atom_count),
                [float(summaries[local].ring_atom_count) for local in children],
                minimum=0,
            ) or [0 for _ in children]

            revealed_hist = [0 for _ in spec.bond_vocab]
            child_set = set(children)
            for (left, right), label in edges.items():
                if left in child_set and right in child_set:
                    label_index = (
                        spec.bond_vocab.index(label)
                        if label in spec.bond_vocab
                        else spec.bond_vocab.index("single")
                    )
                    revealed_hist[label_index] += 1
            remaining_internal: list[int] = []
            if guide_summary.internal_bond_hist:
                remaining_internal = [
                    max(
                        0,
                        int(guide_summary.internal_bond_hist[index]) - revealed_hist[index],
                    )
                    for index in range(len(spec.bond_vocab))
                ]
                hidden_total = sum(remaining_internal)
            else:
                hidden_total = max(0, int(guide_summary.hidden_bonds) - sum(revealed_hist))
            hidden_counts = _positive_allocation(
                hidden_total,
                [float(summaries[local].hidden_bonds) for local in children],
                minimum=0,
            ) or [0 for _ in children]
            internal_histograms = (
                _histogram_allocation(
                    remaining_internal,
                    hidden_counts,
                    [[0 for _ in spec.bond_vocab] for _ in children],
                )
                if remaining_internal
                else [[] for _ in children]
            )

            charges = [int(summaries[local].charge) for local in children]
            if charges:
                charges[-1] += int(guide_summary.charge) - sum(charges)
            for position, (local, count) in enumerate(zip(children, counts)):
                lineage_blocks[local] = allocated_ids[position]
                summaries[local] = NodeSummary(
                    hist=tuple(histograms[position]),
                    charge=charges[position],
                    hidden_bonds=int(hidden_counts[position]),
                    atom_count=int(count),
                    ring_atom_count=min(int(count), int(ring_counts[position])),
                    internal_bond_hist=tuple(internal_histograms[position]),
                )

    used_ids = {value for block in lineage_blocks.values() for value in block}
    guide_ids_all = (
        set()
        if guide_state is None
        else {int(value) for block in guide_state.original_atom_ids for value in block}
    )
    next_synthetic_atom = max(guide_ids_all | used_ids, default=-1) + 1
    originals: list[tuple[int, ...]] = []
    for local, summary in enumerate(summaries):
        if local in lineage_blocks and len(lineage_blocks[local]) == int(summary.atom_count):
            originals.append(lineage_blocks[local])
            continue
        block: list[int] = []
        while len(block) < int(summary.atom_count):
            if next_synthetic_atom not in used_ids:
                block.append(next_synthetic_atom)
                used_ids.add(next_synthetic_atom)
            next_synthetic_atom += 1
        originals.append(tuple(block))

    slots: list[AttachmentSlot] = []
    if not resolved_level:
        generated_by_template = {
            template_local: generated_local
            for generated_local, template_local in generated_to_template.items()
        }
        template_slots_by_anchor: dict[int, list[tuple[int, AttachmentSlot]]] = {}
        if template_state is not None:
            for slot_index, slot in enumerate(template_state.attachment_slots):
                template_slots_by_anchor.setdefault(int(slot.anchor), []).append((slot_index, slot))

        template_atom_to_generated: dict[int, int] = {}
        if template_state is not None:
            for generated_local, template_local in generated_to_template.items():
                source_ids = sorted(int(value) for value in template_state.original_atom_ids[template_local])
                generated_ids = sorted(int(value) for value in originals[generated_local])
                if len(source_ids) == len(generated_ids):
                    template_atom_to_generated.update(zip(source_ids, generated_ids))

        for generated_local, old in enumerate(active_indices):
            template_local = generated_to_template.get(generated_local)
            template_slots = (
                [] if template_local is None else template_slots_by_anchor.get(template_local, [])
            )
            participates = (
                bool(template_slots)
                if old in preserved
                else bool(attachment_logits[old].detach().cpu().item() > threshold)
                if attachment_logits is not None
                else bool(template_slots)
            )
            if not participates or not template_slots:
                # The head predicts participation only; it cannot invent the
                # complete certified boundary tuple without a selected object.
                continue
            for template_slot_index, slot in template_slots:
                source_boundaries = slot.boundary_atoms or (slot.anchor,)
                source_labels = slot.boundary_bond_labels or (slot.bond_label,) * len(source_boundaries)
                if any(int(local) not in generated_by_template for local in source_boundaries):
                    continue
                translated_boundary = tuple(generated_by_template[int(local)] for local in source_boundaries)
                translated_originals = tuple(
                    template_atom_to_generated[int(value)]
                    for value in slot.original_atom_ids
                    if int(value) in template_atom_to_generated
                )
                if slot.original_atom_ids and len(translated_originals) != len(slot.original_atom_ids):
                    continue
                translated_boundary_atom_ids = tuple(
                    template_atom_to_generated[int(value)]
                    for value in slot.boundary_atom_ids
                    if int(value) in template_atom_to_generated
                )
                if slot.boundary_atom_ids and len(translated_boundary_atom_ids) != len(slot.boundary_atom_ids):
                    translated_boundary_atom_ids = tuple()
                translated_atom_anchor = (
                    None
                    if slot.atom_anchor is None
                    else template_atom_to_generated.get(int(slot.atom_anchor))
                )
                translated_z1 = (
                    None
                    if slot.z1_node is None
                    else generated_by_template.get(int(slot.z1_node))
                )
                slots.append(
                    AttachmentSlot(
                        anchor=generated_local,
                        summary=slot.summary,
                        bond_label=slot.bond_label,
                        original_atom_ids=translated_originals,
                        atom_anchor=translated_atom_anchor,
                        z1_node=translated_z1,
                        boundary_atoms=translated_boundary,
                        boundary_atom_ids=translated_boundary_atom_ids,
                        boundary_bond_labels=tuple(str(value) for value in source_labels),
                        representative_slot_id=(
                            int(slot.representative_slot_id)
                            if slot.representative_slot_id is not None
                            else int(template_slot_index)
                        ),
                    )
                )

    return CheckpointState(
        stage=template_state.stage if template_state is not None else "h",
        node_summaries=tuple(summaries),
        edges=edges,
        core_flags=tuple(core_flags),
        ring_flags=tuple(ring_flags),
        original_atom_ids=tuple(originals),
        attachment_slots=tuple(slots),
        tensor_slot_ids=tuple(aliases),
        core_empty=core_empty,
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )


def collate_tensor_dicts(items: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in items[0].keys():
        out[key] = torch.stack([item[key] for item in items], dim=0)
    return out
