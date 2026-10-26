from __future__ import annotations

import copy

import torch

from checkpoint_dag_diffusion.certified_object import (
    CertifiedCheckpointObject,
    certified_object_from_state,
    quotient_graph_match,
)
from checkpoint_dag_diffusion.checkpoint_state import AttachmentSlot, CheckpointState, NodeSummary
from checkpoint_dag_diffusion.dag_topology import build_checkpoint_dag, refinement_correspondence
from checkpoint_dag_diffusion.recontract import recontract_and_compare
from checkpoint_dag_diffusion.tensorize import (
    TensorSpec,
    certified_object_to_tensors,
    relation_to_tensor,
)
from checkpoint_dag_diffusion.topology import MoleculeGraph


ATOM_WIDTH = 12


def _summary(hist: tuple[int, ...], *, hidden: int, atoms: int, ring_atoms: int) -> NodeSummary:
    return NodeSummary(hist=hist, charge=0, hidden_bonds=hidden, atom_count=atoms, ring_atom_count=ring_atoms)


def _published_checkpoint_state() -> CheckpointState:
    carbon = (1,) + (0,) * (ATOM_WIDTH - 1)
    two_carbon = (2,) + (0,) * (ATOM_WIDTH - 1)
    nitrogen = (0, 1) + (0,) * (ATOM_WIDTH - 2)
    c2o = (2, 0, 1) + (0,) * (ATOM_WIDTH - 3)
    oxygen = (0, 0, 1) + (0,) * (ATOM_WIDTH - 3)
    return CheckpointState(
        stage="h",
        node_summaries=(
            _summary(two_carbon, hidden=1, atoms=2, ring_atoms=1),
            _summary(carbon, hidden=0, atoms=1, ring_atoms=1),
            _summary(nitrogen, hidden=0, atoms=1, ring_atoms=1),
            _summary(nitrogen, hidden=0, atoms=1, ring_atoms=1),
            _summary(c2o, hidden=2, atoms=3, ring_atoms=1),
        ),
        edges={
            (0, 1): "aromatic",
            (0, 4): "aromatic",
            (1, 2): "aromatic",
            (2, 3): "aromatic",
            (3, 4): "aromatic",
        },
        core_flags=(True,) * 5,
        ring_flags=(True,) * 5,
        original_atom_ids=((0, 1), (2,), (3,), (4,), (5, 6, 7)),
        attachment_slots=(
            AttachmentSlot(anchor=0, summary=_summary(carbon, hidden=0, atoms=1, ring_atoms=0), original_atom_ids=(0,)),
            AttachmentSlot(
                anchor=4,
                summary=_summary(tuple(a + b for a, b in zip(carbon, oxygen)), hidden=1, atoms=2, ring_atoms=0),
                original_atom_ids=(6, 7),
            ),
        ),
    )


def test_published_certification_fixture_and_object_tensorization():
    state = _published_checkpoint_state()
    obj = certified_object_from_state(state, level=2, mode="pure_core")
    assert obj.partition == [list(block) for block in state.original_atom_ids]
    assert obj.quotient_graph_hash == "02653c67f1aede4407496d0a5ebde52fed8328e534beab7aef4ab7cccd8939fc"
    assert obj.semantic_hash == "4fd7d323251175d4119663c8a6de518f48fd8dc1da4a4f45fb74b521e8421692"
    assert obj.certificate_hash == "1e4b42a146051b35355b7c873b69997d57d654de57c32d73c3acb64b8a25a666"

    tensors = certified_object_to_tensors(obj, TensorSpec(max_nodes=32))
    assert tensors["x"].shape == (32, 21)
    assert tensors["edge"].shape == (32, 32)
    assert tensors["mask"].shape == (32,)
    assert int(tensors["mask"].sum()) == 5
    assert torch.equal(tensors["edge"], tensors["edge"].T)


def test_visible_h_labels_are_separate_from_per_node_sigma():
    state = _published_checkpoint_state()
    obj = certified_object_from_state(state, level=2, mode="pure_core")
    payload = obj.to_dict()

    labels = [row["label"] for row in payload["quotient_graph"]["nodes"]]
    assert labels == ["supernode", "atom:0", "atom:1", "atom:1", "supernode"]
    assert all("h=" not in label and "|q=" not in label and "|hb=" not in label for label in labels)
    assert payload["structural_summaries"] == [summary.to_dict() for summary in state.node_summaries]

    restored = CertifiedCheckpointObject.from_dict(payload)
    assert restored.structural_summaries == state.node_summaries
    assert [row["label"] for row in restored.quotient_graph.nodes] == labels


def test_tensorization_reads_sigma_without_parsing_visible_h_labels():
    state = _published_checkpoint_state()
    obj = CertifiedCheckpointObject.from_dict(
        certified_object_from_state(state, level=2, mode="pure_core").to_dict()
    )

    tensors = certified_object_to_tensors(obj, TensorSpec(max_nodes=32))
    expected_sigma_features = torch.tensor(
        [summary.as_feature() for summary in state.node_summaries], dtype=torch.float32
    )
    assert torch.equal(tensors["x"][: state.num_nodes, : expected_sigma_features.shape[1]], expected_sigma_features)


def test_legacy_v2_summary_labels_migrate_to_separate_sigma():
    state = _published_checkpoint_state()
    expected = certified_object_from_state(state, level=2, mode="pure_core")
    legacy = expected.to_dict()
    legacy.pop("structural_summaries")
    for row, summary in zip(legacy["quotient_graph"]["nodes"], state.node_summaries):
        row["label"] = summary.label()

    restored = CertifiedCheckpointObject.from_dict(legacy)
    assert restored.structural_summaries == state.node_summaries
    assert [row["label"] for row in restored.quotient_graph.nodes] == [
        "supernode",
        "atom:0",
        "atom:1",
        "atom:1",
        "supernode",
    ]
    assert restored.quotient_graph_hash == expected.quotient_graph_hash
    assert restored.semantic_hash == expected.semantic_hash
    assert restored.certificate_hash == expected.certificate_hash


def test_h_and_certified_metadata_are_separate():
    state = _published_checkpoint_state()
    changed_boundary = CheckpointState(
        stage=state.stage,
        node_summaries=state.node_summaries,
        edges=dict(state.edges),
        core_flags=state.core_flags,
        ring_flags=state.ring_flags,
        original_atom_ids=state.original_atom_ids,
        attachment_slots=(
            AttachmentSlot(
                anchor=0,
                summary=state.attachment_slots[0].summary,
                bond_label="double",
                original_atom_ids=state.attachment_slots[0].original_atom_ids,
            ),
            state.attachment_slots[1],
        ),
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    original = certified_object_from_state(state, level=2, mode="pure_core")
    modified = certified_object_from_state(changed_boundary, level=2, mode="pure_core")
    assert original.quotient_graph_hash == modified.quotient_graph_hash
    assert original.semantic_hash == modified.semantic_hash
    assert original.certificate_hash != modified.certificate_hash

    spec = TensorSpec()
    left = certified_object_to_tensors(original, spec)
    right = certified_object_to_tensors(modified, spec)
    for key in ("x", "edge", "mask"):
        assert torch.equal(left[key], right[key])


def test_quotient_hash_depends_only_on_visible_h():
    state = _published_checkpoint_state()
    changed_sigma = list(state.node_summaries)
    changed_sigma[0] = NodeSummary(
        hist=(1, 1) + (0,) * (ATOM_WIDTH - 2),
        charge=1,
        hidden_bonds=0,
        atom_count=2,
        ring_atom_count=0,
    )
    modified = CheckpointState(
        stage=state.stage,
        node_summaries=tuple(changed_sigma),
        edges=dict(state.edges),
        core_flags=state.core_flags,
        ring_flags=state.ring_flags,
        original_atom_ids=state.original_atom_ids,
        attachment_slots=state.attachment_slots,
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    original_obj = certified_object_from_state(state, level=2, mode="pure_core").to_dict()
    modified_obj = certified_object_from_state(modified, level=2, mode="pure_core").to_dict()

    assert original_obj["quotient_graph"] == modified_obj["quotient_graph"]
    assert original_obj["quotient_graph_hash"] == modified_obj["quotient_graph_hash"]
    assert quotient_graph_match(original_obj, modified_obj)
    assert original_obj["certificate_hash"] != modified_obj["certificate_hash"]


def test_quotient_match_verifies_payload_isomorphism_after_hash_equality():
    obj = certified_object_from_state(
        _published_checkpoint_state(), level=2, mode="pure_core"
    ).to_dict()
    forged = copy.deepcopy(obj)
    forged["quotient_graph"]["nodes"][0]["core"] = False

    assert forged["quotient_graph_hash"] == obj["quotient_graph_hash"]
    assert not quotient_graph_match(obj, forged)


def test_exact_certificate_records_boundary_anchor_placement():
    state = _published_checkpoint_state()
    first = state.attachment_slots[0]
    moved = CheckpointState(
        stage=state.stage,
        node_summaries=state.node_summaries,
        edges=dict(state.edges),
        core_flags=state.core_flags,
        ring_flags=state.ring_flags,
        original_atom_ids=state.original_atom_ids,
        attachment_slots=(
            AttachmentSlot(
                anchor=1,
                summary=first.summary,
                bond_label=first.bond_label,
                original_atom_ids=first.original_atom_ids,
            ),
            state.attachment_slots[1],
        ),
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    original_obj = certified_object_from_state(state, level=2, mode="pure_core")
    moved_obj = certified_object_from_state(moved, level=2, mode="pure_core")

    assert original_obj.quotient_graph_hash == moved_obj.quotient_graph_hash
    assert original_obj.semantic_hash == moved_obj.semantic_hash
    assert original_obj.certificate_hash != moved_obj.certificate_hash


def test_exact_certificate_associates_sigma_and_partition_with_visible_nodes():
    state = _published_checkpoint_state()
    summaries = list(state.node_summaries)
    blocks = list(state.original_atom_ids)
    summaries[0], summaries[4] = summaries[4], summaries[0]
    blocks[0], blocks[4] = blocks[4], blocks[0]
    moved_metadata = CheckpointState(
        stage=state.stage,
        node_summaries=tuple(summaries),
        edges=dict(state.edges),
        core_flags=state.core_flags,
        ring_flags=state.ring_flags,
        original_atom_ids=tuple(blocks),
        attachment_slots=state.attachment_slots,
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    original = certified_object_from_state(state, level=2, mode="pure_core")
    moved = certified_object_from_state(moved_metadata, level=2, mode="pure_core")
    assert original.quotient_graph_hash == moved.quotient_graph_hash
    assert original.summary == moved.summary
    assert original.certificate_hash != moved.certificate_hash


def test_exact_certificate_excludes_raw_boundary_source_ids():
    state = _published_checkpoint_state()

    def with_source_offset(offset: int) -> CheckpointState:
        slots = tuple(
            AttachmentSlot(
                anchor=slot.anchor,
                summary=slot.summary,
                bond_label=slot.bond_label,
                original_atom_ids=tuple(value + offset for value in slot.original_atom_ids),
                atom_anchor=slot.anchor + offset,
                z1_node=slot.z1_node,
                boundary_atoms=(slot.anchor,),
                boundary_atom_ids=(slot.anchor + offset,),
                boundary_bond_labels=(slot.bond_label,),
            )
            for slot in state.attachment_slots
        )
        return CheckpointState(
            stage=state.stage,
            node_summaries=state.node_summaries,
            edges=dict(state.edges),
            core_flags=state.core_flags,
            ring_flags=state.ring_flags,
            original_atom_ids=tuple(
                tuple(value + offset for value in block)
                for block in state.original_atom_ids
            ),
            attachment_slots=slots,
            core_empty=state.core_empty,
            atom_vocab=state.atom_vocab,
            bond_vocab=state.bond_vocab,
        )

    first = certified_object_from_state(with_source_offset(100), level=2, mode="pure_core")
    second = certified_object_from_state(with_source_offset(500), level=2, mode="pure_core")
    assert first.certificate_hash == second.certificate_hash


def test_checkpoint_hashes_are_invariant_to_quotient_node_permutation():
    state = _published_checkpoint_state()
    new_to_old = (2, 4, 0, 3, 1)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    permuted_slots = tuple(
        AttachmentSlot(
            anchor=old_to_new[slot.anchor],
            summary=slot.summary,
            bond_label=slot.bond_label,
            original_atom_ids=slot.original_atom_ids,
            atom_anchor=slot.atom_anchor,
            z1_node=slot.z1_node,
            boundary_atoms=tuple(old_to_new[index] for index in slot.boundary_atoms),
            boundary_atom_ids=slot.boundary_atom_ids,
            boundary_bond_labels=slot.boundary_bond_labels,
        )
        for slot in state.attachment_slots
    )
    permuted = CheckpointState(
        stage=state.stage,
        node_summaries=tuple(state.node_summaries[old] for old in new_to_old),
        edges={
            (old_to_new[left], old_to_new[right]): label
            for (left, right), label in state.edges.items()
        },
        core_flags=tuple(state.core_flags[old] for old in new_to_old),
        ring_flags=tuple(state.ring_flags[old] for old in new_to_old),
        original_atom_ids=tuple(state.original_atom_ids[old] for old in new_to_old),
        attachment_slots=permuted_slots,
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    original_obj = certified_object_from_state(state, level=2, mode="pure_core")
    permuted_obj = certified_object_from_state(permuted, level=2, mode="pure_core")

    assert original_obj.quotient_graph_hash == permuted_obj.quotient_graph_hash
    assert original_obj.semantic_hash == permuted_obj.semantic_hash
    assert original_obj.certificate_hash == permuted_obj.certificate_hash
    assert quotient_graph_match(original_obj.to_dict(), permuted_obj.to_dict())


def test_relation_order_is_guide_then_fine():
    relation = relation_to_tensor(
        {"guide_level": 4, "target_level": 1, "delta": 3, "operator_id": 2}
    )
    assert relation.tolist() == [4, 1, 3, 2]


def test_correspondence_is_partition_based_not_prefix_based():
    atom = NodeSummary.from_atom("C")
    finer = CheckpointState(
        stage="h",
        node_summaries=(atom, atom, atom),
        edges={(0, 2): "single", (1, 2): "single"},
        core_flags=(False, False, False),
        ring_flags=(False, False, False),
        original_atom_ids=((2,), (0,), (1,)),
    )
    coarser = CheckpointState(
        stage="h",
        node_summaries=(atom, NodeSummary.merge((atom, atom), extra_hidden_bonds=1)),
        edges={(0, 1): "single"},
        core_flags=(False, False),
        ring_flags=(False, False),
        original_atom_ids=((0,), (1, 2)),
    )
    mapping = refinement_correspondence(finer, coarser)
    assert mapping.guide_to_target == {0: (1,), 1: (0, 2)}
    assert mapping.preserved_target_to_guide == {1: 0}
    assert mapping.preserved_target_to_guide != {0: 0}


def test_recontraction_recovers_every_requested_path_object():
    graph = MoleculeGraph(
        atom_symbols=("C", "C", "C"),
        charges=(0, 0, 0),
        edges={(0, 1): "single", (1, 2): "single"},
        source_id="propane",
    )
    dag = build_checkpoint_dag(graph)
    requested = [checkpoint.to_dict()["certified_object"] for checkpoint in reversed(dag.checkpoints)]
    result = recontract_and_compare(dag.checkpoints[0].state, requested)
    assert result["recontractable"]
    assert result["path_semantic"]
    assert result["path_exact"]
    assert all(row["qmatch"] and row["semantic"] and row["exact"] for row in result["comparisons"])
