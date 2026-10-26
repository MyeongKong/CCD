from __future__ import annotations

import torch

from checkpoint_dag_diffusion.certified_object import certified_object_from_state
from checkpoint_dag_diffusion.checkpoint_state import AttachmentSlot, CheckpointState, NodeSummary
from checkpoint_dag_diffusion.sampling import _representative_to_current_slots
from checkpoint_dag_diffusion.tensorize import TensorSpec, state_to_tensors, tensors_to_state
from checkpoint_dag_diffusion.train import _mixed_node_loss


def _template(spec: TensorSpec) -> CheckpointState:
    atom = NodeSummary.from_atom("C", atom_vocab=spec.atom_vocab)
    hidden = NodeSummary(
        hist=(2,) + (0,) * (len(spec.atom_vocab) - 1),
        hidden_bonds=1,
        atom_count=2,
    )
    return CheckpointState(
        stage="h",
        node_summaries=(atom, hidden, atom, atom),
        edges={(0, 1): "single", (1, 3): "single"},
        core_flags=(False, True, False, True),
        ring_flags=(False, False, False, False),
        original_atom_ids=((0,), (1, 2), (3,), (4,)),
        attachment_slots=(
            AttachmentSlot(
                anchor=1,
                summary=atom,
                bond_label="single",
                original_atom_ids=(2,),
                boundary_atoms=(1, 3),
                boundary_bond_labels=("single", "single"),
            ),
        ),
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )


def test_mismatched_template_does_not_invent_representative_alignment_or_boundary():
    spec = TensorSpec(max_nodes=4)
    atom_dim = spec.atom_feature_dim
    x = torch.zeros(spec.max_nodes, spec.node_feature_dim)
    x[1, 0] = 3.0
    x[1, atom_dim + 2] = 3.0  # block atom count
    x[1, atom_dim + 4] = 1.0  # unresolved
    x[1, atom_dim + 5] = 1.0  # core
    x[1, atom_dim + 7] = 1.0  # empty-core graph descriptor
    x[3, 2] = 1.0
    x[3, atom_dim + 2] = 5.0  # split head projects this resolved row to one atom
    x[3, atom_dim + 7] = 1.0
    active = torch.tensor([False, True, False, True])
    edges = torch.zeros(spec.max_nodes, spec.max_nodes, dtype=torch.long)
    edges[1, 3] = edges[3, 1] = spec.bond_vocab.index("single")
    split = torch.tensor([-2.0, 2.0, -2.0, -2.0])
    attachment = torch.tensor([-2.0, 2.0, -2.0, -2.0])

    state = tensors_to_state(
        x,
        edges,
        active,
        spec,
        split_logits=split,
        attachment_logits=attachment,
        template_state=_template(spec),
        target_level=2,
    )

    assert state.tensor_slot_ids == (-1, -1)
    assert _representative_to_current_slots(state) == {}
    assert tuple(len(block) for block in state.original_atom_ids) == (3, 1)
    assert tuple(summary.atom_count for summary in state.node_summaries) == (3, 1)
    assert state.core_empty
    assert state.attachment_slots == ()

    certified = certified_object_from_state(state, level=2, mode="pure_core")
    payload = certified.to_dict()
    assert set(("quotient_graph", "partition", "summary", "boundary_slots", "semantic")) <= payload.keys()
    assert payload["boundary_slots"] == []


def test_level_zero_projection_is_atom_categorical_and_has_no_hidden_slots():
    spec = TensorSpec(max_nodes=4)
    atom_dim = spec.atom_feature_dim
    x = torch.zeros(spec.max_nodes, spec.node_feature_dim)
    x[2, 1] = 0.1
    x[2, 2] = 3.0  # oxygen is the selected atom category
    x[2, atom_dim + 1] = 4.0
    x[2, atom_dim + 2] = 7.0
    x[2, atom_dim + 4] = 1.0
    active = torch.tensor([False, False, True, False])
    split = torch.full((4,), 4.0)
    attachment = torch.full((4,), 4.0)

    state = tensors_to_state(
        x,
        torch.zeros(4, 4, dtype=torch.long),
        active,
        spec,
        split_logits=split,
        attachment_logits=attachment,
        template_state=_template(spec),
        target_level=0,
    )

    summary = state.node_summaries[0]
    assert summary.hist[2] == 1 and sum(summary.hist) == 1
    assert summary.atom_count == 1
    assert summary.hidden_bonds == 0
    assert state.attachment_slots == ()
    assert state.tensor_slot_ids == (-1,)


def test_visible_isomorphism_aligns_permuted_active_slots_and_remaps_boundary_metadata():
    spec = TensorSpec(max_nodes=5)
    atom_dim = spec.atom_feature_dim
    carbon = NodeSummary.from_atom("C", atom_vocab=spec.atom_vocab)
    nitrogen = NodeSummary.from_atom("N", atom_vocab=spec.atom_vocab)
    carbon_pair = NodeSummary(
        hist=(2,) + (0,) * (atom_dim - 1),
        hidden_bonds=1,
        atom_count=2,
        internal_bond_hist=(0, 1, 0, 0, 0),
    )
    template = CheckpointState(
        stage="h",
        node_summaries=(carbon, carbon_pair, nitrogen),
        edges={(0, 1): "single", (1, 2): "double"},
        core_flags=(False, True, False),
        ring_flags=(False, False, False),
        original_atom_ids=((0,), (1, 2), (3,)),
        attachment_slots=(
            AttachmentSlot(
                anchor=1,
                summary=carbon,
                bond_label="single",
                original_atom_ids=(1,),
                atom_anchor=1,
                boundary_atoms=(1, 2),
                boundary_atom_ids=(1, 3),
                boundary_bond_labels=("single", "double"),
            ),
        ),
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )

    x = torch.zeros(spec.max_nodes, spec.node_feature_dim)
    # Generated local order is [N, C, C2] at unrelated padded positions
    # [1, 3, 4].  The only valid representative order is [2, 0, 1].
    x[1, 1] = 1.0
    x[1, atom_dim + 2] = 1.0
    x[3, 0] = 1.0
    x[3, atom_dim + 2] = 1.0
    x[4, 0] = 2.0
    x[4, atom_dim + 1] = 1.0
    x[4, atom_dim + 2] = 2.0
    x[4, atom_dim + 4] = 1.0
    x[4, atom_dim + 5] = 1.0
    active = torch.tensor([False, True, False, True, True])
    edges = torch.zeros(5, 5, dtype=torch.long)
    edges[3, 4] = edges[4, 3] = spec.bond_vocab.index("single")
    edges[4, 1] = edges[1, 4] = spec.bond_vocab.index("double")
    split = torch.full((5,), -4.0)
    split[4] = 4.0
    attachment = torch.full((5,), -4.0)
    attachment[4] = 4.0

    state = tensors_to_state(
        x,
        edges,
        active,
        spec,
        split_logits=split,
        attachment_logits=attachment,
        template_state=template,
        target_level=1,
    )

    assert state.tensor_slot_ids == (2, 0, 1)
    assert _representative_to_current_slots(state) == {2: 0, 0: 1, 1: 2}
    assert state.node_summaries[2].internal_bond_hist == ()
    assert state.node_summaries[2].hidden_bonds == 1
    assert len(state.attachment_slots) == 1
    slot = state.attachment_slots[0]
    assert slot.anchor == 2
    assert slot.boundary_atoms == (2, 0)
    assert slot.original_atom_ids == (2,)
    assert slot.atom_anchor == 2
    assert slot.boundary_atom_ids == (2, 0)
    assert slot.boundary_bond_labels == ("single", "double")
    assert slot.representative_slot_id == 0
    assert certified_object_from_state(state).quotient_graph_hash == certified_object_from_state(
        template
    ).quotient_graph_hash


def test_preserved_object_fields_override_hostile_auxiliary_logits():
    spec = TensorSpec(max_nodes=1)
    carbon = NodeSummary.from_atom("C", atom_vocab=spec.atom_vocab)
    pair = NodeSummary(
        hist=(2,) + (0,) * (spec.atom_feature_dim - 1),
        hidden_bonds=1,
        atom_count=2,
        internal_bond_hist=(0, 1, 0, 0, 0),
    )
    guide = CheckpointState(
        stage="h",
        node_summaries=(pair,),
        edges={},
        core_flags=(True,),
        ring_flags=(False,),
        original_atom_ids=((10, 11),),
        attachment_slots=(
            AttachmentSlot(
                anchor=0,
                summary=carbon,
                bond_label="double",
                original_atom_ids=(11,),
                atom_anchor=10,
                boundary_atoms=(0,),
                boundary_atom_ids=(10,),
                boundary_bond_labels=("double",),
            ),
        ),
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )
    tensors = state_to_tensors(guide, spec)
    state = tensors_to_state(
        tensors["x"],
        tensors["edge"],
        tensors["mask"],
        spec,
        split_logits=torch.tensor([-20.0]),
        attachment_logits=torch.tensor([-20.0]),
        template_state=guide,
        target_level=1,
        preserved_target_slots={0},
        guide_state=guide,
        correspondence={
            "guide_to_target": {"0": [0]},
            "target_to_guide": {"0": 0},
            "preserved_target_to_guide": {"0": 0},
            "guide_boundary_to_target": {"0": [0]},
        },
    )

    assert state.tensor_slot_ids == (0,)
    assert state.node_summaries == (pair,)
    assert state.original_atom_ids == ((10, 11),)
    assert len(state.attachment_slots) == 1
    assert state.attachment_slots[0].bond_label == "double"
    assert state.attachment_slots[0].original_atom_ids == (11,)
    assert state.attachment_slots[0].atom_anchor == 10
    assert state.attachment_slots[0].boundary_atom_ids == (10,)


def test_binary_object_fields_are_trained_as_categorical_logits():
    spec = TensorSpec(max_nodes=2)
    pred = torch.zeros(1, 2, spec.node_feature_dim, requires_grad=True)
    target = torch.zeros_like(pred)
    target[0, 0, spec.categorical_node_indices[0]] = 1.0
    mask = torch.tensor([[True, False]])
    relation = torch.tensor([[2, 1, 1, 1]])

    loss = _mixed_node_loss(pred, target, mask, relation, spec)
    expected = len(spec.categorical_node_indices) * torch.log(torch.tensor(2.0))
    assert torch.allclose(loss.detach(), expected, atol=1e-6)
    loss.backward()
    assert pred.grad is not None
    assert float(pred.grad[0, 0, spec.categorical_node_indices[0]]) < 0.0
