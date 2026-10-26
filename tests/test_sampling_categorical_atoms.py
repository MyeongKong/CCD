from __future__ import annotations

import torch

from checkpoint_dag_diffusion.global_sampling import sample_global_transition
from checkpoint_dag_diffusion.checkpoint_state import CheckpointState, NodeSummary
from checkpoint_dag_diffusion.model import DenoiserOutput
from checkpoint_dag_diffusion.noise import DiscreteGraphNoise
from checkpoint_dag_diffusion.sampling import _terminal_graph_state, sample_edge_transition
from checkpoint_dag_diffusion.tensorize import TensorSpec


def _noise(spec: TensorSpec) -> DiscreteGraphNoise:
    node_marginal = torch.zeros(spec.node_feature_dim)
    # Deliberately use a non-one-hot block-composition marginal.  It must be
    # projected to an atom identity only for level-0 targets.
    node_marginal[0] = 2.0
    node_marginal[1] = 1.0
    node_marginal[spec.atom_feature_dim + 1] = 2.5
    node_marginal[list(spec.categorical_node_indices)] = 0.5
    return DiscreteGraphNoise(
        timesteps=4,
        node_marginal=node_marginal,
        edge_marginal=torch.tensor([0.8, 0.1, 0.05, 0.025, 0.025]),
        node_noise_scale=0.0,
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=1.0,
        atom_feature_dim=spec.atom_feature_dim,
    )


class _RecordingNoise(DiscreteGraphNoise):
    def __init__(self, base: DiscreteGraphNoise) -> None:
        super().__init__(
            timesteps=base.timesteps,
            node_marginal=base.node_marginal,
            edge_marginal=base.edge_marginal,
            node_noise_scale=base.node_noise_scale,
            categorical_node_indices=base.categorical_node_indices,
            reserved_node_indices=base.reserved_node_indices,
            activity_marginal=base.activity_marginal,
            atom_feature_dim=base.atom_feature_dim,
        )
        self.categorical_atom_calls: list[bool] = []

    def reverse_step(self, **kwargs):
        self.categorical_atom_calls.append(bool(kwargs.get("categorical_atom", False)))
        return super().reverse_step(**kwargs)


class _CopyThroughNoise(DiscreteGraphNoise):
    def reverse_step(self, **kwargs):
        edge = kwargs["clean_edge_logits"].argmax(dim=-1)
        edge = torch.triu(edge, diagonal=1)
        edge = edge + edge.transpose(1, 2)
        return (
            kwargs["clean_x"].clone(),
            edge,
            torch.ones_like(kwargs["current_active"], dtype=torch.bool),
        )


class _DummyDenoiser:
    def __init__(self, edge_dim: int) -> None:
        self.edge_dim = int(edge_dim)

    def __call__(self, x, edge, mask, *_args, **_kwargs) -> DenoiserOutput:
        clean_x = torch.zeros_like(x)
        clean_x[..., 0] = 4.0
        edge_logits = torch.zeros(*edge.shape, self.edge_dim, device=x.device)
        edge_logits[..., 0] = 4.0
        active_logits = torch.full(mask.shape, 4.0, device=x.device)
        auxiliary = torch.zeros_like(active_logits)
        return DenoiserOutput(
            node_features=clean_x,
            edge_logits=edge_logits,
            active_logits=active_logits,
            split_logits=auxiliary,
            attachment_logits=auxiliary,
        )


def _relation(target_level: int) -> dict[str, int]:
    return {
        "guide_level": int(target_level) + 1,
        "target_level": int(target_level),
        "delta": 1,
        "operator_id": 1,
    }


def test_terminal_state_uses_one_hot_atoms_only_for_level_zero_mode():
    torch.manual_seed(7)
    spec = TensorSpec(max_nodes=4)
    noise = _noise(spec)

    fine_x, _, fine_mask = _terminal_graph_state(
        noise, spec, torch.device("cpu"), categorical_atom=True
    )
    fine_atoms = fine_x[0, fine_mask[0], : spec.atom_feature_dim]
    assert torch.all(fine_atoms.sum(dim=-1) == 1)
    assert torch.all((fine_atoms == 0) | (fine_atoms == 1))

    coarse_x, _, coarse_mask = _terminal_graph_state(
        noise, spec, torch.device("cpu"), categorical_atom=False
    )
    coarse_atoms = coarse_x[0, coarse_mask[0], : spec.atom_feature_dim]
    assert torch.allclose(coarse_atoms[:, 0], torch.full_like(coarse_atoms[:, 0], 2.0))
    assert torch.allclose(coarse_atoms[:, 1], torch.full_like(coarse_atoms[:, 1], 1.0))
    # Non-atom summary channels retain their numerical marginal in both modes.
    summary_index = spec.atom_feature_dim + 1
    assert torch.allclose(
        fine_x[0, fine_mask[0], summary_index],
        torch.full_like(fine_x[0, fine_mask[0], summary_index], 2.5),
    )
    assert torch.allclose(
        coarse_x[0, coarse_mask[0], summary_index],
        torch.full_like(coarse_x[0, coarse_mask[0], summary_index], 2.5),
    )


def test_local_transition_enables_categorical_atoms_only_at_level_zero():
    torch.manual_seed(8)
    spec = TensorSpec(max_nodes=4)
    noise = _RecordingNoise(_noise(spec))
    model = _DummyDenoiser(spec.edge_dim)

    sample_edge_transition(
        model=model,
        spec=spec,
        noise=noise,
        relation_edge=_relation(0),
        num_nodes=2,
        guide_state=None,
        device=torch.device("cpu"),
        guidance_weight=1.0,
        steps=1,
    )
    sample_edge_transition(
        model=model,
        spec=spec,
        noise=noise,
        relation_edge=_relation(1),
        num_nodes=2,
        guide_state=None,
        device=torch.device("cpu"),
        guidance_weight=1.0,
        steps=1,
    )
    assert noise.categorical_atom_calls == [True, False]


def test_global_transition_enables_categorical_atoms_at_level_zero():
    torch.manual_seed(9)
    spec = TensorSpec(max_nodes=4)
    noise = _RecordingNoise(_noise(spec))

    sample_global_transition(
        model=_DummyDenoiser(spec.edge_dim),
        spec=spec,
        noise=noise,
        relation_edge=_relation(0),
        global_edge_id=0,
        target_condition_id=0,
        guide_condition_id=-1,
        num_nodes=2,
        guide_state=None,
        device=torch.device("cpu"),
        guidance_weight=1.0,
        steps=1,
    )
    assert noise.categorical_atom_calls == [True]


def test_missing_guide_representative_never_aliases_python_negative_index():
    torch.manual_seed(10)
    spec = TensorSpec(max_nodes=2)
    base = _noise(spec)
    noise = _CopyThroughNoise(
        timesteps=base.timesteps,
        node_marginal=base.node_marginal,
        edge_marginal=base.edge_marginal,
        node_noise_scale=base.node_noise_scale,
        categorical_node_indices=base.categorical_node_indices,
        reserved_node_indices=base.reserved_node_indices,
        activity_marginal=base.activity_marginal,
        atom_feature_dim=base.atom_feature_dim,
    )
    carbon = NodeSummary.from_atom("C", atom_vocab=spec.atom_vocab)
    nitrogen = NodeSummary.from_atom("N", atom_vocab=spec.atom_vocab)
    guide = CheckpointState(
        stage="h",
        node_summaries=(carbon, nitrogen),
        edges={},
        core_flags=(False, False),
        ring_flags=(False, False),
        original_atom_ids=((0,), (1,)),
        tensor_slot_ids=(4, 7),
        atom_vocab=spec.atom_vocab,
        bond_vocab=spec.bond_vocab,
    )
    relation = {
        **_relation(0),
        # Representative 999 is absent.  The old guard accepted lookup -1
        # and silently copied the last (nitrogen) guide row.
        "correspondence": {"preserved_target_to_guide": {"0": 999}},
    }

    local = sample_edge_transition(
        model=_DummyDenoiser(spec.edge_dim),
        spec=spec,
        noise=noise,
        relation_edge=relation,
        num_nodes=2,
        guide_state=guide,
        device=torch.device("cpu"),
        guidance_weight=1.0,
        steps=1,
    )
    global_sample = sample_global_transition(
        model=_DummyDenoiser(spec.edge_dim),
        spec=spec,
        noise=noise,
        relation_edge=relation,
        global_edge_id=0,
        target_condition_id=0,
        guide_condition_id=0,
        num_nodes=2,
        guide_state=guide,
        device=torch.device("cpu"),
        guidance_weight=1.0,
        steps=1,
    )
    for state in (local, global_sample):
        assert all(summary.hist[0] == 1 for summary in state.node_summaries)
        assert all(summary.hist[1] == 0 for summary in state.node_summaries)
