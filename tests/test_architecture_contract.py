from __future__ import annotations

import torch

from checkpoint_dag_diffusion.global_codebook import (
    CheckpointCodebookModel,
    contrastive_hash_loss,
)
from checkpoint_dag_diffusion.model import CheckpointDAGDenoiser, GraphStateEncoder
from checkpoint_dag_diffusion.noise import DiscreteGraphNoise
from checkpoint_dag_diffusion.tensorize import TensorSpec


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_reference_model_parameter_counts_match_supplement():
    codebook = CheckpointCodebookModel(
        node_feature_dim=21,
        edge_dim=5,
        max_nodes=32,
        codebook_size=512,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    assert _parameter_count(codebook) == 2_076_738

    # The supplementary parameter count was measured with 484 populated
    # global identifiers.  The edge table is reported separately because its
    # size depends on the observed level-augmented DAG.
    denoiser = CheckpointDAGDenoiser(
        node_feature_dim=21,
        edge_dim=5,
        hidden_dim=192,
        num_layers=4,
        guide_num_layers=2,
        num_heads=6,
        dropout=0.1,
        max_timesteps=200,
        max_level=16,
        max_delta=16,
        num_global_codes=484,
        max_global_edges=0,
    )
    assert _parameter_count(denoiser) == 3_606_557

    with_edge_table = CheckpointDAGDenoiser(
        node_feature_dim=21,
        edge_dim=5,
        hidden_dim=192,
        num_layers=4,
        guide_num_layers=2,
        num_heads=6,
        dropout=0.1,
        max_timesteps=200,
        max_level=16,
        max_delta=16,
        num_global_codes=484,
        max_global_edges=7,
    )
    assert _parameter_count(with_edge_table) - _parameter_count(denoiser) == 192 * (7 + 1)


def test_graph_transformer_is_permutation_equivariant():
    torch.manual_seed(4)
    encoder = GraphStateEncoder(
        node_feature_dim=21,
        edge_dim=5,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_timesteps=8,
        max_level=8,
        max_delta=8,
    ).eval()
    x = torch.randn(1, 6, 21)
    edge = torch.randint(0, 5, (1, 6, 6))
    edge = torch.triu(edge, diagonal=1)
    edge = edge + edge.transpose(1, 2)
    mask = torch.tensor([[True, True, True, True, False, False]])
    t = torch.tensor([3])
    relation = torch.tensor([[4, 2, 2, 1]])
    permutation = torch.tensor([2, 0, 3, 1, 5, 4])

    with torch.no_grad():
        reference = encoder(x, edge, mask, t, relation)
        permuted = encoder(
            x[:, permutation],
            edge[:, permutation][:, :, permutation],
            mask[:, permutation],
            t,
            relation,
        )
    assert torch.allclose(permuted, reference[:, permutation], atol=1e-5, rtol=1e-5)


def test_graph_transformer_edge_aggregation_excludes_self_and_padding():
    torch.manual_seed(5)
    encoder = GraphStateEncoder(
        node_feature_dim=21,
        edge_dim=5,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_timesteps=8,
        max_level=8,
        max_delta=8,
    ).eval()
    x = torch.randn(1, 4, 21)
    mask = torch.tensor([[True, True, False, False]])
    edge = torch.zeros(1, 4, 4, dtype=torch.long)
    edge[:, 0, 1] = edge[:, 1, 0] = 1
    t = torch.tensor([3])
    relation = torch.tensor([[4, 2, 2, 1]])

    # Change every diagonal and every pair incident to padding. These entries
    # are outside A_Q over active quotient nodes and must not affect the output.
    contaminated = edge.clone()
    contaminated[:, torch.arange(4), torch.arange(4)] = torch.tensor([1, 2, 3, 4])
    contaminated[:, :2, 2:] = torch.tensor([[[2, 3], [4, 2]]])
    contaminated[:, 2:, :2] = contaminated[:, :2, 2:].transpose(1, 2)
    contaminated[:, 2, 3] = contaminated[:, 3, 2] = 4

    with torch.no_grad():
        reference = encoder(x, edge, mask, t, relation)
        observed = encoder(x, contaminated, mask, t, relation)
    assert torch.allclose(observed, reference, atol=1e-6, rtol=1e-6)


def test_hash_loss_keeps_unit_margin_negatives_without_positive_pair():
    embedding = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], requires_grad=True)
    loss = contrastive_hash_loss(embedding, ["a", "b", "c"], margin=1.0)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()


def test_mixed_state_corruption_and_reverse_transition_include_activity():
    torch.manual_seed(12)
    spec = TensorSpec(max_nodes=4)
    node_marginal = torch.zeros(spec.node_feature_dim)
    node_marginal[0] = 1.0
    node_marginal[list(spec.categorical_node_indices)] = 0.5
    edge_marginal = torch.tensor([0.7, 0.2, 0.05, 0.025, 0.025])
    noise = DiscreteGraphNoise(
        timesteps=8,
        node_marginal=node_marginal,
        edge_marginal=edge_marginal,
        categorical_node_indices=spec.categorical_node_indices,
        reserved_node_indices=spec.reserved_node_indices,
        activity_marginal=0.5,
    )
    x = torch.zeros(1, 4, spec.node_feature_dim)
    x[:, :2, 0] = 1.0
    clean_mask = torch.tensor([[True, True, False, False]])
    clean_edge = torch.zeros(1, 4, 4, dtype=torch.long)
    clean_edge[:, 0, 1] = clean_edge[:, 1, 0] = 1
    corrupted = noise.corrupt(x, clean_edge, clean_mask, t=torch.tensor([8]))
    assert corrupted.active_t.shape == clean_mask.shape
    assert corrupted.active_t.dtype == torch.bool
    assert torch.equal(corrupted.edge_t, corrupted.edge_t.transpose(1, 2))

    clean_edge_logits = torch.zeros(1, 4, 4, spec.edge_dim)
    clean_edge_logits[..., 0] = 2.0
    clean_active_logits = torch.tensor([[2.0, 2.0, -2.0, -2.0]])
    x_s, edge_s, active_s = noise.reverse_step(
        clean_x=x,
        clean_edge_logits=clean_edge_logits,
        clean_active_logits=clean_active_logits,
        current_x=corrupted.x_t,
        current_edge=corrupted.edge_t,
        current_active=corrupted.active_t,
        t=8,
        s=0,
        none_idx=0,
        edge_temperature=0.9,
        none_bias=2.0,
        max_degree=4,
    )
    assert x_s.shape == x.shape
    assert active_s.shape == clean_mask.shape
    assert bool(active_s.any())
    assert torch.equal(edge_s, edge_s.transpose(1, 2))
    assert torch.all(x_s[~active_s] == 0)
