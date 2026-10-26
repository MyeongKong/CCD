import torch

from checkpoint_dag_diffusion.dataset import CheckpointDAGEdgeDataset, collate_edges
from checkpoint_dag_diffusion.model import CheckpointDAGDenoiser
from checkpoint_dag_diffusion.noise import DiscreteGraphNoise, estimate_marginals
from checkpoint_dag_diffusion.preprocess import build_records_from_graphs, load_graph_json, write_corpus
from checkpoint_dag_diffusion.train import compute_loss
from checkpoint_dag_diffusion.tensorize import TensorSpec


def test_dag_model_forward_and_loss(tmp_path):
    graphs = load_graph_json("data/toy_graphs.json", max_molecules=2)
    records = build_records_from_graphs(graphs, max_nodes=16)
    write_corpus(records, tmp_path, max_nodes=16)
    spec = TensorSpec(max_nodes=16)
    dataset = CheckpointDAGEdgeDataset(tmp_path, split="train", spec=spec)
    batch = collate_edges([dataset[0], dataset[1]])
    node_marginal, edge_marginal = estimate_marginals(dataset, spec.node_feature_dim, spec.edge_dim)
    noise = DiscreteGraphNoise(8, node_marginal, edge_marginal)
    noisy = noise.corrupt(batch["target"]["x"], batch["target"]["edge"], batch["target"]["mask"])
    model = CheckpointDAGDenoiser(
        node_feature_dim=spec.node_feature_dim,
        edge_dim=spec.edge_dim,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_timesteps=8,
        max_level=8,
        max_delta=8,
    )
    out = model(
        noisy.x_t,
        noisy.edge_t,
        batch["target"]["mask"],
        noisy.t,
        batch["relation"],
        batch["guide"]["x"],
        batch["guide"]["edge"],
        batch["guide"]["mask"],
        batch["has_guide"],
    )
    loss, logs = compute_loss(out, batch, spec, weights={})
    assert torch.isfinite(loss)
    assert logs["loss"] > 0
