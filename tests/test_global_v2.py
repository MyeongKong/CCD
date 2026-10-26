import json

import torch

from checkpoint_dag_diffusion.certified_object import certified_object_from_state
from checkpoint_dag_diffusion.checkpoint_state import CheckpointState
from checkpoint_dag_diffusion.global_eval import evaluate_global_samples
from checkpoint_dag_diffusion.global_codebook import CheckpointCodebookModel, assign_codebook, train_codebook
from checkpoint_dag_diffusion.global_graph import build_global_graph, build_hash_assignments, load_global_graph
from checkpoint_dag_diffusion.global_sampling import sample_global_pipeline
from checkpoint_dag_diffusion.global_train import train_global_from_config
from checkpoint_dag_diffusion.preprocess import build_records_from_graphs, load_graph_json, write_corpus
from checkpoint_dag_diffusion.tensorize import TensorSpec, state_to_tensors


def _toy_corpus(tmp_path):
    graphs = load_graph_json("data/toy_graphs.json", max_molecules=3)
    records = build_records_from_graphs(graphs, max_nodes=16)
    corpus = tmp_path / "corpus"
    write_corpus(records, corpus, max_nodes=16)
    return corpus, records


def test_hash_global_assignments_merge_identical_hashes(tmp_path):
    corpus, records = _toy_corpus(tmp_path)
    result = build_hash_assignments(corpus, tmp_path / "hash_global")
    payload = torch.load(result["assignments"], map_location="cpu")
    hash_to_gid = {}
    for item in payload["assignments"]:
        h = item["hash"]
        gid = item["global_id"]
        if h in hash_to_gid:
            assert hash_to_gid[h] == gid
        hash_to_gid[h] = gid
    assert len(payload["assignments"]) == sum(len(record["checkpoints"]) for record in records)


def test_certified_object_hash_ignores_raw_original_atom_ids(tmp_path):
    corpus, records = _toy_corpus(tmp_path)
    state = CheckpointState.from_dict(records[0]["checkpoints"][0]["state"])
    shifted = CheckpointState(
        stage=state.stage,
        node_summaries=state.node_summaries,
        edges=dict(state.edges),
        core_flags=state.core_flags,
        ring_flags=state.ring_flags,
        original_atom_ids=tuple(tuple(atom_id + 100 for atom_id in ids) for ids in state.original_atom_ids),
        attachment_slots=state.attachment_slots,
        core_empty=state.core_empty,
        atom_vocab=state.atom_vocab,
        bond_vocab=state.bond_vocab,
    )
    cert_a = certified_object_from_state(state, level=0, mode="full").to_dict()
    cert_b = certified_object_from_state(shifted, level=0, mode="full").to_dict()
    assert cert_a["certificate_hash"] == cert_b["certificate_hash"]
    assert cert_a["format"].endswith(".v2")


def test_learned_codebook_assignment_range(tmp_path):
    corpus, _ = _toy_corpus(tmp_path)
    out = tmp_path / "codebook"
    trained = train_codebook(
        corpus,
        out,
        codebook_size=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    assigned = assign_codebook(corpus, trained["codebook"], out, batch_size=4, device="cpu")
    payload = torch.load(assigned["assignments"], map_location="cpu")
    ids = [int(item["global_id"]) for item in payload["assignments"]]
    assert ids
    assert min(ids) >= 0
    assert max(ids) < 8


def test_global_graph_aggregation_preserves_local_edge_count(tmp_path):
    corpus, records = _toy_corpus(tmp_path)
    assigned = build_hash_assignments(corpus, tmp_path / "global")
    graph_result = build_global_graph(corpus, assigned["assignments"], tmp_path / "global")
    graph = load_global_graph(graph_result["global_graph"])
    assert graph["num_local_edge_observations"] == sum(len(record["dag_edges"]) for record in records)
    assert sum(int(edge["count"]) for edge in graph["global_edges"]) == graph["num_local_edge_observations"]
    assert all("prototype_certificate" in node for node in graph["global_nodes"])
    assert "mean_topology_purity" in graph_result["manifest"]


def test_global_model_smoke_and_controlled_sampling(tmp_path):
    corpus, _ = _toy_corpus(tmp_path)
    global_dir = tmp_path / "global"
    assigned = build_hash_assignments(corpus, global_dir)
    graph_result = build_global_graph(corpus, assigned["assignments"], global_dir)
    graph = load_global_graph(graph_result["global_graph"])
    start_code = int(next(iter(graph["coarsest_prior"].keys())))
    config = {
        "seed": 0,
        "diffusion": {"timesteps": 4, "node_noise_scale": 0.01},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "train": {"device": "cpu", "epochs": 1, "batch_size": 4, "lr": 0.001, "cond_drop_prob": 0.1},
        "loss": {"node": 1.0, "edge": 1.0, "active": 0.25},
    }
    config_path = tmp_path / "global_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    trained = train_global_from_config(
        config_path,
        corpus_dir=corpus,
        global_graph_path=graph_result["global_graph"],
        assignments_path=assigned["assignments"],
        run_dir=tmp_path / "global_train",
        device="cpu",
    )
    summary = sample_global_pipeline(
        checkpoint=trained["checkpoint"],
        global_graph_path=graph_result["global_graph"],
        out_dir=tmp_path / "samples",
        num_samples=2,
        start_code=start_code,
        device="cpu",
        max_path_steps=3,
        seed=0,
    )
    data = json.loads((tmp_path / "samples" / "samples.json").read_text(encoding="utf-8"))
    assert len(data) == 2
    assert all(item["start_global_id"] == start_code for item in data)
    assert all(item["target_certificate_hash"] for item in data)
    assert all(item["path_certificate_hashes"] for item in data)
    valid_edges = {
        (int(edge["guide_global_id"]), int(edge["target_global_id"]))
        for edge in graph["global_edges"]
    }
    for item in data:
        trace = item["path_trace"]
        assert trace
        for row in trace:
            assert (int(row["guide_global_id"]), int(row["target_global_id"])) in valid_edges
            realized = row["realized_checkpoint_object"]
            assert realized["format"].endswith(".v2")
            assert realized["structural_summaries"]
            assert "partition" in realized and "boundary_slots" in realized
    assert summary["num_samples"] == 2
    result = evaluate_global_samples(
        tmp_path / "samples" / "samples.json",
        corpus_dir=corpus,
        reference_split="test",
        global_graph_path=graph_result["global_graph"],
    )
    assert "certification" in result
    assert result["certification"]["target_available_fraction"] == 1.0
