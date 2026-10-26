#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from checkpoint_dag_diffusion.artifacts import load_corpus_manifest
from checkpoint_dag_diffusion.global_codebook import assign_codebook, train_codebook


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned global checkpoint codebook.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_assign", action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config is not None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    representation = dict(config.get("representation", {}))
    training = dict(config.get("train", {}))
    manifest = load_corpus_manifest(args.corpus_dir)
    expected_max_nodes = int(representation.get("max_nodes", manifest["max_nodes"]))
    expected_node_dim = int(representation.get("node_feature_dim", len(manifest["atom_vocab"]) + 9))
    expected_edge_dim = int(representation.get("edge_categories", len(manifest["bond_vocab"])))
    actual_node_dim = len(manifest["atom_vocab"]) + 9
    if int(manifest["max_nodes"]) != expected_max_nodes:
        raise ValueError("Corpus max_nodes does not match the representation configuration.")
    if actual_node_dim != expected_node_dim or len(manifest["bond_vocab"]) != expected_edge_dim:
        raise ValueError("Corpus node/edge dimensions do not match the representation configuration.")

    def choose(value, configured, default):
        return default if value is None and configured is None else (configured if value is None else value)

    codebook_size = int(choose(args.codebook_size, representation.get("codebook_size"), 512))
    hidden_dim = int(
        choose(args.hidden_dim, representation.get("encoder_hidden_dim", representation.get("code_dim")), 128)
    )
    code_dim = int(representation.get("code_dim", hidden_dim))
    if code_dim != hidden_dim:
        raise ValueError("The reference VQ implementation requires encoder_hidden_dim == code_dim.")
    num_layers = int(choose(args.num_layers, representation.get("encoder_layers"), 2))
    num_heads = int(choose(args.num_heads, representation.get("encoder_heads"), 4))
    dropout = float(choose(args.dropout, representation.get("encoder_dropout"), 0.1))
    epochs = int(choose(args.epochs, representation.get("epochs"), 1))
    batch_size = int(choose(args.batch_size, training.get("batch_size"), 64))
    lr = float(choose(args.lr, training.get("lr"), 2e-4))
    beta = float(choose(args.beta, representation.get("commitment_weight"), 0.25))
    weight_decay = float(choose(args.weight_decay, training.get("weight_decay"), 1e-4))
    grad_clip = float(choose(args.grad_clip, training.get("grad_clip"), 1.0))
    seed = int(choose(args.seed, config.get("seed"), 0))
    loss_weights = dict(representation.get("loss", {}))
    result = train_codebook(
        corpus_dir=args.corpus_dir,
        out_dir=args.out_dir,
        split=args.split,
        codebook_size=codebook_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        loss_weights=loss_weights,
        device=args.device,
        seed=seed,
    )
    if not args.skip_assign:
        result["assignment"] = assign_codebook(
            corpus_dir=args.corpus_dir,
            codebook_path=result["codebook"],
            out_dir=args.out_dir,
            split=args.split,
            batch_size=batch_size,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
