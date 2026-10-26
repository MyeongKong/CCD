from __future__ import annotations

from checkpoint_dag_diffusion.checkpoint_state import CheckpointState, NodeSummary
from checkpoint_dag_diffusion import global_eval


def _chain_state(num_nodes: int) -> CheckpointState:
    summaries = tuple(NodeSummary.from_atom("C") for _ in range(num_nodes))
    return CheckpointState(
        stage="full",
        node_summaries=summaries,
        edges={(idx, idx + 1): "single" for idx in range(num_nodes - 1)},
        core_flags=tuple(True for _ in range(num_nodes)),
        ring_flags=tuple(False for _ in range(num_nodes)),
        original_atom_ids=tuple((idx,) for idx in range(num_nodes)),
    )


def test_certification_denominators_distinguish_valid_only_and_all_attempts(monkeypatch):
    generated = [_chain_state(num_nodes) for num_nodes in (1, 2, 3, 4)]
    requested = {"certificate_hash": "requested"}
    data = [
        {"requested_path_objects": [requested]},
        {"requested_path_objects": [requested]},
        {"requested_path_objects": [requested]},
        {},  # A missing request cannot count as a certified generation.
    ]

    # Samples 1, 2, and 4 are chemically valid.  The third sample is invalid,
    # even though recontraction would otherwise match its requested object.
    monkeypatch.setattr(
        global_eval,
        "molecular_validity",
        lambda state, require_rdkit=False: state.num_nodes != 3,
    )

    def fake_recontract(state, objects):
        exact = state.num_nodes in {1, 3}
        comparison = {
            "exact": exact,
            "semantic": True,
            "qmatch": True,
            "summary_error": 0.0,
        }
        return {
            "recontractable": True,
            "comparisons": [comparison],
        }

    monkeypatch.setattr(global_eval, "recontract_and_compare", fake_recontract)
    metrics = global_eval._certification_metrics(data, generated, None)

    assert metrics["target_available_fraction"] == 3 / 4
    assert metrics["valid_target_count"] == 2
    # Cert@Exact/QMatch use only valid samples for which a request is present.
    assert metrics["cert_exact"] == 1 / 2
    assert metrics["cert_semantic"] == 1.0
    assert metrics["qmatch"] == 1.0
    # Certified Validity uses all four generation attempts and exact identity,
    # not the weaker quotient-graph match.
    assert metrics["certified_validity"] == 1 / 4
    assert metrics["certified_semantic_validity"] == 2 / 4
    assert metrics["path_certification"] == 2 / 4

