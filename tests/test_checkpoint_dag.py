from checkpoint_dag_diffusion.dag_topology import build_checkpoint_dag, checkpoint_contractible_to
from checkpoint_dag_diffusion.topology import MoleculeGraph


def branched_ring() -> MoleculeGraph:
    return MoleculeGraph(
        atom_symbols=("C", "C", "C", "C", "C", "C", "C", "O"),
        charges=(0, 0, 0, 0, 0, 0, 0, 0),
        edges={
            (0, 1): "single",
            (1, 2): "single",
            (2, 3): "single",
            (3, 4): "single",
            (4, 5): "single",
            (0, 5): "single",
            (0, 6): "single",
            (6, 7): "single",
        },
        ring_atoms=frozenset({0, 1, 2, 3, 4, 5}),
        ring_edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}),
        source_id="branched_ring",
    )


def propane() -> MoleculeGraph:
    return MoleculeGraph(
        atom_symbols=("C", "C", "C"),
        charges=(0, 0, 0),
        edges={(0, 1): "single", (1, 2): "single"},
        source_id="propane",
    )


def test_radius_checkpoint_sequence_is_deterministic():
    first = build_checkpoint_dag(branched_ring())
    second = build_checkpoint_dag(branched_ring())
    assert [cp.hash for cp in first.checkpoints] == [cp.hash for cp in second.checkpoints]
    assert len(first.checkpoints) >= 3
    assert any(edge.delta > 1 for edge in first.edges)


def test_every_dag_edge_is_contractible():
    dag = build_checkpoint_dag(branched_ring())
    for edge in dag.edges:
        if edge.guide is None:
            continue
        finer = dag.checkpoints[edge.target].state
        coarser = dag.checkpoints[edge.guide].state
        assert checkpoint_contractible_to(finer, coarser)


def test_ringless_fallback_creates_valid_dag():
    dag = build_checkpoint_dag(propane())
    assert len(dag.checkpoints) >= 2
    assert len(dag.edges) >= 2
    assert dag.checkpoints[-1].state.core_empty


def test_canonical_hashes_can_merge_equivalent_checkpoints():
    first = build_checkpoint_dag(branched_ring())
    second = build_checkpoint_dag(branched_ring())
    shared = {cp.hash for cp in first.checkpoints} & {cp.hash for cp in second.checkpoints}
    assert len(shared) == len(first.checkpoints)


def test_ring_flags_do_not_depend_on_noncontiguous_source_atom_ids():
    graph = branched_ring()
    shifted = MoleculeGraph(
        atom_symbols=graph.atom_symbols,
        charges=graph.charges,
        edges=graph.edges,
        ring_atoms=graph.ring_atoms,
        ring_edges=graph.ring_edges,
        original_atom_ids=tuple(10 + 2 * index for index in range(graph.num_atoms)),
        source_id="shifted-source-ids",
    )
    pure_core = build_checkpoint_dag(shifted).checkpoints[-1].state
    assert pure_core.ring_flags
    assert all(pure_core.ring_flags)
    assert pure_core.attachment_slots[0].atom_anchor == 10
    assert pure_core.attachment_slots[0].boundary_atom_ids == (10,)


def test_certified_sigma_retains_internal_bond_type_histograms():
    pure_core = build_checkpoint_dag(branched_ring()).checkpoints[-1].state
    single_index = pure_core.bond_vocab.index("single")
    contracted_anchor = pure_core.node_summaries[0]
    # The off-core C--O component contains one bond and its connection to the
    # ring becomes a second hidden bond after pure-core contraction.
    assert contracted_anchor.internal_bond_hist[single_index] == 2
    assert sum(contracted_anchor.internal_bond_hist) == contracted_anchor.hidden_bonds == 2
    certified = build_checkpoint_dag(branched_ring()).checkpoints[-1].to_dict()["certified_object"]
    assert certified["structural_summaries"][0]["internal_bond_hist"][single_index] == 2
