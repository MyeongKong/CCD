from __future__ import annotations

from typing import Optional

DEFAULT_ATOM_VOCAB = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "other")
DEFAULT_BOND_VOCAB = ("none", "single", "double", "triple", "aromatic")


def _rdkit_modules():
    try:
        from rdkit import Chem  # type: ignore

        return Chem
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "RDKit is required for SMILES/SDF preprocessing. Install the optional "
            "chem dependencies, for example `pip install rdkit`, or use graph JSON input."
        ) from exc


def rdkit_available() -> bool:
    try:
        _rdkit_modules()
        return True
    except ImportError:
        return False


def bond_type_to_label(bond) -> str:
    Chem = _rdkit_modules()
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE:
        return "single"
    if btype == Chem.BondType.DOUBLE:
        return "double"
    if btype == Chem.BondType.TRIPLE:
        return "triple"
    if btype == Chem.BondType.AROMATIC:
        return "aromatic"
    return "single"


def mol_from_smiles(smiles: str):
    Chem = _rdkit_modules()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def smiles_to_molecule_graph(smiles: str, source_id: Optional[str] = None):
    from .topology import MoleculeGraph

    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    heavy_to_local: dict[int, int] = {}
    atom_symbols: list[str] = []
    charges: list[int] = []
    original_atom_ids: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        local_idx = len(atom_symbols)
        heavy_to_local[atom.GetIdx()] = local_idx
        atom_symbols.append(atom.GetSymbol())
        charges.append(int(atom.GetFormalCharge()))
        original_atom_ids.append(int(atom.GetIdx()))

    edges: dict[tuple[int, int], str] = {}
    ring_edges: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a not in heavy_to_local or b not in heavy_to_local:
            continue
        u = heavy_to_local[a]
        v = heavy_to_local[b]
        key = (u, v) if u < v else (v, u)
        edges[key] = bond_type_to_label(bond)
        if bond.IsInRing():
            ring_edges.add(key)

    ring_atoms = {
        heavy_to_local[idx]
        for idx in range(mol.GetNumAtoms())
        if idx in heavy_to_local and mol.GetAtomWithIdx(idx).IsInRing()
    }

    return MoleculeGraph(
        atom_symbols=tuple(atom_symbols),
        charges=tuple(charges),
        edges=edges,
        ring_atoms=frozenset(ring_atoms),
        ring_edges=frozenset(ring_edges),
        original_atom_ids=tuple(original_atom_ids),
        source_id=source_id or smiles,
    )


def checkpoint_state_to_rdkit_mol(state, *, sanitize: bool = True):
    """Convert an atom-level decoded checkpoint to an RDKit molecule.

    Returns ``None`` when the state is unresolved or cannot be sanitized.  The
    import stays local so graph-only preprocessing and tests do not require the
    optional RDKit dependency.
    """

    Chem = _rdkit_modules()
    rw = Chem.RWMol()
    for summary in state.node_summaries:
        if int(summary.atom_count) != 1 or sum(int(v) for v in summary.hist) != 1:
            return None
        active = [idx for idx, value in enumerate(summary.hist) if int(value) == 1]
        if len(active) != 1:
            return None
        symbol = str(state.atom_vocab[active[0]])
        if symbol == "other":
            return None
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(summary.charge))
        rw.AddAtom(atom)

    bond_types = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }
    aromatic_atoms: set[int] = set()
    try:
        for (left, right), label in state.edges.items():
            if label == "none":
                continue
            if label not in bond_types:
                return None
            rw.AddBond(int(left), int(right), bond_types[label])
            if label == "aromatic":
                aromatic_atoms.update((int(left), int(right)))
        mol = rw.GetMol()
        for idx in aromatic_atoms:
            mol.GetAtomWithIdx(idx).SetIsAromatic(True)
        if sanitize:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def state_to_canonical_smiles(state) -> str | None:
    if not rdkit_available():
        return None
    Chem = _rdkit_modules()
    mol = checkpoint_state_to_rdkit_mol(state, sanitize=True)
    if mol is None:
        return None
    try:
        return str(Chem.MolToSmiles(mol, canonical=True))
    except Exception:
        return None
