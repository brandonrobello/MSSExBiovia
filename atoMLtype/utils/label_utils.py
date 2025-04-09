from rdkit import Chem
from collections import defaultdict

# Atom type groups to unify ambiguity
atom_collapse_label_map = {
    'cp': 'c_pq', 'cq': 'c_pq',
    'cc': 'c_cd', 'cd': 'c_cd',
    'ce': 'c_ef', 'cf': 'c_ef',
    'ch': 'c_hg', 'cg': 'c_hg',
    'nc': 'n_cd', 'nd': 'n_cd',
    'nf': 'n_fe', 'ne': 'n_fe',
    'pc': 'p_cd', 'pd': 'p_cd',
    'pe': 'p_ef', 'pf': 'p_ef'
}

def collapse_atom_types(labels):
    """
    Collapse specific atom type labels (e.g., alternating bond pairs)
    into a unified label while preserving all others.
    
    Args:
        labels (List[str]): Original GAFF2 atom types

    Returns:
        List[str]: Collapsed labels, same length as input
    """
    return [atom_collapse_label_map.get(label, label) for label in labels]

# Define reverse mapping
# TO BE FINISHED AND TESTED
reverse_atom_map = defaultdict(list)
for k, v in atom_collapse_label_map.items():
    reverse_atom_map[v].append(k)

def assign_alternating_labels(mol, collapsed_labels):
    """
    Expand collapsed labels back to their alternating pairs (e.g., cp/cq) 
    based on ring/chain connectivity and alternating pattern.

    Args:
        mol (RDKit Mol): Molecule object
        collapsed_labels (List[str]): Predicted labels (collapsed)

    Returns:
        List[str]: Alternating predicted labels
    """
    n_atoms = mol.GetNumAtoms()
    new_labels = [None] * n_atoms

    # Group atoms by their collapsed label
    label_to_atoms = defaultdict(list)
    for idx, label in enumerate(collapsed_labels):
        if label in reverse_atom_map:
            label_to_atoms[label].append(idx)
        else:
            new_labels[idx] = label  # No change needed for non-collapsed labels

    for label, indices in label_to_atoms.items():
        pair = reverse_atom_map[label]
        if len(pair) != 2:
            raise ValueError(f"Expected 2 labels in alternation for {label}, got {pair}")

        # Build subgraph of just those atoms
        submol = Chem.RWMol(mol)
        atoms_to_remove = [i for i in range(n_atoms) if i not in indices]
        for idx in sorted(atoms_to_remove, reverse=True):
            submol.RemoveAtom(idx)

        # Assign alternating labels based on connectivity
        visited = set()
        stack = [indices[0]]  # Start with the first atom in the group
        current_label = pair[0]

        while stack:
            atom_idx = stack.pop()
            if atom_idx in visited:
                continue
            visited.add(atom_idx)
            new_labels[atom_idx] = current_label

            # Alternate the label for the next neighbors
            current_label = pair[1] if current_label == pair[0] else pair[0]

            # Add neighbors to the stack
            atom = submol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in indices and neighbor_idx not in visited:
                    stack.append(neighbor_idx)

    return new_labels
