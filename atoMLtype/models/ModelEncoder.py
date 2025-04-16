from collections import defaultdict
from typing import List, Dict
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder

class ModelEncoder:
    """
    Handles atom-type label encoding and optional label collapsing for model training and inference.

    Supports:
        - Label collapsing (e.g., mapping 'cp', 'cq' → 'c_pq') for symmetry handling
        - Inverse transformation and label recovery after prediction
        - Alternating label assignment using molecular topology after inference

    Attributes:
        collapse (bool): Whether to collapse GAFF2 atom types into unified labels.
        is_fitted (bool): Tracks if the LabelEncoder has been fit.
        label_map (Dict[str, str]): Mapping of specific GAFF2 atom types to collapsed classes.
        reverse_map (Dict[str, List[str]]): Inverse mapping from collapsed class to original atom types.
    """
    def __init__(self, collapse: bool = True, label_map: dict = {}):
        self.label_encoder = LabelEncoder()
        self.collapse = collapse
        self.is_fitted = False

        # Default label collapsing for GAFF2 aromatic/conjugated atom pairs
        zinc_label_map = {
            'cp': 'c_pq', 'cq': 'c_pq',
            'cc': 'c_cd', 'cd': 'c_cd',
            'ce': 'c_ef', 'cf': 'c_ef',
            'ch': 'c_hg', 'cg': 'c_hg',
            'nc': 'n_cd', 'nd': 'n_cd',
            'nf': 'n_fe', 'ne': 'n_fe',
            'pc': 'p_cd', 'pd': 'p_cd',
            'pe': 'p_ef', 'pf': 'p_ef'
        }

        # Default Mapping
        self.label_map = zinc_label_map if not label_map else label_map
        self.reverse_map = self._build_reverse_map(self.label_map)      

    def _build_reverse_map(self, label_map):
        """
        Builds a reverse map from collapsed labels to their original atom types.

        Returns:
            Dict[str, List[str]]: Mapping from collapsed label to list of original labels.
        """
        reverse_map = defaultdict(list)
        for k, v in label_map.items():
            reverse_map[v].append(k)
        return reverse_map

    def fit(self, labels: List[str]):
        """
        Fits the internal LabelEncoder using (optionally collapsed) atom-type labels.

        Args:
            labels (List[str]): Atom-type labels from training data.
        """
        processed = self._collapse(labels) if self.collapse else labels
        self.label_encoder.fit(processed)
        self.is_fitted = True

    def transform(self, labels: List[str]) -> List[int]:
        """
        Transforms labels into integer-encoded indices.

        Args:
            labels (List[str]): Atom-type strings.

        Returns:
            List[int]: Encoded integer labels.
        """
        processed = self._collapse(labels) if self.collapse else labels
        return self.label_encoder.transform(processed)

    def inverse_transform(self, indices: List[int]) -> List[str]:
        """
        Converts integer-encoded indices back into (collapsed) atom-type labels.

        Args:
            indices (List[int]): Encoded integer labels.

        Returns:
            List[str]: Decoded atom-type labels.
        """
        return self.label_encoder.inverse_transform(indices)

    def _collapse(self, labels: List[str]) -> List[str]:
        """
        Collapses atom types into merged categories based on label_map.

        Args:
            labels (List[str]): Original GAFF2 labels.

        Returns:
            List[str]: Collapsed labels.
        """
        return [self.label_map.get(label, label) for label in labels]
    
    ####******** NEEDS TO BE VALIDATED *******#######
    def uncollapse(self, mol: Chem.Mol, collapsed_preds: List[str]) -> List[str]:
        """
        Recovers original alternating atom labels from collapsed predictions.

        Uses molecular topology (connectivity) to alternate between the possible
        original labels for atoms within the same collapsed group.

        Args:
            mol (Chem.Mol): RDKit molecule.
            collapsed_preds (List[str]): Predicted collapsed labels.

        Returns:
            List[str]: Uncollapsed atom-type predictions.
        """
        n_atoms = mol.GetNumAtoms()
        new_labels = [None] * n_atoms  # Will hold the final uncollapsed labels

        label_to_atoms = defaultdict(list)  # Map collapsed label → atom indices

        # Group atom indices by collapsed label
        for idx, label in enumerate(collapsed_preds): 
            if label in self.reverse_map:
                label_to_atoms[label].append(idx)
            else:
                new_labels[idx] = label  # Non-collapsed labels are assigned directly

        # For each group of collapsed labels that maps to a pair (e.g., 'c_pq' -> ['cp', 'cq'])
        for label, indices in label_to_atoms.items():
            pair = self.reverse_map[label]
            if len(pair) != 2:
                raise ValueError(f"Expected 2 labels in alternation for {label}, got {pair}")

            # Create submol by removing all atoms not in `indices`
            submol = Chem.RWMol(mol)
            atoms_to_remove = [i for i in range(n_atoms) if i not in indices]
            for idx in sorted(atoms_to_remove, reverse=True):
                submol.RemoveAtom(idx)

            # Traverse submol to assign alternating labels based on connectivity
            visited = set()
            stack = [indices[0]]  # Start from the first atom in the group
            current_label = pair[0]

            while stack:
                atom_idx = stack.pop()
                if atom_idx in visited:
                    continue
                visited.add(atom_idx)
                new_labels[atom_idx] = current_label

                # Alternate label for the next connected atoms
                current_label = pair[1] if current_label == pair[0] else pair[0]

                atom = submol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in indices and neighbor_idx not in visited:
                        stack.append(neighbor_idx)

        return new_labels

    @property
    def num_classes(self) -> int:
        """
        Returns the number of unique labels learned during fitting.

        Returns:
            int: Number of classes.
        """
        return len(self.label_encoder.classes_) if self.is_fitted else 0

    @property
    def classes(self) -> List[str]:
        """
        Returns the list of label classes.

        Returns:
            List[str]: Class label names.
        """
        return list(self.label_encoder.classes_) if self.is_fitted else []
    
    def __repr__(self):
        """
        String representation of the encoder status.

        Returns:
            str: Status string showing fitted/collapsed state.
        """
        collapse_status = "with collapse" if self.collapse else "without collapse"
        status = "fitted" if self.is_fitted else "not fitted"
        return f"<ModelEncoder ({status}, {collapse_status}, {self.num_classes} classes)>"