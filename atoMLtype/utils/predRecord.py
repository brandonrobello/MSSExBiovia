from typing import List, Dict, Any
import numpy as np

class AtomPrediction:
    def __init__(self, atom_idx_in_mol: int, global_atom_idx: int, mol_name: str,
                 true_label: str, pred_label: str, x_embedding: np.ndarray,
                 clf_embeddings: np.ndarray):
        self.atom_idx_in_mol = atom_idx_in_mol
        self.global_atom_idx = global_atom_idx
        self.mol_name = mol_name
        self.true_label = true_label
        self.pred_label = pred_label
        self.x_embedding = x_embedding
        self.clf_embeddings = clf_embeddings

class PredRecord:
    def __init__(self):
        self.atom_records: List[AtomPrediction] = []
        self.by_mol_name: Dict[int, List[AtomPrediction]] = {}  # mol_name_idx → atoms
        self.molecule_attn: Dict[str, List[Dict]] = {}  # mol_name_idx → atoms

    def add_atom(self, atom: AtomPrediction):
        self.atom_records.append(atom)
        if atom.mol_name not in self.by_mol_name:
            self.by_mol_name[atom.atom_idx_in_mol] = []
        self.by_mol_name[atom.atom_idx_in_mol].append(atom)

    def add_molecule_attention(self, mol_name, attention_maps):
        self.molecule_attn[mol_name] = attention_maps

    def get_x_embedding(self) -> np.ndarray:
        return np.stack([a.x_embedding for a in self.atom_records])
    
    def get_clf_embedding(self) -> np.ndarray:
        return np.stack([a.clf_embeddings for a in self.atom_records])

    def get_labels(self, label_type='true') -> List[int]:
        return [getattr(a, f"{label_type}_label") for a in self.atom_records]
    
    def get_matches(self) -> List[AtomPrediction]:
        return [a for a in self.atom_records if a.true_label == a.pred_label]

    def get_mismatches(self) -> List[AtomPrediction]:
        return [a for a in self.atom_records if a.true_label != a.pred_label]
    
    def get_mismatched_molecules(self) -> Dict[str, List[AtomPrediction]]:
        """
        Returns a dict of mol_name : list of AtomPrediction for molecules with any mismatch.
        """
        mismatched_molecules = {}
        for atom in self.atom_records:
            if atom.true_label != atom.pred_label:
                if atom.mol_name not in mismatched_molecules:
                    mismatched_molecules[atom.mol_name] = []
                mismatched_molecules[atom.mol_name].append(atom)
        return mismatched_molecules

    def get_graph_atoms(self, atom_idx_in_mol: int) -> List[AtomPrediction]:
        return self.by_mol_name.get(atom_idx_in_mol, [])

    def to_dataframe(self):
        """Export all atom records to a pandas DataFrame for analysis."""
        import pandas as pd
        records = [{
            "atom_idx_in_mol": a.atom_idx_in_mol,
            "global_atom_idx": a.global_atom_idx,
            "true_label": a.true_label,
            "pred_label": a.pred_label,
            **a.extra  # any additional info
        } for a in self.atom_records]
        return pd.DataFrame(records)

    def summary(self):
        from collections import Counter
        correct = sum(a.true_label == a.pred_label for a in self.atom_records)
        total = len(self.atom_records)
        acc = correct / total if total > 0 else 0
        print(f"Prediction Summary: {correct}/{total} correct ({acc:.2%} accuracy)")
        print("True label distribution:", Counter(self.get_labels('true')))
        print("Pred label distribution:", Counter(self.get_labels('pred')))
