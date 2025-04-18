from typing import List, Dict, Any, NamedTuple, Optional, List, DefaultDict
import numpy as np
import torch
import pandas as pd
from collections import Counter



class ModelOutput(NamedTuple):
    """
    A standardized container for outputs returned by all GNN models.

    Attributes:
        logits (torch.Tensor): Raw prediction scores (before softmax) for each node,
                               shape [num_nodes, num_classes].

        analysis (Optional[Dict[str, Any]]): Optional intermediate representations
                                             (e.g., embeddings, layer outputs) captured
                                             during forward pass for analysis/visualization.
                                             Keys should describe the layer/type.
    """

    logits: torch.Tensor
    analysis: Optional[Dict[str, Any]] = None


class AtomResult:
    """
    Stores metadata and prediction result for a single atom.

    Attributes:
        atom_idx_in_mol (int): Index of the atom within its molecule.
        global_atom_idx (int): Global index of the atom in the full batch.
        mol_name (str): Name of the molecule.
        graph_idx (int): Index of the molecule graph in the batch.
        true_label (str): True atom type label (if available).
        pred_label (str): Predicted atom type label.
        confidence (float): Softmax probability of the predicted class.
        logits (torch.Tensor): Raw logits for all classes.
        analysis (Dict[str, np.ndarray]): Intermediate per-atom features (e.g., embeddings).
    """

    def __init__(self,
                 atom_idx_in_mol: int,
                 global_atom_idx: int,
                 mol_name: str,
                 graph_idx: int,
                 true_label: Optional[str],
                 pred_label: str,
                 confidence: float,
                 logits: torch.Tensor,
                 analysis: Dict[str, np.ndarray]):
        
        self.atom_idx_in_mol = atom_idx_in_mol
        self.global_atom_idx = global_atom_idx
        self.mol_name = mol_name
        self.graph_idx = graph_idx
        self.true_label = true_label
        self.pred_label = pred_label 
        self.confidence = confidence
        self.logits = logits
        self.analysis = analysis

    @classmethod
    def from_batch(cls, i, batch, output, label_encoder, mol_name, graph_idx, confidence=None):
        """
        Extracts AtomResult from a batch and ModelOutput.

        Args:
            i (int): Atom index in the batch.
            batch: PyG batch object with fields like .y, .mol_name, .atom_idx_in_mol, etc.
            output (ModelOutput): Output from model.forward().
            label_encoder (ModelEncoder): Used for decoding true/predicted labels.
            mol_name (str): Molecule identifier.
            graph_idx (int): Index of the graph in the batch.
            confidence (Tensor or None): Confidence tensor [num_atoms].

        Returns:
            AtomResult: Encoded per-atom result object.
        """
        pred_idx = output.logits.argmax(dim=1)[i].item()
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        true_label = None
        if hasattr(batch, 'y') and batch.y is not None:
            true_idx = batch.y[i].item()
            true_label = label_encoder.inverse_transform([true_idx])[0]

        return cls(
            atom_idx_in_mol=int(batch.atom_idx_in_mol[i]),
            global_atom_idx=int(batch.global_atom_idx[i]),
            mol_name=mol_name,
            graph_idx=int(graph_idx),
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence[i].item() if confidence is not None else None,
            logits=output.logits[i].detach().cpu(),
            analysis={k: v[i].detach().cpu().numpy() for k, v in (output.analysis or {}).items() if v.ndim == 2},
        )

class PredictionRecord:
    """
    Stores all AtomResult objects and supports filtering, summarizing, and exporting.

    Attributes:
        atom_records (List[AtomResult]): List of atom-level predictions.
        by_mol_name (DefaultDict[str, List[AtomResult]]): Molecule-wise groupings.
    """

    def __init__(self, atom_predictions: Optional[List[AtomResult]] = None):
        self.atom_records: List[AtomResult] = atom_predictions or []
        self.by_mol_name: DefaultDict[str, List[AtomResult]] = DefaultDict(list)
        for atom in self.atom_records:
            self.by_mol_name[atom.mol_name].append(atom)

    def add_atom(self, atom: AtomResult):
        """Adds an atom-level prediction to the record."""
        self.atom_records.append(atom)
        self.by_mol_name[atom.mol_name].append(atom)
    
    @property
    def true_labels(self) -> List[str]:
        """Returns all true labels (if available)."""
        return [a.true_label for a in self.atom_records]

    @property
    def pred_labels(self) -> List[str]:
        """Returns all predicted labels."""
        return [a.pred_label for a in self.atom_records]

    @property
    def matches(self) -> List[AtomResult]:
        """Returns atom predictions where prediction matches ground truth."""
        return [a for a in self.atom_records if a.true_label == a.pred_label]

    @property
    def mismatches(self) -> List[AtomResult]:
        """Returns atom predictions where prediction does NOT match ground truth."""
        return [a for a in self.atom_records if a.true_label != a.pred_label]

    @property
    def mismatched_molecules(self) -> Dict[str, List[AtomResult]]:
        """
        Returns a dictionary of molecule names → list of mismatched atoms.
        """
        mismatch_dict = defaultdict(list)
        for atom in self.mismatches:
            mismatch_dict[atom.mol_name].append(atom)
        return dict(mismatch_dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the prediction record into a pandas DataFrame for analysis.

        Returns:
            pd.DataFrame: Tabular representation with labels, confidence, and analysis.
        """
        rows = []
        for a in self.atom_records:
            row = {
                "mol_name": a.mol_name,
                "atom_idx_in_mol": a.atom_idx_in_mol,
                "global_atom_idx": a.global_atom_idx,
                "true_label": a.true_label,
                "pred_label": a.pred_label,
                "confidence": a.confidence,
            }
            for k, v in a.analysis.items():
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    row.update({f"{k}_{i}": v[i] for i in range(len(v))})
            rows.append(row)
        return pd.DataFrame(rows)


    def summary(self):
        """
        Prints an overview of model performance and analysis dimensions.
        Includes:
            - Accuracy
            - Label distribution
            - Confidence stats
            - Captured analysis keys and shapes
        """
        correct = sum(a.true_label == a.pred_label for a in self.atom_records)
        total = len(self.atom_records)
        acc = correct / total if total > 0 else 0
        print(f"Prediction Summary: {correct}/{total} correct ({acc:.2%} accuracy)")

        print("True label distribution:", Counter(self.true_labels))
        print("Pred label distribution:", Counter(self.pred_labels))

        if self.atom_records and hasattr(self.atom_records[0], 'analysis'):
            print(f"\nCaptured analysis for {total} atoms.")
            key_counts = Counter(
                k for a in self.atom_records for k in a.analysis.keys()
            )
            for key, count in key_counts.items():
                shapes = {a.analysis[key].shape for a in self.atom_records if key in a.analysis}
                shape_str = ', '.join(map(str, shapes))
                print(f" - {key}: {count} atoms | shapes: {shape_str}")

        if self.atom_records and self.atom_records[0].confidence is not None:
            confidences = [a.confidence for a in self.atom_records if a.confidence is not None]
            print(f"\nConfidence: min={min(confidences):.2f}, mean={np.mean(confidences):.2f}, max={max(confidences):.2f}")


