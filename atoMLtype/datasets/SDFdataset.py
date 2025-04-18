import json
import os
from typing import Dict, List, Tuple
from rdkit import Chem
from atoMLtype.datasets.BaseDataset import BaseDataset

class SDFdataset(BaseDataset):
    """
    Dataset class for reading molecules from an SDF file and associating them with atom type labels.

    Attributes:
        label_dict (Dict[str, List[str]]): Maps molecule names to atom type labels.
        all_molecules (List[Chem.Mol]): All molecules parsed from the SDF file.
        labeled_molecules (List[Chem.Mol]): Subset of molecules that have valid labels.
        has_labels (bool): Indicates whether a label file was provided.
    """

    def __init__(self, data_path: str, label_path: str = None):
        """
        Initializes the dataset, loads molecules and labels (if available).

        Args:
            data_path (str): Path to the .sdf file containing molecule structures.
            label_path (str, optional): Path to the JSON file containing atom type labels.
        """
        super().__init__(data_path, label_path)
        self.data_path = data_path
        self.label_path = label_path

        # AT Label Processing
        self.has_labels = label_path is not None
        self.label_dict = self._load_labels(self.label_path) if self.has_labels else {}

        # SDF Processing
        self.all_molecules, self.labeled_molecules = self._load_molecules()

    def get_molecules(self, labeled: bool = True) -> List[Chem.Mol]:
        """
        Returns the list of molecules, optionally filtered by whether labels are available.

        Args:
            labeled (bool): If True, returns only labeled molecules.

        Returns:
            List[Chem.Mol]: List of RDKit molecule objects.
        """
        return self.labeled_molecules if labeled else self.all_molecules

    def _load_labels(self, json_path: str) -> Dict[str, List[str]]:
        """
        Loads atom type labels from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing labels.

        Returns:
            Dict[str, List[str]]: Mapping from molecule name to atom type label list.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        label_dict = {entry["Name"]: entry["Atom_types"] for entry in data}
        self.log.info(f"Loaded atom type labels for {len(label_dict)} molecules from {json_path}")
        return label_dict

    def _load_molecules(self, sanitize: bool = False, removeHs: bool = False) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        """
        Loads molecules from the SDF file and attaches labels if available.

        Args:
            sanitize (bool): Whether to sanitize molecules using RDKit.
            removeHs (bool): Whether to remove hydrogen atoms.

        Returns:
            Tuple[List[Chem.Mol], List[Chem.Mol]]: 
                - All parsed molecules.
                - Molecules with successfully matched atom labels.
        """
        suppl = Chem.SDMolSupplier(self.data_path, sanitize=sanitize, removeHs=removeHs)
        all_molecules = []
        labeled_molecules = []
        skipped = {}
        mismatch_count = 0

        for mol in suppl:
            if mol is None or not mol.HasProp("_Name"):
                continue # Skip invalid molecules & unnamed ones

            name = mol.GetProp("_Name")

            # Handle labeling if labels are available and match by name
            if self.has_labels and name in self.label_dict:
                labels = self.label_dict[name]

                # Check if the number of labels matches the number of atoms
                if len(labels) != mol.GetNumAtoms():
                    self.log.warning(
                        f"Atom count mismatch for {name}: "
                        f"{mol.GetNumAtoms()} atoms in SDF, but {len(labels)} labels in JSON."
                    )
                    skipped[name] = F"Atom count mismatch | {mol.GetNumAtoms()} atoms in SDF, but {len(labels)} labels in JSON."
                    mismatch_count += 1
                    continue
                
                # Store full label list as property
                mol.SetProp("atom_labels", json.dumps(labels))

                # Assign each atom its corresponding label
                for atom, label in zip(mol.GetAtoms(), labels):
                    atom.SetProp("atom_type", label)

                labeled_molecules.append(mol)
            
            elif self.has_labels:
                # Molecule not found in label dictionary
                skipped[name] = "mol_name does not have labels"

            # Retain all valid molecules regardless of labeling            
            all_molecules.append(mol)

        # Write skipped molecule report
        if skipped:
            sdf_name = os.path.splitext(os.path.basename(self.data_path))[0]
            report_path = f"nonlabeled_molecules_{sdf_name}.json"
            index = 1
            while os.path.exists(report_path):
                report_path = f"nonlabeled_molecules_{sdf_name}_{index}.json"
                index += 1
            with open(report_path, "w") as f:
                json.dump(skipped, f, indent=4)
            self.log.warning(f"Skipped {len(skipped)} molecules. Report saved to {report_path}")

        self.log.info(
            f"Loaded {len(all_molecules)} total molecules | "
            f"{len(labeled_molecules)} labeled | "
            f"{mismatch_count} skipped due to atom/label mismatch."
        )
        
        return all_molecules, labeled_molecules