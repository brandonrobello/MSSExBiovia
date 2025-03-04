import json
import logging
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from rdkit import Chem

class BaseDataset(ABC):
    """
    Abstract base class for molecular datasets.
    Ensures all dataset types implement essential methods.
    """

    def __init__(self, data_source: str, label_source: str = None):
        """
        Args:
            data_source (str): Path to the primary data file (SDF, CSV, etc.).
            label_source (str, optional): Path to the labels file, if applicable.
        """
        self.data_source = data_source
        self.label_source = label_source
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.molecules = self._load_molecules()
        self.Y_labels = self._load_labels() if label_source else None
        self.X_molecules = self._filter_molecules()

        self.log.info(f"Dataset initialized with {len(self.filtered_molecules)} molecules.")

    @abstractmethod
    def _load_molecules(self) -> Dict[str, Chem.Mol]:
        """Must be implemented in subclasses to load molecules from the dataset."""
        pass

    @abstractmethod
    def _load_labels(self) -> Dict[str, List[str]]:
        """Must be implemented in subclasses to load labels from the dataset."""
        pass
    
    @abstractmethod
    def _filter_molecules(self) -> Dict[str, List[str]]:
        """Must be implemented in subclasses to filter X_molecules based on load Y_labels from the dataset."""
        pass

class SDFdataset(BaseDataset):
    """
    Class to organize SDF files and atom type labels for featurization.
    
    - Loads SDF files
    - Extracts molecular structures
    - Processes atom ordering and atom type labels for ML pipelines

    Attributes:
        data_source (str): Path to the SDF file.
        json_path (str): Path to the JSON file containing atom labels.
        molecules (List[Chem.Mol]): List of RDKit molecule objects.
        molNames (List[str]): List of molecule names.
        atoms (List[List[int]]): Atom indices, continuous across molecules.
        atom_labels (List[List[str]]): List of atom type labels for each molecule.
    """
    def __init__(self, data_source: str, label_source: str = None):
        self.data_source = data_source
        self.label_source = label_source
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        # SDF Processing and Ordering
        self.molecules = self._load_molecules()
        self.molNames, self.atoms = self._parse_atoms()

        # AT Label Processing and Ordering
        if label_source:
            self.label_dict = self._load_labels(label_source)
            self.Y_labels = self._map_labels()
            self.X_molecules = self._filter_molecules()
        else:
            self.label_dict = None
            self.Y_labels = None
            self.X_molecules = None

        # Check if molecule names match the number of valid molecules
        assert len(self.molNames) == len(self.molecules), (
            f"Mismatch: {len(self.molNames)} names, but {len(self.molecules)} molecules."
        )

        self.log.info(f"Successfully parsed {len(self.molecules)} molecules with valid names.")

    
    def _load_molecules(self, sanitize: bool = False) -> List[Chem.Mol]:
        """Loads molecules from an SDF file, allowing invalid valences to be ignored."""
        suppl = Chem.SDMolSupplier(self.data_source, sanitize=sanitize, removeHs=False)
        molecules = []
        invalid_count = 0

        for i, mol in enumerate(suppl):
            if mol is None:
                self.log.warning(f"Skipping molecule at index {i}: RDKit could not read it.")
                invalid_count += 1
                continue  # Skip invalid molecules

            # Try manual sanitization, but allow errors
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            except ValueError as e:
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
                self.log.warning(f"Sanitization failed for molecule at index {i}, Name: {mol_name}. Skipping sanitization: {e}")

            molecules.append(mol)  # Keep even unsanitized molecules

        self.log.info(f"Loaded {len(molecules)} molecules. Skipped {invalid_count} due to critical errors.")
        return molecules
    
    def _parse_atoms(self) -> Tuple[List[str], List[List[int]]]:
        """Parses molecule names and atom ordering."""
        return self._load_sdfName_sdfAtOrd(self.molecules)

    def get_molecule(self, index: int) -> Chem.Mol:
        """Returns a specific molecule by index."""
        return self.molecules[index]

    def get_labels(self) -> np.ndarray:
        """
        Returns a 1D array of labels from Y_labels (which is Dict[str, List[str]]).
        
        Ensures that labels are ordered to match the order of atoms in X_molecules.
        
        Returns:
            np.ndarray: Flattened array of atom labels for all molecules.
        """
        if not self.Y_labels:
            raise ValueError("Y_labels is empty or not loaded. Ensure labels are provided.")

        labels = []
        for mol_name in self.X_molecules.keys():  # Ensure we only get labels for included molecules
            if mol_name not in self.Y_labels:
                raise ValueError(f"Missing labels for molecule: {mol_name}")

            labels.extend(self.Y_labels[mol_name])  # Flattening the list

        return np.array(labels)  # Return as a NumPy array

    
    def _load_labels(self, json_path: str) -> Dict[str, List[str]]:
        """
        Loads atom type labels from a JSON file.
        JSON format:
            [
                {"Name": "Molecule_1", "Atom_types": ["C_sp3", "O_sp2", "N_sp3"]},
                {"Name": "Molecule_2", "Atom_types": ["H", "C_sp2", "Cl"]}
            ]
        Returns:
            Dict[str, List[str]]: {mol_name: [atom_type_1, atom_type_2, ...]}
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        label_dict = {entry["Name"]: entry["Atom_types"] for entry in data}

        if not self.label_source:
            self.label_source = json_path
            self.label_dict = label_dict
            self.Y_labels = self._map_labels()
            self.X_molecules = self._filter_molecules()


        self.log.info(f"Loaded atom type labels for {len(label_dict)} molecules from {json_path}")

        return label_dict
    
    def _map_labels(self) -> List[List[str]]:
        """Assigns labels to each atom based on the JSON dictionary, logging missing molecules."""
        labels = {}
        total_skipped = 0  # Track total atoms skipped
        skipped_molecules = {}   # Dict of molecules that were missing from the JSON, and why

        for mol, mol_name in zip(self.molecules, self.molNames):
            # Ensure molecule names in the SDF match the JSON labels
            assert mol.GetProp("_Name") == mol_name, f"Mismatch in molecule names: SDF={mol.GetProp('_Name')}, JSON={mol_name}"

            # Check if the molecule is missing in the JSON file
            if mol_name not in self.label_dict:
                skipped_molecules[mol_name] = "Missing from JSON"
                self.log.warning(f"Skipping molecule {mol_name}: Missing from JSON.")
                continue  # Skip this molecule

            mol_labels = self.label_dict[mol_name]
            num_atoms = len(mol.GetAtoms())

            # Ensure that the number of labels matches the number of atoms in the molecule
            # Handle atom count mismatches
            if len(mol_labels) != num_atoms:
                self.log.warning(
                    f"Atom count mismatch for {mol_name}: "
                    f"expected {num_atoms}, but got {len(mol_labels)} from JSON. Skipping."
                )
                skipped_molecules[mol_name] = f"Mismatch atom count - expected {num_atoms} != {len(mol_labels)} labels in JSON"
                total_skipped += 1
                continue

            labels[mol_name] = mol_labels

        # Log a summary of missing molecules and shifted atom count
        if skipped_molecules:
            sdf_name = os.path.splitext(os.path.basename(self.data_source))[0]  # Extract base name
            skipped_file = f"skipped_molecules_{sdf_name}.json"

            # Check if file exists and append _1, _2, etc.
            file_index = 1
            while os.path.exists(skipped_file):
                skipped_file = f"skipped_molecules_{sdf_name}_{file_index}.json"
                file_index += 1
            with open(skipped_file, "w") as f:
                json.dump(skipped_molecules, f, indent=4)
            self.log.warning(f"Skipped molecule report saved to {skipped_file}")
            self.log.warning(
                f"Skipped {len(skipped_molecules)} molecules due to missing labels."
            )
            for mol_name in skipped_molecules:
                self.log.warning(f" - {mol_name}")

        return labels
    
    def _filter_molecules(self) -> Dict[str, Chem.Mol]:
        """
        Filters molecules based on loaded Y labels so that there are only (X, Y) pairs.

        Returns:
            - filtered_molecules: Dict[str, Chem.Mol] - Only molecules with labels.
        """
        filtered_molecules = {}

        for mol in self.molecules:
            # Get molecule name
            mol_name = mol.GetProp("_Name")

            if mol_name not in self.Y_labels:
                continue  # Skip molecules without labels

            # Ensure the number of atoms matches the number of labels
            num_atoms = len(mol.GetAtoms())
            num_labels = len(self.Y_labels[mol_name])

            assert num_atoms == num_labels, (
                f"Mismatch for {mol_name}: {num_atoms} atoms in SDF, {num_labels} labels in JSON."
            )

            filtered_molecules[mol_name] = mol

        # Compare Sets of filtered_molecules and Y_labels
        missing_from_Y = set(filtered_molecules.keys()) - set(self.Y_labels.keys())
        missing_from_filtered = set(self.Y_labels.keys()) - set(filtered_molecules.keys())

        if missing_from_Y:
            self.log.warning(f"Warning: {len(missing_from_Y)} molecules in filtered dataset are missing labels.")
            for mol_name in missing_from_Y:
                self.log.warning(f" - {mol_name}")

        if missing_from_filtered:
            self.log.warning(f"Warning: {len(missing_from_filtered)} molecules in Y_Atomtypes are missing SDF entries.")
            for mol_name in missing_from_filtered:
                self.log.warning(f" - {mol_name}")
            self.log.info(f"Filtered {len(filtered_molecules)} valid molecules for training.")
            return filtered_molecules
        
        assert len(filtered_molecules) == len(self.Y_labels), (
            f"Mismatch: {len(filtered_molecules)} valid molecules, but {len(self.Y_labels)} labels."
            )
        
        self.log.info(f"Filtered {len(filtered_molecules)} valid molecules for training.")
        return filtered_molecules



    @staticmethod
    def _load_sdfName_sdfAtOrd(molecules: List[Chem.Mol]) -> Tuple[List[str], List[List[int]]]:
        """
        Extracts:
        - A list of molecule names.
        - A list of atom order indices (continuous across molecules).
        
        Returns:
            Tuple[List[str], List[List[int]]]: Names and atom orders.
        """
        sdfName_data = []
        sdfAtOrd_data = []
        atom_order_stIdx = 0  # Start index for atom numbering across molecules

        for mol in molecules:
            if mol is None:
                continue  # Skip invalid molecules

            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"

            # Get total number of atoms including hydrogens
            num_atoms = len(mol.GetAtoms())

            # Assign atom order (continuous across molecules)
            atom_order = list(range(atom_order_stIdx, atom_order_stIdx + num_atoms))
            
            # Update the starting index for the next molecule
            atom_order_stIdx += num_atoms

            # Append to lists
            sdfName_data.append(mol_name)
            sdfAtOrd_data.append(atom_order)

        return sdfName_data, sdfAtOrd_data