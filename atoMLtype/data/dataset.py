import logging
from abc import ABC, abstractmethod
from typing import Dict, List
from rdkit import Chem

class BaseDataset(ABC):
    """
    Abstract base class for molecular datasets.
    Ensures all dataset types implement essential methods.

    Attributes:
        data_source (str): Path to the primary data file (e.g., SDF, CSV).
        label_source (str): Path to the labels file, if applicable.
        molecules (Dict[str, Chem.Mol]): Dictionary of molecule names and RDKit molecule objects.
        Y_labels (Dict[str, List[str]]): Dictionary of molecule names and atom type labels.
        X_molecules (Dict[str, Chem.Mol]): Filtered molecules with valid labels.
    """

    def __init__(self, data_source: str, label_source: str = None):
        """
        Initializes the dataset by loading molecules and labels.

        Args:
            data_source (str): Path to the primary data file (e.g., SDF, CSV).
            label_source (str, optional): Path to the labels file, if applicable.
        """
        self.data_source = data_source
        self.label_source = label_source
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        self.molecules = self._load_molecules()
        self.Y_labels = self._load_labels() if label_source else None
        self.X_molecules = self._filter_molecules()

        self.log.info(f"Dataset initialized with {len(self.X_molecules)} molecules.")

    @abstractmethod
    def _load_molecules(self) -> Dict[str, Chem.Mol]:
        """
        Abstract method to load molecules from the dataset.

        Returns:
            Dict[str, Chem.Mol]: Dictionary of molecule names and RDKit molecule objects.
        """
        pass

    @abstractmethod
    def _load_labels(self) -> Dict[str, List[str]]:
        """
        Abstract method to load labels from the dataset.

        Returns:
            Dict[str, List[str]]: Dictionary of molecule names and atom type labels.
        """
        pass

    @abstractmethod
    def _filter_molecules(self) -> Dict[str, Chem.Mol]:
        """
        Abstract method to filter molecules based on loaded labels.

        Returns:
            Dict[str, Chem.Mol]: Filtered dictionary of molecule names and RDKit molecule objects.
        """
        pass