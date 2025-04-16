from abc import ABC, abstractmethod
from atoMLtype.utils.logging_utils import get_logger
from typing import Dict, List, Tuple
from rdkit import Chem

class BaseDataset(ABC):
    """
    Abstract base class for molecular datasets (SDF etc).
    Defines a consistent interface for dataset loading and label parsing.

    Attributes:
        data_path (str): Path to the primary molecular file (e.g., SDF).
        label_source (str): Optional path to label source (e.g., JSON file).
        log (Logger): Logger instance for tracking dataset operations.
    """

    def __init__(self, data_path: str, label_source: str = None):
        self.data_path = data_path
        self.label_source = label_source
        self.log = get_logger(__name__)

    @abstractmethod
    def _load_molecules(self) -> Dict[str, Chem.Mol]:
        """
        Abstract method to load molecules into memory.
        Must be implemented in the subclass.
        """
        pass

    @abstractmethod
    def _load_labels(self) -> Dict[str, List[str]]:
        """
        Abstract method to load atom type labels.
        Must be implemented in the subclass.
        """
        pass
