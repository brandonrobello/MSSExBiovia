from abc import ABC, abstractmethod

class atoMLtype_featurizer(ABC):
    """
    Abstract base class for molecular featurization.

    This class defines the structure for featurizers used in the `atoMLtype` project.
    All featurizers must implement the `featurize()` method, which converts a molecule
    into a specific feature representation.

    Methods:
        featurize(molecule): Abstract method to be implemented by subclasses for featurizing molecules.
    """

    @abstractmethod
    def featurize(self, molecule):
        """
        Abstract method to featurize a molecule.

        Args:
            molecule: The input molecule object to be featurized. The type of the molecule
                      (e.g., RDKit Mol object) depends on the specific implementation.

        Returns:
            Any: The featurized representation of the molecule. The return type depends on
                 the specific implementation (e.g., NumPy array, PyTorch tensor, etc.).
        """
        pass