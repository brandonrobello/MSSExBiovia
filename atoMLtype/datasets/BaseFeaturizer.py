from abc import ABC, abstractmethod

class BaseFeaturizer(ABC):
    """
    Abstract base class for molecular featurizers used in the atoMLtype project.

    This interface ensures that all featurizers implement a `featurize()` method
    which converts an input molecule into a structured representation usable by ML models.

    Subclasses should define the specific featurization strategy, input molecule type,
    and output data format (e.g., graph features, fingerprints, embeddings).
    """

    @abstractmethod
    def featurize(self, molecule: any) -> any:
        """
        Featurizes a molecule into a machine learning-compatible representation.

        Args:
            molecule (Any): A molecule object (e.g., RDKit `Chem.Mol`).

        Returns:
            Any: A structured representation of the molecule, such as:
                - Tuple of numpy arrays for graph-based featurizers
                - Fingerprint vector (e.g., ECFP)
                - Tensor embeddings for transformer-based models
        """
        pass