from abc import ABC, abstractmethod

class atoMLtype_featurizer(ABC):
    """
    Abstract base class for molecular featurization.
    Ensures that all featurizers implement a `featurize()` method.
    """
    @abstractmethod
    def featurize(self, molecule):
        pass