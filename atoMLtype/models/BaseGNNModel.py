import torch
import os
import torch.nn as nn
from abc import ABC, abstractmethod
from atoMLtype.models.ModelOutput import ModelOutput
from atoMLtype.models.ModelEncoder import ModelEncoder

class BaseGNNModel(nn.Module, ABC):
    """
    Abstract base class for GNN models used in atom-type prediction.

    Includes support for:
        - Encoder attachment and validation
        - Analysis mode toggling (for extracting intermediate embeddings)
        - Training status tracking

    Attributes:
        encoder (ModelEncoder): Fitted encoder for label transformation.
    """

    def __init__(self, encoder: ModelEncoder):
        super().__init__()
        self._analysis = False
        self._trained = False
        self.encoder = encoder

        # Ensure a fitted encoder is provided
        if self.encoder is None or not self.encoder.is_fitted:
            raise RuntimeError("Model requires a fitted encoder before training or inference.")

    def enable_analysis(self):
        """Enable analysis mode (e.g., to return final embeddings)."""
        self._analysis = True

    def disable_analysis(self):
        """Disable analysis mode."""
        self._analysis = False

    def is_analysis_enabled(self) -> bool:
        """Check if analysis mode is enabled."""
        return self._analysis
    
    def set_trained(self):
        """Mark the model as trained (for inference tracking)."""
        self._trained = True

    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._trained

    @property
    def num_classes(self) -> int:
        """Return number of output atom types from the encoder."""
        if self.encoder is None:
            raise RuntimeError("No label encoder attached to model.")
        return self.encoder.num_classes

    @abstractmethod
    def forward(self, data) -> ModelOutput:
        """
        Forward pass to be implemented by subclasses.

        Args:
            data (Data): A PyTorch Geometric graph Data object.

        Returns:
            ModelOutput: Container with:
                - logits: torch.Tensor [num_nodes, num_classes]
                - analysis (optional): intermediate outputs for visualization/debugging
        """
        pass

    @abstractmethod
    def _get_metadata(self) -> dict:
        """
        Returns architecture metadata needed to reconstruct the model.

        Must be implemented by subclasses.

        Returns:
            dict: Dictionary of model arguments.
        """
        pass

    def save(self, filepath: str):
        """
        Save model state_dict, encoder, and architecture metadata to file.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'encoder': self.encoder,
            'metadata': self._get_metadata()
        }
        torch.save(checkpoint, filepath)
        print(f"Model + encoder + metadata saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Load model, encoder, and architecture metadata from file.

        Args:
            filepath (str): Path to the saved checkpoint.

        Returns:
            model (cls): Reconstructed model with loaded weights and encoder.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} not found.")

        checkpoint = torch.load(filepath, weights_only=False)
        encoder = checkpoint['encoder']
        metadata = checkpoint['metadata']

        # Create the model using metadata
        model = cls(encoder=encoder, **metadata)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model + encoder loaded from {filepath} with metadata: {metadata}")

        return model