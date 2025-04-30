# engines/analysis.py
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from atoMLtype.models.ModelOutput import PredictionRecord, AtomResult
from atoMLtype.models.BaseGNNModel import BaseGNNModel

class ModelEngine:
    """
    Executes inference using a trained GNN model and collects atom-level predictions.

    Supports confidence scoring, optional embedding analysis, and returns
    structured output in a `PredictionRecord` format.

    Args:
        model (BaseGNNModel): Trained model implementing the forward method and label encoder.
        dataset (Dataset): PyTorch Geometric dataset of molecular graphs.
        device (str): Torch device string ('cpu' or 'cuda').
        batch_size (int): Batch size for batched inference.
    """
    def __init__(self, 
             model: BaseGNNModel, 
             dataset, 
             device: str = "cpu", 
             batch_size: int = 32):
        
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

    def predict(self, analysis: bool = False) -> PredictionRecord:
        """
        Runs model inference and returns structured atom-level predictions.

        Args:
            analysis (bool): If True, enables model's analysis mode (e.g., saves embeddings).

        Returns:
            PredictionRecord: A container of AtomResult objects with per-atom predictions.
        """
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        prediction_record = PredictionRecord()

        # Enable or disable analysis hooks
        if analysis:
            self.model.enable_analysis()
        else:
            self.model.disable_analysis()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                logits = output.logits
                probs = torch.softmax(logits, dim=1)
                confidence_scores, _ = probs.max(dim=1)

                # Metadata
                graph_indices = batch.batch.cpu().numpy()       # [atom_idx] → graph_idx
                mol_names = batch.mol_name                      # List of molecule names per graph


                for i in range(len(batch.x)):
                    atom = AtomResult.from_batch(
                        i=i,
                        batch=batch,
                        output=output,
                        label_encoder=self.model.encoder,
                        mol_name=mol_names[graph_indices[i]],
                        graph_idx=graph_indices[i],
                        confidence=confidence_scores
                    )
                    prediction_record.add_atom(atom)

        if analysis:
            self.model.disable_analysis()
            
        prediction_record.compute_statistics()


        return prediction_record