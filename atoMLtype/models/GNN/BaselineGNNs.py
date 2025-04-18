import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv


from atoMLtype.models.ModelOutput import ModelOutput
from atoMLtype.models.ModelEncoder import ModelEncoder
from atoMLtype.models.BaseGNNModel import BaseGNNModel


class BaselineGCN(BaseGNNModel):
    """
    Baseline Graph Convolutional Network (GCN) model for atom type classification.

    Architecture:
        - 2-layer GCN with ReLU activations
        - Final linear layer for atom type classification
    """

    def __init__(self, num_node_features: int, encoder: ModelEncoder, hidden_dim: int = 64):
        """
        Initializes the GCN model.

        Args:
            num_node_features (int): Dimension of input node features.
            encoder (ModelEncoder): Encoder for atom type labels (must be fitted).
            hidden_dim (int): Hidden feature size for GCN layers.
        """
        super().__init__(encoder)
        self.nn1 = GCNConv(num_node_features, hidden_dim)
        self.nn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, data):
        """
        Executes a forward pass through the GCN.

        Args:
            data (Data): PyG Data object with x (node features) and edge_index.

        Returns:
            ModelOutput: Object containing classification logits and optional analysis.
        """
        x, edge_index = data.x, data.edge_index
        
        # GCN layers with ReLU
        x = F.relu(self.nn1(x, edge_index))
        x = F.relu(self.nn2(x, edge_index))    
        
        # Classify with final linear layer
        logits = self.fc(x)

        # Prepare analysis dictionary if enabled
        analysis = {} if self.is_analysis_enabled() else None

        if self.is_analysis_enabled():
            # print("[DEBUG] Analysis mode is ON in model.")
            analysis = {
                "final_embeddings": x.detach().cpu()
            }

        return ModelOutput(
            logits=logits,
            analysis=analysis,
        )

class BaselineGAT(BaseGNNModel):
    """
    Baseline Graph Attention Network (GAT) model for atom type classification.

    Architecture:
        - GATConv → GATConv → Linear
        - Uses multi-head attention in the first GAT layer
    """

    def __init__(self, num_node_features: int, encoder: ModelEncoder, hidden_dim: int = 64, heads: int = 4):
        """
        Initializes the BaselineGAT model.

        Args:
            num_node_features (int): Number of input node features.
            encoder (ModelEncoder): Label encoder for atom types.
            hidden_dim (int): Hidden layer dimensionality.
            heads (int): Number of attention heads in the first GAT layer.
        """
        super().__init__(encoder)
        self.nn1 = GATConv(num_node_features, hidden_dim, heads=heads)
        self.nn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, data) -> ModelOutput:
        """
        Forward pass through the GAT model.

        Args:
            data (Data): PyTorch Geometric Data object.

        Returns:
            ModelOutput: Contains logits and optional embedding analysis.
        """
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.nn1(x, edge_index))
        x = F.elu(self.nn2(x, edge_index))
        logits = self.fc(x)

        # For analsyis mode
        analysis = {} if self.is_analysis_enabled() else None

        if self.is_analysis_enabled():
            # print("[DEBUG] Analysis mode is ON in model.")
            analysis = {
                "final_embeddings": x.detach().cpu()
            }

        return ModelOutput(
            logits=logits,
            analysis=analysis,
        )


class BaselineGIN(BaseGNNModel):
    """
    Baseline Graph Isomorphism Network (GIN) model for atom type classification.

    Architecture:
        - GINConv(MLP) → GINConv(MLP) → Linear
    """

    def __init__(self, num_node_features: int, encoder: ModelEncoder, hidden_dim: int = 512):
        """
        Initializes the BaselineGIN model.

        Args:
            num_node_features (int): Number of input node features.
            encoder (ModelEncoder): Label encoder for atom types.
            hidden_dim (int): Hidden layer dimensionality.
        """
        super().__init__(encoder)

        self.mlp1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.nn1 = GINConv(self.mlp1)
        self.nn2 = GINConv(self.mlp2)
        self.fc = nn.Linear(hidden_dim, self.num_classes)


    def forward(self, data) -> ModelOutput:
        """
        Forward pass through the GIN model.

        Args:
            data (Data): PyTorch Geometric Data object.

        Returns:
            ModelOutput: Contains logits and optional embedding analysis.
        """
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.nn1(x, edge_index))
        x = F.elu(self.nn2(x, edge_index))
        logits = self.fc(x)


        # For analsyis mode
        analysis = {} if self.is_analysis_enabled() else None

        if self.is_analysis_enabled():
            # print("[DEBUG] Analysis mode is ON in model.")
            analysis = {
                "final_embeddings": x.detach().cpu()
            }

        return ModelOutput(
            logits=logits,
            analysis=analysis,
        )