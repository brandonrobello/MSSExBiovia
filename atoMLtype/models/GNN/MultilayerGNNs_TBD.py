import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, MessagePassing, GINConv
from torch_geometric.utils import add_self_loops

from atoMLtype.models.ModelOutput import ModelOutput
from atoMLtype.models.ModelEncoder import ModelEncoder
from atoMLtype.models.BaseGNNModel import BaseGNNModel


class GCN_4Layer(BaseGNNModel):
    """
    A 4-layer Graph Convolutional Network (GCN) for atom type classification.

    This model uses four GCN layers followed by a fully connected layer for classification.
    """

    def __init__(self, num_node_features, num_atom_types, hidden_dim=512):
        """
        Initializes the GCN_4Layer model.

        Args:
            num_node_features (int): Number of input node features.
            num_atom_types (int): Number of output atom type classes.
            hidden_dim (int): Hidden dimension for the GCN layers.
        """
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.conv4 = GCNConv(hidden_dim // 4, hidden_dim // 4)
        self.fc = nn.Linear(hidden_dim // 4, num_atom_types)

    def forward(self, data):
        """
        Forward pass of the GCN_4Layer.

        Args:
            data (Data): PyTorch Geometric Data object containing graph data.
            return_graph_embedding (bool): If True, returns the graph embedding.

        Returns:
            Tensor: Predicted atom types or graph embedding.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
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


class GAT_4L(BaseGNNModel):
    """
    A 4-layer Graph Attention Network (GAT) for atom type classification.

    This model uses four GAT layers with multi-head attention followed by a fully connected layer for classification.
    """

    def __init__(self, num_node_features, num_atom_types, hidden_dim=1024, num_heads=4, dropout=0.2):
        """
        Initializes the GAT_4L model.

        Args:
            num_node_features (int): Number of input node features.
            num_atom_types (int): Number of output atom type classes.
            hidden_dim (int): Hidden dimension for the GAT layers.
            num_heads (int): Number of attention heads in the GAT layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        # Define GAT layers with multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=num_heads)
        self.conv4 = GATConv(hidden_dim // 2 * num_heads, hidden_dim // 4, heads=1, concat=False)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim // 4, num_atom_types)

    def forward(self, data):
        """
        Forward pass of the GAT_4L model.

        Args:
            data (Data): PyTorch Geometric Data object containing graph data.

        Returns:
            Tensor: Predicted atom types (logits).
        """
        x, edge_index = data.x, data.edge_index

        # Apply GAT layers with ELU activation
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
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


class MPNNLayer(MessagePassing):
    """
    A single Message Passing Neural Network (MPNN) layer.

    This layer aggregates messages from neighboring nodes and updates the central node's features.
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        """
        Initializes the MPNNLayer.

        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            dropout (float): Dropout rate for regularization.
        """
        super(MPNNLayer, self).__init__(aggr='add')  # Use "add" aggregation for message passing
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the MPNNLayer.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (Tensor): Edge index tensor of shape [2, num_edges].

        Returns:
            Tensor: Updated node feature matrix of shape [num_nodes, out_channels].
        """
        # Optionally add self-loops to include a node's own features
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """
        Constructs messages for each edge.

        Args:
            x_i (Tensor): Target node features of shape [num_edges, in_channels].
            x_j (Tensor): Source node features of shape [num_edges, in_channels].

        Returns:
            Tensor: Messages for each edge of shape [num_edges, out_channels].
        """
        # Concatenate source and target node features and pass through the MLP
        msg = self.mlp(torch.cat([x_i, x_j], dim=1))
        return msg

    def update(self, aggr_out):
        """
        Updates the node features after message aggregation.

        Args:
            aggr_out (Tensor): Aggregated messages of shape [num_nodes, out_channels].

        Returns:
            Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        return F.relu(aggr_out)


class MPNN_4L(BaseGNNModel):
    """
    A 4-layer Message Passing Neural Network (MPNN) for atom type classification.

    This model uses four MPNN layers followed by a fully connected layer for classification.
    """

    def __init__(self, num_node_features, num_atom_types, hidden_dim=1024, dropout=0.2):
        """
        Initializes the MPNN_4L model.

        Args:
            num_node_features (int): Number of input node features.
            num_atom_types (int): Number of output atom type classes.
            hidden_dim (int): Hidden dimension for the first layer.
            dropout (float): Dropout rate for regularization.
        """
        super(MPNN_4L, self).__init__()

        # Define four MPNN layers
        self.layer1 = MPNNLayer(num_node_features, hidden_dim, dropout)
        self.layer2 = MPNNLayer(hidden_dim, hidden_dim, dropout)
        self.layer3 = MPNNLayer(hidden_dim, hidden_dim // 2, dropout)
        self.layer4 = MPNNLayer(hidden_dim // 2, hidden_dim // 4, dropout)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim // 4, num_atom_types)

    def forward(self, data):
        """
        Forward pass of the MPNN_4L model.

        Args:
            data (Data): PyTorch Geometric Data object containing graph data.

        Returns:
            Tensor: Predicted atom types (logits).
        """
        x, edge_index = data.x, data.edge_index

        # Apply MPNN layers
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)
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