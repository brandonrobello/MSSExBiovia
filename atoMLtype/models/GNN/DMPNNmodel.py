import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from atoMLtype.models.ModelOutput import ModelOutput
from atoMLtype.models.ModelEncoder import ModelEncoder
from atoMLtype.models.BaseGNNModel import BaseGNNModel



class AtomBondMPNNLayer(MessagePassing):
    """
    Message-passing layer for atom-bond graphs with optional attention.

    Constructs edge-conditioned messages from source/target atoms and bond features.
    Optionally applies learned attention to modulate messages.

    Args:
        atom_dim (int): Dimension of input atom embeddings.
        bond_dim (int): Dimension of bond feature vectors.
        hidden_dim (int): Dimension of hidden messages.
        use_attention (bool): Whether to apply attention to messages.
    """
    def __init__(self, 
                 atom_dim: int, 
                 bond_dim: int, 
                 hidden_dim: int = 1024,
                 use_attention: bool = True):
        super().__init__(aggr='add')
        self.use_attention = use_attention
        input_dim = atom_dim * 2 + bond_dim


        # Message MLP: transforms concatenated (x_i, x_j, edge_attr) to message
        self.edge_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Optional attention mechanism to weight messages
        if self.use_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Applies message passing to the input graph.

        Args:
            x (Tensor): Atom embeddings [num_atoms, atom_dim]
            edge_index (Tensor): Edge indices [2, num_edges]
            edge_attr (Tensor): Bond features [num_edges, bond_dim]

        Returns:
            Tensor: Updated atom embeddings [num_atoms, hidden_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages for each edge.

        Args:
            x_i: Target node embedding (receives the message)
            x_j: Source node embedding (sends the message)
            edge_attr: Bond feature for this edge

        Returns:
            msg: Message vector, optionally attention-weighted
        """        
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.edge_net(message_input)

        if self.use_attention:
            attn_score = self.att_mlp(message_input).squeeze(-1)         # [num_edges]
            attn_weight = torch.sigmoid(attn_score)                      # Use sigmoid to scale [0, 1]
            msg = msg * attn_weight.unsqueeze(-1)                        # Apply attention per message

        return msg

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Updates node features after aggregation.

        Args:
            aggr_out: Aggregated messages for each node.

        Returns:
            Updated node features.
        """
        return aggr_out
    

class AtomBondMPNN(BaseGNNModel):
    """
    Custom Message-Passing Neural Network with edge-conditioned messages and optional attention.

    Applies multiple AtomBondMPNNLayer layers to propagate information through atom-bond graphs.

    Architecture:
        - Atom encoder (Linear)
        - N× MPNNLayer (residual)
        - MLP classifier head

    Args:
        atom_input_dim (int): Dimension of initial atom features.
        bond_input_dim (int): Dimension of bond features.
        encoder (ModelEncoder): Encoder used for atom type labels.
        num_layers (int): Number of message-passing layers.
        hidden_dim (int): Dimension of intermediate layers.
        use_attention (bool): Whether to apply attention in each MPNNLayer.
    """

    def __init__(self, 
                 atom_input_dim: int, 
                 bond_input_dim: int, 
                 encoder: ModelEncoder,
                 num_layers: int = 5, 
                 hidden_dim: int = 1024,
                 use_attention: bool = False):
        super().__init__(encoder)

        # Metadata required for save method
        self.atom_input_dim = atom_input_dim
        self.bond_input_dim = bond_input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Initial linear projection from raw atom features
        self.atom_encoder = nn.Linear(atom_input_dim, hidden_dim)

        # Stack of MPNN layers
        self.message_layers = nn.ModuleList([
            AtomBondMPNNLayer(
                atom_dim=hidden_dim,
                bond_dim=bond_input_dim,
                hidden_dim=hidden_dim,
                use_attention=use_attention
            ) for _ in range(num_layers)
        ])

        # Classifier head for atom type prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes)
        )
    
    def _get_metadata(self) -> dict:
        """
        Returns architecture metadata for saving/loading.
        """
        return {
            'atom_input_dim': self.atom_input_dim,
            'bond_input_dim': self.bond_input_dim,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'use_attention': self.use_attention
        }

    def forward(self, data) -> ModelOutput:
        """
        Forward pass through the MPNN.

        Args:
            data (Data): PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_attr: Bond features

        Returns:
            ModelOutput: Contains logits and optional analysis outputs.
        """
        x = self.atom_encoder(data.x)   # Project initial atom features
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # For analsyis mode
        analysis = {} if self.is_analysis_enabled() else None

        if self.is_analysis_enabled():
            analysis["x_initial"] = x.clone().detach()

        for i, layer in enumerate(self.message_layers):
            x = x + layer(x, edge_index, edge_attr) # Residual update

        if self.is_analysis_enabled():
            analysis["clf_embeddings"] = self.classifier[:-1](x).detach()

        logits = self.classifier(x)

        return ModelOutput(
            logits=logits,
            analysis=analysis,
        )
    