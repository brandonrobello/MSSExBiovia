import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import List
from atoMLtype.utils.SDFdataset import SDFdataset
from atoMLtype.GNN.GNNfeaturizer import GraphFeaturizer


class GNNdataset(Dataset):
    """
    Converts an SDF files into a PyTorch Geometric dataset for GNN training and loads a JSON file as
    the labels of the SDF molecule's atom typings. Only the filtered X_molecules are processed into graphs.
    Uses `GraphFeaturizer` to extract features.

    - Uses `SDFdataset` to load molecules and atom type labels, then filters and aligns the atoms.
    - Converts molecules into graph structures.    
    """

    def __init__(self, sdf_path: str, json_labels: str = None):
        """
        Args:
            sdf_path (str): Path to the SDF file.
            json_labels (str, optional): Path to the JSON file containing atom labels.
        """
        super().__init__()

        # Load the molecules from SDF into a custom class
        self.sdf_dataset = SDFdataset(sdf_path, json_labels)

        # Initialize the featurizer
        self.featurizer = GraphFeaturizer()

        # Create a mapping of unique atom types to indices
        self.label_encoder = LabelEncoder()
        self.all_labels = [label for labels in self.sdf_dataset.Y_labels.values() for label in labels]
        self.label_encoder.fit(self.all_labels)  # Learn the mapping
        
        # Process molecules into PyG graph format
        self.mol_graphs = self._process_molecules()

    def _process_molecules(self) -> List[Data]:
        """
        Converts X_molecules filtered molecules into PyTorch Geometric graph objects.
        """
        mol_graphs = []

        for mol_name, mol in self.sdf_dataset.X_molecules.items():
            try:
                graph = self.mol_to_graph(mol, mol_name)
                mol_graphs.append(graph)
            except Exception as e:
                print(f"Skipping molecule {mol_name} due to error: {e}")

        return mol_graphs

    def mol_to_graph(self, mol, mol_name):
        """
        Converts an RDKit molecule into a PyTorch Geometric graph.
        """
        atom_features, edge_indices, bond_features = self.featurizer.featurize(mol)

        # Convert to PyTorch tensors
        x = torch.tensor(atom_features, dtype=torch.float)  # Shape: [num_atoms, feature_dim]
        edge_index = torch.tensor(edge_indices, dtype=torch.long)  # Shape: [2, num_edges]
        edge_attr = torch.tensor(bond_features, dtype=torch.float)  # Shape: [num_edges, bond_feature_dim]

        # Convert labels to tensor
        y_values = self.sdf_dataset.Y_labels[mol_name]
        y = torch.tensor(self.label_encoder.transform(y_values), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


    def __len__(self):
        return len(self.mol_graphs)

    def __getitem__(self, idx):
        return self.mol_graphs[idx]