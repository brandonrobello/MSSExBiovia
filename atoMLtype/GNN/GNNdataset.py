import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import List
from atoMLtype.utils.SDFdataset import SDFdataset
from atoMLtype.GNN.GNNfeaturizer import GraphFeaturizer, GraphFeaturizer_DMPNN
from atoMLtype.utils.label_utils import collapse_atom_types


class GNNdataset(Dataset):
    """
    Converts SDF files into a PyTorch Geometric dataset for GNN training and loads a JSON file
    as the labels of the SDF molecule's atom typings. Only the filtered X_molecules are processed
    into graphs. Uses `GraphFeaturizer` to extract features.

    - Uses `SDFdataset` to load molecules and atom type labels, then filters and aligns the atoms.
    - Converts molecules into graph structures.
    """

    def __init__(self, sdf_path: str, json_labels: str = None, collapse: bool = False):
        """
        Initializes the GNNdataset.

        Args:
            sdf_path (str): Path to the SDF file.
            json_labels (str, optional): Path to the JSON file containing atom labels.
            collapse (bool, optional): Whether to collapse atom type labels. Defaults to False.
        """
        super().__init__()
        self.collapse = collapse

        # Load the molecules from SDF into a custom class
        self.sdf_dataset = SDFdataset(sdf_path, json_labels)

        # Initialize the featurizer
        self.featurizer = GraphFeaturizer()

        # Create a mapping of unique atom types to indices
        self.label_encoder = LabelEncoder()
        raw_labels = [label for labels in self.sdf_dataset.Y_labels.values() for label in labels]

        # Apply collapsing if enabled
        if self.collapse:
            collapsed_labels = collapse_atom_types(raw_labels)
            self.label_encoder.fit(collapsed_labels)
            self.all_labels = collapsed_labels
        else:
            self.label_encoder.fit(raw_labels)
            self.all_labels = raw_labels

        # Process molecules into PyG graph format
        self.mol_graphs, self.mol_names = self._process_molecules()

        # Load molecule graphs into subgraph dataset
        self.dataset_subgraphs = GNNdataset_subgraphs(self.mol_graphs, self.collapse)

    def _process_molecules(self) -> List[Data]:
        """
        Converts filtered molecules into PyTorch Geometric graph objects.

        Returns:
            List[Data]: List of PyTorch Geometric graph objects.
        """
        mol_graphs = []
        mol_names = []

        for mol_name, mol in self.sdf_dataset.X_molecules.items():
            try:
                graph = self.mol_to_graph(mol, mol_name)
                mol_graphs.append(graph)
                mol_names.append(mol_name)
            except Exception as e:
                print(f"Skipping molecule {mol_name} due to error: {e}")

        return mol_graphs, mol_names

    def mol_to_graph(self, mol, mol_name):
        """
        Converts an RDKit molecule into a PyTorch Geometric graph.

        Args:
            mol (Chem.Mol): RDKit molecule object.
            mol_name (str): Name of the molecule.

        Returns:
            Data: PyTorch Geometric graph object.
        """
        # Extract atom and bond features using the featurizer
        atom_features, edge_indices, bond_features = self.featurizer.featurize(mol)

        # Convert features to PyTorch tensors
        x = torch.tensor(atom_features, dtype=torch.float)  # Shape: [num_atoms, feature_dim]
        edge_index = torch.tensor(edge_indices, dtype=torch.long)  # Shape: [2, num_edges]
        edge_attr = torch.tensor(bond_features, dtype=torch.float)  # Shape: [num_edges, bond_feature_dim]

        # Convert labels to tensor
        y_values = self.sdf_dataset.Y_labels[mol_name]
        labels = collapse_atom_types(y_values) if self.collapse else y_values
        y = torch.tensor(self.label_encoder.transform(labels), dtype=torch.long)

        # Create a PyTorch Geometric Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_values=y_values)
        graph.mol_name = mol_name  # Attach molecule identifier

        return graph

    def __len__(self):
        """
        Returns the number of molecules in the dataset.

        Returns:
            int: Number of molecules.
        """
        return len(self.mol_graphs)

    def __getitem__(self, idx):
        """
        Retrieves a molecule graph by index.

        Args:
            idx (int): Index of the molecule.

        Returns:
            Data: PyTorch Geometric graph object.
        """
        return self.mol_graphs[idx]


class GNNdataset_subgraphs(Dataset):
    """
    Creates a dataset of per-atom subgraphs for all molecules in `GNNdataset`.

    - Extracts 1-hop subgraphs for each atom.
    - Preserves molecule metadata for later reference.
    """

    def __init__(self, mol_graphs, collapse: bool = False):
        """
        Initializes the GNNdataset_subgraphs.

        Args:
            mol_graphs (List[Data]): List of PyTorch Geometric graphs representing full molecules.
            collapse (bool, optional): Whether to collapse atom type labels. Defaults to False.
        """
        super().__init__()
        self.collapse = collapse

        # Store extracted subgraphs
        self.mol_subgraphs = self._process_subgraphs(mol_graphs)

        # Store all labels for assessing distribution
        self.all_labels = [
            collapse_atom_types([label.item()])[0] if self.collapse else label.item()
            for subgraph in self.mol_subgraphs if hasattr(subgraph, 'y') for label in subgraph.y
        ]

    def _extract_subgraph(self, node_idx, data):
        """
        Extracts a subgraph centered around `node_idx`, including only its direct neighbors.

        Args:
            node_idx (int): The central node index.
            data (torch_geometric.data.Data): The full PyTorch Geometric graph.

        Returns:
            Data: A PyTorch Geometric subgraph containing only the correct nodes and edges.
        """
        # Find 1-hop neighbors
        mask = (data.edge_index[0] == node_idx) | (data.edge_index[1] == node_idx)
        edge_indices = torch.nonzero(mask, as_tuple=True)[0]

        # Collect all connected nodes
        directly_connected_nodes = data.edge_index[:, edge_indices].view(-1)

        # Ensure uniqueness and include the central node itself
        directly_connected_nodes = torch.unique(torch.cat([directly_connected_nodes, torch.tensor([node_idx])]))

        # Convert to tensor
        directly_connected_nodes = torch.tensor(sorted(directly_connected_nodes), dtype=torch.long)

        # Use Data.subgraph() method to extract the subgraph
        sub_data = data.subgraph(directly_connected_nodes)

        # Retain molecule metadata
        sub_data.mol_name = data.mol_name
        sub_data.central_atom_idx = node_idx  # Keep track of the atom at the center

        return sub_data

    def _process_subgraphs(self, whole_graphs):
        """
        Extracts 1-hop subgraphs for all atoms in all molecules.

        Args:
            whole_graphs (List[Data]): List of full-molecule PyTorch Geometric graphs.

        Returns:
            List[Data]: List of per-atom subgraphs.
        """
        subgraphs = []

        for whole_graph in whole_graphs:
            # Extract subgraphs for each atom in the molecule
            atomic_graphs = [self._extract_subgraph(i, whole_graph) for i in range(whole_graph.num_nodes)]
            subgraphs.extend(atomic_graphs)

        return subgraphs

    def __len__(self):
        """
        Returns the number of subgraphs in the dataset.

        Returns:
            int: Number of subgraphs.
        """
        return len(self.mol_subgraphs)

    def __getitem__(self, idx):
        """
        Retrieves a subgraph by index.

        Args:
            idx (int): Index of the subgraph.

        Returns:
            Data: PyTorch Geometric subgraph object.
        """
        return self.mol_subgraphs[idx]


class MPNNdataset(Dataset):
    """
    PyTorch Geometric dataset for training bond-focused MPNNs using molecules from an SDF file 
    and atom labels from a JSON file.

    Each molecule is encoded using `GraphFeaturizer_DMPNN` into graph format with atom features, 
    edge indices, and bond features. Atom labels are mapped to class indices via `LabelEncoder`.
    """

    def __init__(self, sdf_path: str, json_labels: str = None, collapse=True):
        """
        Initializes the MPNNdataset.

        Args:
            sdf_path (str): Path to the SDF file.
            json_labels (str, optional): Path to the JSON file containing atom labels.
            collapse (bool, optional): Whether to collapse atom type labels. Defaults to True.
        """
        super().__init__()
        self.collapse = collapse

        # Load the molecules from SDF into a custom class
        self.sdf_dataset = SDFdataset(sdf_path, json_labels)

        # Initialize the featurizer
        self.featurizer = GraphFeaturizer_DMPNN()

        # Create a mapping of unique atom types to indices
        self.label_encoder = LabelEncoder()

        # Get flat list of labels from all molecules
        raw_labels = [label for labels in self.sdf_dataset.Y_labels.values() for label in labels]

        # Apply collapsing if enabled
        if self.collapse:
            collapsed_labels = [collapse_atom_types([label])[0] for label in raw_labels]
            self.label_encoder.fit(collapsed_labels)
            self.all_labels = collapsed_labels
        else:
            self.label_encoder.fit(raw_labels)
            self.all_labels = raw_labels

        # Process molecules into PyG graph format
        self.mol_graphs, self.mol_names = self._process_molecules()

    def _process_molecules(self) -> List[Data]:
        """
        Converts filtered molecules into PyTorch Geometric graph objects.

        Returns:
            List[Data]: List of PyTorch Geometric graph objects.
        """
        mol_graphs = []
        mol_names = []
        global_atom_idx = 0  # running index across all atoms in all molecules of the dataset

        for mol_name, mol in self.sdf_dataset.X_molecules.items():
            try:
                bondGraph = self.mol_to_graph(mol, mol_name, global_atom_idx) # Pass global index
                global_atom_idx += bondGraph.num_nodes  # increment by number of atoms
                mol_graphs.append(bondGraph)
                mol_names.append(mol_name)
            except Exception as e:
                print(f"Skipping molecule {mol_name} due to error: {e}")

        return mol_graphs, mol_names

    def mol_to_graph(self, mol, mol_name, global_atom_start_idx):
        """
        Converts an RDKit molecule into a PyTorch Geometric graph.

        Args:
            mol (Chem.Mol): RDKit molecule object.
            mol_name (str): Name of the molecule.

        Returns:
            Data: PyTorch Geometric graph object.
        """
        x, edge_index, edge_attr = self.featurizer.featurize(mol)

        # Convert labels from str to labels in a tensor
        y_values = self.sdf_dataset.Y_labels[mol_name]  # Str y_values
        labels = collapse_atom_types(y_values) if self.collapse else y_values
        y = torch.tensor(self.label_encoder.transform(labels), dtype=torch.long)

        # Get the size/num_atoms in the data
        num_atoms = x.size(0)

        # Create a PyTorch Geometric Data object
        Graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_values=y_values
        )
        # Custom metadata
        Graph.mol_name = mol_name  # Attach molecule identifier
        Graph.atom_idx_in_mol = torch.arange(num_atoms)  # [0, ..., num_atoms-1] in mol
        Graph.global_atom_idx = torch.arange(
            global_atom_start_idx,
            global_atom_start_idx + num_atoms)  # Idx of atoms globally is processed dataset
        
        return Graph

    def __len__(self):
        """
        Returns the number of molecules in the dataset.

        Returns:
            int: Number of molecules.
        """
        return len(self.mol_graphs)

    def __getitem__(self, idx):
        """
        Retrieves a molecule graph by index.

        Args:
            idx (int): Index of the molecule.

        Returns:
            Data: PyTorch Geometric graph object.
        """
        return self.mol_graphs[idx]