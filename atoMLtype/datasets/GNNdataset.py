import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from typing import List
from atoMLtype.datasets.SDFdataset import SDFdataset
from atoMLtype.datasets.GNNfeaturizer import GraphFeaturizer
from atoMLtype.models.ModelEncoder import ModelEncoder
from atoMLtype.utils.logging_utils import get_logger

class GNNdataset(Dataset):
    """
    Dataset class for converting labeled or unlabeled molecules from an SDF file into 
    PyTorch Geometric graph objects for GNN training.

    Attributes:
        sdf_dataset (SDFdataset): The dataset wrapper for molecules and labels.
        featurizer (GraphFeaturizer): Extracts atom/bond features and connectivity.
        mol_graphs (List[Data]): List of processed PyTorch Geometric graph objects.
        raw_labels (List[str]): Flattened list of all atom labels.
        encoded_labels (List[int] | None): Flattened list of encoded atom labels, if available.
        labelled_graphs (bool): Whether to return graphs with labels.
    """


    def __init__(self, 
                 sdf_path: str, 
                 label_path: str = None, 
                 directed_graph: bool = False, 
                 labeled: bool = True, 
                 encoder: ModelEncoder = None):
        """
        Initializes the GNNdataset by loading molecules and converting them to graphs.

        Args:
            sdf_path (str): Path to the SDF file.
            label_path (str, optional): Path to the JSON file with atom labels.
            directed_graph (bool): Whether to treat bonds as directed edges.
            labeled (bool): Whether to include atom labels in graph data.
            encoder (ModelEncoder, optional): Label encoder to use for atom types.
        """
        super().__init__()
        self.log = get_logger(__name__)
        self.labelled_graphs = labeled

        # Load molecules and atom labels (if provided)
        self.sdf_dataset = SDFdataset(sdf_path, label_path)

        # Initialize molecular graph featurizer, directed or undirected
        self.featurizer = GraphFeaturizer(directed_graph)

        # Flatten all atom labels for fitting the label encoder
        self.raw_labels = [
            label for labels in self.sdf_dataset.label_dict.values()
            for label in labels
        ]

        # Encode labels only if the dataset is labeled and labels are available
        if self.sdf_dataset.has_labels and self.labelled_graphs:
            encoder.fit(self.raw_labels)
            self.encoded_labels = encoder.transform(self.raw_labels)
        else:
            self.encoded_labels = None

        # Convert molecules to graph representations
        self.mol_graphs = self._process_molecules(self.labelled_graphs, encoder)

    def _process_molecules(self, labeled: bool = True, encoder: ModelEncoder = None) -> List[Data]:
        """
        Processes molecules into graph objects with atom/bond features.

        Args:
            labeled (bool): Whether to include atom labels in graph objects.
            encoder (ModelEncoder, optional): Encoder for transforming labels to integers.

        Returns:
            List[Data]: PyTorch Geometric graph objects representing molecules.
        """
        graphs = []
        global_atom_idx = 0  # Track atom index across all molecules

        mols = self.sdf_dataset.get_molecules(labeled=labeled)

        self.log.info(
            f"Attempting to process {len(mols)} {'*labeled*' if labeled else '*all*'} mols from sdf dataset to graphs."
        )

        for mol in mols:
            try:
                graph = self._mol_to_graph(mol, global_atom_idx, labeled, encoder)
                global_atom_idx += graph.num_nodes  # increment by number of atoms
                graphs.append(graph)
            except Exception as e:
                mol_name = mol.GetProp("_Name")
                self.log.warning(f"Skipping molecule {mol_name} due to error: {e}")
        
        self.log.info(f"Processed {len(graphs)} molecular graphs from {len(mols)} molecules.")
        return graphs


    def _mol_to_graph(self, 
                      mol: Chem.Mol, 
                      global_atom_start_idx: int, 
                      labeled: bool, 
                      encoder: ModelEncoder) -> Data:
        """
        Converts a single RDKit molecule into a PyTorch Geometric Data object.

        Args:
            mol (Chem.Mol): RDKit molecule.
            global_atom_start_idx (int): Running global atom index.
            labeled (bool): Whether to include label data.
            encoder (ModelEncoder): Encoder to transform string labels.

        Returns:
            Data: Graph object with atom features, bond edges, and optional labels.
        """
        # Extract molecular graph features
        atom_features, edge_indices, bond_features = self.featurizer.featurize(mol)

        # Convert to PyTorch tensors
        x = torch.tensor(atom_features, dtype=torch.float)  # Shape: [num_atoms, feature_dim]
        edge_index = torch.tensor(edge_indices, dtype=torch.long)  # Shape: [2, num_edges]
        edge_attr = torch.tensor(bond_features, dtype=torch.float)  # Shape: [num_edges, bond_feature_dim]
        y = None

        # Encode labels if labeled=True
        if labeled:
            labels = [atom.GetProp("atom_type") for atom in mol.GetAtoms()]
            y = torch.tensor(encoder.transform(labels), dtype=torch.long)

        # Get the size/num_atoms in the data
        num_atoms = x.size(0)

        # Construct PyTorch Geometric Data object
        graph = Data(x=x, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=y
                     )
        
        # Attach metadata for downstream analysis
        graph.has_labels = labeled
        graph.mol_name = mol.GetProp("_Name")  # Attach molecule identifier
        graph.atom_idx_in_mol = torch.arange(num_atoms)  # [0, ..., num_atoms-1] in mol
        graph.global_atom_idx = torch.arange(
            global_atom_start_idx,
            global_atom_start_idx + num_atoms)  # Idx of atoms globally is processed dataset
        
        if labeled:
            graph.json_labels = mol.GetProp("atom_labels")

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