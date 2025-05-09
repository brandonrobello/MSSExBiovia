import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from typing import List, Union
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

        # Set the is filtered flag based on encoder passed to dataset
        self.is_filtered = True if encoder and encoder.is_fitted else False

        # Load molecules and atom labels (if provided)
        self.sdf_dataset = SDFdataset(sdf_path, label_path)

        # Initialize molecular graph featurizer, directed or undirected
        self.featurizer = GraphFeaturizer(directed_graph)

        # Flatten all atom labels for fitting the label encoder
        self.raw_labels = [
            label for labels in self.sdf_dataset.label_dict.values()
            for label in labels
        ]

        # Encode labels only if the dataset has labels and labelled graphs are desired
        if self.sdf_dataset.has_labels and self.labelled_graphs:
            if encoder is not None:
                if not encoder.is_fitted:
                    encoder.fit(self.raw_labels)
                    self.log.info("Fitted encoder on dataset labels.")
                    self.encoded_labels = encoder.transform(self.raw_labels)
                else:
                    self.log.info("Encoder already fitted; skipping re-fitting.")
            else:
                raise ValueError("Labeled dataset requires a provided ModelEncoder.")
        else:
            self.encoded_labels = None

        # Convert molecules to graph representations
        self.mols = self.sdf_dataset.get_molecules(labeled=labeled)
        self.mol_graphs = self._process_molecules(self.labelled_graphs, encoder)

        # If encoder already fitted, filter mols, graphs, and labels
        if self.labelled_graphs and self.is_filtered:
            self.filtered_mols, self.filtered_mol_graphs = self.filter_to_encoder_labels(encoder)
            self.filtered_raw_labels = [
                atom.GetProp("atom_type") for mol in self.filtered_mols
                for atom in mol.GetAtoms()
            ]
            self.filtered_encoded_labels = encoder.transform(self.raw_labels)

    def get_mol(self, idx: Union[int, str]) -> Chem.Mol:
        """
        Retrieve the RDKit molecule by index or name (_Name property).

        Args:
            idx (int or str): Index or molecule name.

        Returns:
            Chem.Mol: RDKit Mol object.

        Raises:
            ValueError: If no molecule with that name exists.
        """
        target_list = self.filtered_mols if self.is_filtered else self.mols

        if isinstance(idx, str):
            for mol in target_list:
                if mol.HasProp("_Name") and mol.GetProp("_Name") == idx:
                    return mol
            raise ValueError(f"Molecule with name '{idx}' not found.")
        elif isinstance(idx, int):
            return target_list[idx]
        else:
            raise TypeError("Index must be an integer or a string (_Name).")
        
    def get_mol_graph(self, idx: Union[int, str]) -> Data:
        """
        Get PyG graph by molecule name or index.

        Args:
            idx (int or str): Molecule index or _Name.

        Returns:
            torch_geometric.data.Data: Corresponding graph.
        """
        target_list = self.filtered_mol_graphs if self.is_filtered else self.mol_graphs

        if isinstance(idx, str):
            for i, g in enumerate(target_list):
                if hasattr(g, "mol_name") and g.mol_name == idx:
                    return g
            raise ValueError(f"Molecule with name '{idx}' not found in graphs.")
        elif isinstance(idx, int):
            return target_list[idx]
        else:
            raise TypeError("Index must be int or str")
        
    def filter_to_encoder_labels(self, encoder: ModelEncoder):
        """
        Filters out molecules containing atom labels not present in the encoder.

        Args:
            encoder (ModelEncoder): Trained encoder whose classes define allowed labels.
        """
        
        known_labels = set(encoder.classes)
        filtered_mols = []
        filtered_graphs = []

        for mol, graph in zip(self.mols, self.mol_graphs):
            atom_labels = [atom.GetProp("atom_type") for atom in mol.GetAtoms()]
            if encoder.collapse:
                atom_labels = [encoder.label_map.get(label, label) for label in atom_labels]

            if all(label in known_labels for label in atom_labels):
                filtered_mols.append(mol)
                filtered_graphs.append(graph)
            else:
                mol_name = mol.GetProp("_Name")
                self.log.warning(f"Skipping molecule '{mol_name}' due to unknown labels.")

        removed = len(self.mols) - len(filtered_mols)
        self.log.info(f"Filtered out {removed} molecules with unknown labels.")

        return filtered_mols, filtered_graphs



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

        self.log.info(
            f"Attempting to process {len(self.mols)} {'*labeled*' if labeled else '*all*'} mols from sdf dataset to graphs."
        )

        for mol in self.mols:
            try:
                graph = self._mol_to_graph(mol, global_atom_idx, labeled, encoder)
                global_atom_idx += graph.num_nodes  # increment by number of atoms
                graphs.append(graph)
            except Exception as e:
                mol_name = mol.GetProp("_Name")
                self.log.warning(f"Skipping molecule {mol_name} due to error: {e}")
        
        self.log.info(f"Processed {len(graphs)} molecular graphs from {len(self.mols)} molecules.")
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
        return len(self.filtered_mol_graphs) if self.is_filtered else len(self.mol_graphs)

    def __getitem__(self, idx):
        """
        Retrieves a molecule graph by index.

        Args:
            idx (int): Index of the molecule.

        Returns:
            Data: PyTorch Geometric graph object.
        """
        return self.filtered_mol_graphs[idx] if self.is_filtered else self.mol_graphs[idx]