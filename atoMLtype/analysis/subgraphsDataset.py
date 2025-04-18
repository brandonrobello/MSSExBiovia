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

