import numpy as np
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import rdqueries
from atoMLtype.datasets.BaseFeaturizer import BaseFeaturizer

class GraphFeaturizer(BaseFeaturizer):
    """
    Graph featurizer for molecular graph construction with atom and bond features.

    Generates directed or undirected edges, and encodes features suitable for 
    graph neural networks like D-MPNN or GCN.

    Atom features include:
        - Element, degree, formal charge, chirality, #Hs, aromaticity
        - Mass, hybridization, bridgehead indicator, EWG neighbor count
        - Ring size context

    Bond features include:
        - Bond type, stereo, in-ring status, conjugation

    Args:
        directed (bool): Whether to use directed bonds (i→j and j→i).
    """

    def __init__(self, directed: bool = False):
        """
        Initializes the GraphFeaturizer with a predefined hybridization map & directed.
        """
        self.directed = directed
        self.hybridization_map = {
            Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0, 0],
            Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1, 0],
            Chem.rdchem.HybridizationType.UNSPECIFIED: [0, 0, 0, 0, 0, 1],  # Unknown case
        }


    def featurize(self, molecule: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Converts a molecule into graph format with atom and bond features.

        Args:
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Atom features [num_atoms, atom_feat_dim]
                - Edge indices [2, num_edges]
                - Bond features [num_edges, bond_feat_dim]
        """
        # Extract atom features
        atom_features = [
            self.get_atom_features(atom, molecule) for atom in molecule.GetAtoms()
            ] 

        # Bond features and edges
        edge_index = []
        edge_attr = []

        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)

            # Always add i → j
            edge_index.append([i, j])
            edge_attr.append(bond_feat)

            # If directed, also add j → i
            if self.directed:
                edge_index.append([j, i])
                edge_attr.append(bond_feat)


        return (
            np.array(atom_features, dtype=np.float32), # Shape: [num_atoms, atom_dim]
            np.array(edge_index, dtype=np.int64).T, # Shape: [2, num_edges]
            np.array(edge_attr, dtype=np.float32), # Shape: [num_edges, bond_dim]
        )

    def get_atom_features(self, atom: Chem.Atom, molecule: Chem.Mol) -> np.ndarray:
        """
        Extracts features for an atom in an RDKit molecule.

        Args:
            atom (Chem.Atom): RDKit atom object.
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            np.ndarray: Atom feature vector.
        """
        # Standard atomic descriptors
        atom_type = self.one_hot_encode(atom.GetAtomicNum(), list(range(1, 101)))
        degree = self.one_hot_encode(atom.GetDegree(), list(range(12)))
        formal_charge = [atom.GetFormalCharge()]
        chiral_center = [int(atom.HasProp('_ChiralityPossible'))]
        chirality_type = self.one_hot_encode(atom.GetChiralTag(), [
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        ])
        num_h = self.one_hot_encode(atom.GetTotalNumHs(), list(range(6)))
        atomic_mass = [atom.GetMass() / 100.0]
        aromaticity = [int(atom.GetIsAromatic())]
        radical_electrons = self.one_hot_encode(atom.GetNumRadicalElectrons(), list(range(6)))
        hybridization = self.hybridization_map.get(atom.GetHybridization(), [0, 0, 0, 0, 0, 1])  # Use map

        # Bridgehead indicator
        qa = rdqueries.IsBridgeheadQueryAtom()
        bridge_atoms = molecule.GetAtomsMatchingQuery(qa)
        bridgehead = [1 if atom.GetIdx() in bridge_atoms else 0]

        # Count electron-withdrawing neighbors (F, Cl, Br, I, N, O, S)
        ewg_atoms = [9, 17, 35, 53, 7, 8, 16]
        sum_ewg_neighbors = [sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() in ewg_atoms)]

        # Smallest ring size containing this atom
        rings = Chem.GetSymmSSSR(molecule)
        ring_sizes = [len(ring) for ring in rings if atom.GetIdx() in ring]
        smallest_ring_size = min(ring_sizes) if ring_sizes else 0
        ring_size_encoding = self.one_hot_encode(smallest_ring_size, list(range(10)))

        return np.array(
            atom_type + degree + formal_charge + chiral_center + chirality_type +
            num_h + atomic_mass + aromaticity + radical_electrons + hybridization +
            bridgehead + sum_ewg_neighbors + ring_size_encoding,
            dtype=np.float32
        )

    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        Computes features for a bond.

        Args:
            bond (Chem.Bond): Bond object from RDKit.

        Returns:
            np.ndarray: Bond feature vector.
        """
        if bond is None:
            return np.zeros(14, dtype=np.float32)

        # Bond features
        bond_type = self.one_hot_encode(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE, 
            Chem.rdchem.BondType.DOUBLE, 
            Chem.rdchem.BondType.TRIPLE, 
            Chem.rdchem.BondType.AROMATIC
        ])

        stereo = self.one_hot_encode(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ])  

        in_ring = [int(bond.IsInRing())]      
        conjugated = [int(bond.GetIsConjugated())]  

        features = bond_type + stereo + in_ring + conjugated
        return np.array(features, dtype=np.float32)

    @staticmethod
    def one_hot_encode(value, choices: list) -> list:
        """
        Encodes `value` as a one-hot vector based on `choices`.

        Args:
            value: Input value to encode.
            choices (list): Valid options for one-hot encoding.

        Returns:
            list: One-hot encoded vector of length len(choices) + 1 (extra bin for unknown).
        """
        encoding = [0] * (len(choices) + 1)

        if value in choices:
            encoding[choices.index(value)] = 1  # Standard one-hot encoding
        else:
            encoding[-1] = 1  # Assign to last index if value > max(choices)

        return encoding