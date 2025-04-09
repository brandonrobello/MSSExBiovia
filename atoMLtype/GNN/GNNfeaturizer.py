import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from atoMLtype.utils.featurizer import atoMLtype_featurizer
import torch
from rdkit.Chem import rdqueries


class GraphFeaturizer(atoMLtype_featurizer):
    """
    Generates atom and bond features for molecular graphs using RDKit.

    Atom features include:
        - Atomic number
        - Degree
        - Formal charge
        - Chirality
        - Hydrogen count
        - Mass
        - Aromaticity
        - Radical electrons
        - Hybridization
        - Smallest ring size

    Bond features include:
        - Bond type
        - Stereochemistry
        - Conjugation
        - Ring membership

    Suitable for general GNN architectures (e.g., GCN, GAT, GIN).
    """

    def __init__(self):
        """
        Initializes the GraphFeaturizer with a predefined hybridization map.
        """
        self.hybridization_map = {
            Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0, 0],
            Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1, 0],
            Chem.rdchem.HybridizationType.UNSPECIFIED: [0, 0, 0, 0, 0, 1],  # Unknown case
        }

    def featurize(self, molecule):
        """
        Extracts features for all atoms and bonds in a molecule.

        Args:
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - atom_features: (num_atoms, feature_dim)
                - edge_indices: (2, num_edges)
                - bond_features: (num_edges, bond_feature_dim)
        """
        # Extract atom features
        atom_features = [self.get_atom_features(atom, molecule) for atom in molecule.GetAtoms()]

        # Extract bond features and edge indices
        bond_features = []
        edge_indices = []
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)

            # Add edge i → j
            edge_indices.append([i, j])
            bond_features.append(bond_feat)

        return (
            np.array(atom_features, dtype=np.float32),
            np.array(edge_indices, dtype=np.int64).T,
            np.array(bond_features, dtype=np.float32),
        )

    def get_atom_features(self, atom, molecule):
        """
        Extracts features for an atom in an RDKit molecule.

        Args:
            atom (Chem.Atom): RDKit atom object.
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            np.ndarray: Atom feature vector.
        """
        # Basic atom features
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

        # Smallest ring size
        rings = Chem.GetSymmSSSR(molecule)
        ring_sizes = [len(ring) for ring in rings if atom.GetIdx() in ring]
        smallest_ring_size = min(ring_sizes) if ring_sizes else 0
        ring_size_encoding = self.one_hot_encode(smallest_ring_size, list(range(10)))

        return np.array(
            atom_type + degree + formal_charge + chiral_center + chirality_type +
            num_h + atomic_mass + aromaticity + radical_electrons + hybridization +
            ring_size_encoding,
            dtype=np.float32
        )

    def get_bond_features(self, bond):
        """
        Extracts features for a bond in an RDKit molecule.

        Args:
            bond (Chem.Bond): RDKit bond object.

        Returns:
            np.ndarray: Bond feature vector.
        """
        if bond is None:
            return np.zeros(14, dtype=np.float32)  # Zero vector for non-existent bonds

        # Bond features
        bond_type = self.one_hot_encode(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE, 
            Chem.rdchem.BondType.DOUBLE, 
            Chem.rdchem.BondType.TRIPLE, 
            Chem.rdchem.BondType.AROMATIC
        ])
        bond_type.append(1 if sum(bond_type) == 0 else 0)  # 'Unknown' category

        stereo = self.one_hot_encode(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ])
        stereo.append(1 if sum(stereo) == 0 else 0)  # 'Unknown' category

        in_ring = [int(bond.IsInRing())]
        conjugated = [int(bond.GetIsConjugated())]

        return np.array(bond_type + stereo + in_ring + conjugated, dtype=np.float32)

    @staticmethod
    def one_hot_encode(value, choices):
        """
        One-hot encodes a value given a list of choices.

        Args:
            value: The value to encode.
            choices (list): List of possible choices.

        Returns:
            list: One-hot encoded vector.
        """
        encoding = [0] * (len(choices) + 1)

        if value in choices:
            encoding[choices.index(value)] = 1  # Standard one-hot encoding
        else:
            encoding[-1] = 1  # Assign to last index if value > max(choices)

        return encoding

class GraphFeaturizer_DMPNN(atoMLtype_featurizer):
    """
    Generates directed atom-bond features for use in MPNNs such as D-MPNN.

    Constructs directed edges with shared bond features for i→j and j→i directions.
    Atom features extend the base featurizer by including:
        - Bridgehead atom indicator
        - Count of electron-withdrawing group (EWG) neighbors

    Designed for models requiring directional message passing with chemically enriched inputs.
    """

    def __init__(self):
        """
        Initializes the GraphFeaturizer_DMPNN with a predefined hybridization map.
        """
        self.hybridization_map = {
            Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0, 0],
            Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1, 0],
            Chem.rdchem.HybridizationType.UNSPECIFIED: [0, 0, 0, 0, 0, 1],  # Unknown case
        }

    def featurize(self, molecule):
        """
        Extracts features for all atoms and bonds in a molecule.

        Args:
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - atom_features: (num_atoms, feature_dim)
                - edge_index: (2, num_edges)
                - bond_features: (num_edges, bond_feature_dim)
        """
        # Extract atom features
        atom_features = np.array([
            self.get_atom_features(atom, molecule) for atom in molecule.GetAtoms()
        ])
        atom_features = torch.tensor(atom_features, dtype=torch.float)  # [num_atoms, atom_dim]

        # Extract bond features and edge indices
        edge_index = []
        edge_attr = []
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = torch.tensor(self.get_bond_features(bond), dtype=torch.float)

            # Add directed edge i → j
            edge_index.append([i, j])
            edge_attr.append(bond_feat)

            # Add directed edge j → i
            edge_index.append([j, i])
            edge_attr.append(bond_feat)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_attr = torch.stack(edge_attr, dim=0)  # [num_edges, bond_dim]

        return atom_features, edge_index, edge_attr

    def get_atom_features(self, atom, molecule):
        """
        Extracts features for an atom in an RDKit molecule.

        Args:
            atom (Chem.Atom): RDKit atom object.
            molecule (Chem.Mol): RDKit molecule object.

        Returns:
            np.ndarray: Atom feature vector.
        """
        # Basic atom features
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

        # Bridgehead binary
        qa = rdqueries.IsBridgeheadQueryAtom()
        bridge_atoms = molecule.GetAtomsMatchingQuery(qa)
        bridgehead = [1 if atom.GetIdx() in bridge_atoms else 0]

        # Basic EWG 
        ewg_atoms = [9, 17, 35, 53, 7, 8, 16]  # F, Cl, Br, I, N, O, S
        sum_ewg_neighbors = [sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() in ewg_atoms)]

        # Smallest ring size
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

    def get_bond_features(self, bond):
        """
        Extracts features for a bond in an RDKit molecule.

        Args:
            bond (Chem.Bond): RDKit bond object.

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
        ])  # → fixed length 5

        stereo = self.one_hot_encode(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ])  # → fixed length 7

        in_ring = [int(bond.IsInRing())]      # → 1
        conjugated = [int(bond.GetIsConjugated())]  # → 1

        features = bond_type + stereo + in_ring + conjugated
        assert len(features) == 14, f"Got bond feature length: {len(features)}"
        return np.array(features, dtype=np.float32)

    @staticmethod
    def one_hot_encode(value, choices):
        """
        One-hot encodes a value given a list of choices.

        Args:
            value: The value to encode.
            choices (list): List of possible choices.

        Returns:
            list: One-hot encoded vector.
        """
        encoding = [0] * (len(choices) + 1)

        if value in choices:
            encoding[choices.index(value)] = 1  # Standard one-hot encoding
        else:
            encoding[-1] = 1  # Assign to last index if value > max(choices)

        return encoding