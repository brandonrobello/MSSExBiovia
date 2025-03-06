import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from atoMLtype.utils.featurizer import atoMLtype_featurizer

class GraphFeaturizer(atoMLtype_featurizer):
    """
    Extracts both per-atom (node) and per-bond (edge) features from RDKit molecules.
    """

    def __init__(self):
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
        
        Returns:
            - atom_features: (num_atoms, feature_dim)
            - edge_indices: (2, num_edges)
            - bond_features: (num_edges, bond_feature_dim)
        """
        atom_features = [self.get_atom_features(atom) for atom in molecule.GetAtoms()]
        bond_features = []
        edge_indices = []

        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)

            # Add only (i, j), NOT (j, i)
            edge_indices.append([i, j])
            bond_features.append(bond_feat)

        return np.array(atom_features, dtype=np.float32), \
            np.array(edge_indices, dtype=np.int64).T, \
            np.array(bond_features, dtype=np.float32)


    def get_atom_features(self, atom):
        """Extract features for an atom in an RDKit molecule."""
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
        
        return np.array(atom_type + degree + formal_charge + chiral_center + chirality_type + 
                        num_h + atomic_mass + aromaticity + radical_electrons + hybridization, dtype=np.float32)

    def get_bond_features(self, bond):
        """Extract features for a bond in an RDKit molecule."""
        if bond is None:
            return np.zeros(14, dtype=np.float32)  # Zero vector for non-existent bonds
        
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
        """One-hot encode a value given a list of choices."""
        encoding = [1 if value == choice else 0 for choice in choices]
        return encoding