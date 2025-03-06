import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler

class MolecularFeaturizer(ABC):
    """
    Abstract base class for molecular featurization.
    Ensures that all featurizers implement a `featurize()` method.
    """
    @abstractmethod
    def featurize(self, molecule):
        pass

class AtomFeaturizer_RF(MolecularFeaturizer):
    """
    Extracts per-atom features from RDKit molecules.
    """

    def featurize(self, molecule, apply_scaling=False, include_neighborhood=True) -> pd.DataFrame:
        """Generates per-atom feature vectors from an RDKit molecule.
        
        Args:
            molecule (rdkit.Chem.Mol): RDKit molecule object.
            apply_scaling (bool): If True, applies StandardScaler normalization.
            include_neighborhood (bool): If False, excludes features related to atom neighborhood.

        Returns:
            pd.DataFrame: Feature matrix (scaled if apply_scaling=True).
        """
        features = []

        hybridization_map = {
            Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0, 0],
            Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0, 0],
            Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1, 0],
        }

        # Get a conformer of molecule 
        # Get the center of the conformer as the average of the atom positions
        conf = molecule.GetConformer(0)
        atom_positions = np.array([list(conf.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())])
        center_of_mass = np.mean(atom_positions, axis=0)
        
        # Get positions of all atoms
        pos = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in molecule.GetAtoms()])
        # Use vectorization to get the euclidian distance of all atoms pairwise
        # distances[i, j] is the Euclidean distance between atoms i and j.
        distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

        for atom in molecule.GetAtoms():
            # Get atom index
            atom_idx = atom.GetIdx()

            # One-hot encode hybridization
            hybridization = atom.GetHybridization()
            hybridization_encoded = hybridization_map.get(hybridization, [0, 0, 0, 0, 0, 1])

            # Compute neighborhood-based features
            neighborhood_features = []
            if include_neighborhood:
                # Get relative distance of atom to center of mass
                dist_to_com = np.linalg.norm(pos[atom_idx] - center_of_mass)
                # Compute nearest neighbor distances (excluding self-distance)
                sorted_distances = np.sort(distances[atom_idx, :])  # Sort distances
                non_self_distances = sorted_distances[sorted_distances > 0]  # Exclude 0 (self-distance)

                # Handle case where atom has fewer than 3 neighbors
                if len(non_self_distances) >= 3:
                    mean_k3_dist = np.mean(non_self_distances[:3])
                elif len(non_self_distances) > 0:
                    mean_k3_dist = np.mean(non_self_distances)  # If fewer than 3, take mean of available
                else:
                    mean_k3_dist = 0  # If no neighbors exist (isolated atom), set to 0

                min_dist = non_self_distances[0] if len(non_self_distances) > 0 else 0  # Handle isolated atoms
                
                neighborhood_features = [dist_to_com, mean_k3_dist, min_dist]


            base_features = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetFormalCharge(),  # Formal charge
                atom.GetTotalNumHs(),  # Number of implicit hydrogens
                int(atom.GetIsAromatic()),  # Aromaticity (binary)
                atom.GetDegree(),  # Degree (number of bonded neighbors)
            ]

            if include_neighborhood:
                neighborhood_features = [dist_to_com,  # Distance to center of mass
                                        mean_k3_dist,  # Mean distance to 3 nearest neighbors
                                        min_dist] # Closest neighbor distance
            else: neighborhood_features = []

            features.append(base_features + neighborhood_features + hybridization_encoded)


        # Define column names dynamically based on whether neighborhood features are included
        base_columns = ["AtomicNum", "FormalCharge", "NumHs", "Aromaticity", "Degree"]
        neighborhood_columns = ["DistToCenterOfMass", "MeanDistK3", "MinDist"] if include_neighborhood else []
        hybridization_columns = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OtherHybrid"]
        
        column_names = base_columns + neighborhood_columns + hybridization_columns
        
        # Build pandas df with features and columns
        df = pd.DataFrame(features, columns=column_names)

        # Apply scaling if enabled
        if apply_scaling:
            scaler = StandardScaler()
            df.iloc[:, :] = scaler.fit_transform(df)  # Normalize all columns

        return df

class MoleculeFeaturizer(MolecularFeaturizer):
    """
    Extracts molecular-level features using various descriptor methods (ECFP, MACCS, etc.).
    """

    def __init__(self, method="ECFP4"):
        """
        Args:
            method (str): Feature extraction method (e.g., "ECFP4", "MACCS").
        """
        self.method = method

    def featurize(self, molecule) -> np.array:
        """Generates molecular features based on the selected method."""
        if self.method == "ECFP4":
            return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))
        elif self.method == "MACCS":
            return np.array(rdMolDescriptors.GetMACCSKeysFingerprint(molecule))
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")

class GraphFeaturizer(MolecularFeaturizer):
    """
    Placeholder class for graph-based featurization.
    This can later be extended with PyG (PyTorch Geometric) representations.
    """
    def featurize(self, molecule):
        raise NotImplementedError("Graph featurization is not yet implemented.")
