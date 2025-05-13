University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: May 2025  
Last Updated: May 2025  

# Dataset Loaders and Featurizers

This module provides standardized classes for loading, parsing, and featurizing molecular data for use in both Random Forest (RF) and Graph Neural Network (GNN) pipelines. All classes are modular and support `.sdf` molecular structures and `.json` atom-type labels.

---

## Module Contents

| File               | Description |
|--------------------|-------------|
| `BaseDataset.py`   | Abstract base class for dataset loaders. Defines shared logic for parsing, validation, and label alignment. |
| `SDFdataset.py`    | Loads RDKit molecules from `.sdf` files and atom-type labels from `.json` files. Prepares input for RF models. |
| `BaseFeaturizer.py`| Abstract base class for molecular featurizers. Ensures consistent interface across featurizer types. |
| `GNNfeaturizer.py` | Builds atom and bond features for GNN input using RDKit. Supports both standard and D-MPNN-style features. |
| `GNNdataset.py`    | Converts `.sdf` + `.json` data into PyTorch Geometric graph datasets. Supports subgraph construction and label collapsing. |

---

## Supported Input Formats

- **Molecules:** 3D `.sdf` file with RDKit-compatible molecular structures.
- **Atom Labels:** `.json` file structured as:
  ```json
  {
    "Name": "mol1",
    "Atom_types": ["C.3", "N.ar", "H", ...]
  }

Both formats must maintain atom ordering consistency for label alignment.

## Dataset Loader Overview
Loader Class	            Output Format	                                Used By
SDFdataset	                Pandas DataFrame (atom rows)	                RF pipeline
GNNdataset	                List of torch_geometric.data.Data objects	    GNN pipeline

## Featurizer Overview
Featurizer Class	                    Description
GraphFeaturizer	                        Atom/bond feature vectors for standard GNN models
GraphFeaturizer_DMPNN	                Directed message-passing features for D-MPNN models

Each featurizer returns tensors or dictionaries compatible with PyTorch Geometric.

## Dependencies
- RDKit

- NumPy, Pandas

- PyTorch

- PyTorch Geometric

- scikit-learn (for scaling)

## Integration
These dataset and featurizer classes are used by:

Module:	        Purpose
- models/RF/:	Random Forest pipeline uses SDFdataset and AtomFeaturizer_RF
- models/GNN/:	GNN pipeline uses GNNdataset and GNNfeaturizer
- analysis/:	Visualizations and confusion analysis rely on label-matched datasets

This module serves as the foundation of the ML pipeline, ensuring clean, reliable, and reproducible dataset construction for all modeling experiments.