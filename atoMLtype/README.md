University of California Berkeley x BIOVIA  
MSSE Capstone Project - Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: Wednesday April 9th 2025  
Last Updated: Wednesday April 9th 2025  

# atoMLtype Package

The `atoMLtype` package provides a modular framework for atom type classification using both classical machine learning and deep learning approaches. It defines a consistent interface for loading molecular data, featurizing atomic environments, and applying classification models.

This package serves as the core engine for Random Forest (RF) and Graph Neural Network (GNN) modeling pipelines used throughout the project.

## Submodules

- `GNN/`: Graph-based models and datasets for atom type classification using PyTorch Geometric.
- `RF/`: Feature-based Random Forest classifier and featurizer for per-atom prediction.
- `utils/`: Shared utilities for molecule loading, label mapping, feature standardization, and metric evaluation.

## Input Requirements

- All submodules expect molecules in RDKit format.
- Atom-level labels should be provided via JSON files linked by molecule name.

## Dependencies

- Python 3.x
- RDKit
- NumPy
- Pandas
- scikit-learn
- PyTorch (for GNN)
- PyTorch Geometric (for GNN)

## Related Modules

- `data/`: Raw and labeled molecular datasets used by this package.
- Root-level notebooks (e.g., `GNNmodel_testing.ipynb`, `atomMLtype_testing.ipynb`) demonstrate how this package is used in practice.
