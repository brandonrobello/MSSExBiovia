University of California Berkeley x BIOVIA  
MSSE Capstone Project - Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: Wednesday April 9th 2025  
Last Updated: Wednesday April 9th 2025  

# Random Forest Module for Atom Type Classification

This module implements a Random Forest-based atom classification workflow. It provides a lightweight and interpretable machine learning alternative to GNN models by leveraging molecular featurization of individual atoms and their environments.  

## Module Contents

| File | Description |
|------|-------------|
| `RFfeaturizer.py` | Extracts per-atom feature vectors from RDKit molecules using local and spatial descriptors. |
| `RFmodel.py` | Defines a modular Random Forest classifier with training, prediction, and serialization utilities. |

## Feature Engineering

The `AtomFeaturizer_RF` class converts an RDKit molecule into a pandas `DataFrame` containing one row per atom. Features include:

- Atomic number, formal charge, number of hydrogens
- Aromaticity, hybridization, and degree
- Spatial neighborhood descriptors:
  - Distance to molecular center of mass
  - Mean distance to 3 nearest atoms
  - Closest neighbor distance

Features can optionally be scaled using `StandardScaler`.

## Model Description

The `RandomForestModel` class extends a custom `BaseModel` and includes:

- Automatic train/test splitting
- Label encoding for atom type classes
- Pickle-based model saving and loading
- Built-in error handling and training safeguards
- Returns decoded predictions as string labels

The classifier is built using `sklearn.ensemble.RandomForestClassifier` with a default of 100 trees.

### Input Requirements
- Molecule input must be an RDKit Mol object with 3D conformers.

- Labels are expected as a list of atom-type strings (e.g., "C.3", "N.ar").

- Featurization and model training are compatible with pandas or NumPy formats.

### Dependencies
- Python 3.x

- RDKit

- scikit-learn

- NumPy

- Pandas

### Related Modules
- atoMLtype/GNN: GNN-based atom typing framework

- atoMLtype/utils: Core dataset utilities and shared featurization tools

- data/: SDF molecules and JSON labels used in training

