University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025  

# Random Forest Model for Atom Type Classification

This module implements a classical machine learning pipeline for atom type prediction using a Random Forest classifier. It is designed as an interpretable and lightweight alternative to deep learning models, offering fast training and simple diagnostics using engineered atomic descriptors.

---

## Module Contents

| File              | Description |
|-------------------|-------------|
| `RFfeaturizer.py` | Converts RDKit molecules into atom-level features using chemical and spatial descriptors. |
| `RFmodel.py`      | Defines a modular Random Forest classifier with training, evaluation, prediction, and serialization capabilities. |

---

## Feature Engineering (`RFfeaturizer.py`)

The `AtomFeaturizer_RF` class extracts per-atom features from RDKit `Mol` objects and returns a structured `pandas.DataFrame`.

### Extracted Features Include:

#### Local Atomic Properties:
- Atomic number
- Formal charge
- Aromaticity
- Hybridization
- Degree
- Number of hydrogens (explicit + implicit)

#### Spatial Descriptors:
- Distance to molecular center of mass
- Minimum distance to any neighboring atom
- Mean distance to 3 nearest atoms

### Options:
- Feature scaling using `StandardScaler` (optional)

---

## Model Description (`RFmodel.py`)

The `RandomForestModel` class encapsulates:
- `RandomForestClassifier` (from scikit-learn)
- `LabelEncoder` for atom-type strings

### Features:
- `train(X, y)`: Encodes labels, fits the model
- `predict(X)`: Returns predicted labels (decoded)
- `predict_proba(X)`: Returns class probability distributions
- `evaluate(X, y)`: Computes accuracy and prints a confusion matrix
- `save(model_path)`: Pickles model and encoder to disk
- `load(model_path)`: Loads previously trained model and label encoder
- `get_feature_importance()`: Returns a dictionary of ranked feature importances

---

## Input Requirements

- Molecule input: RDKit `Mol` objects with valid 3D conformers
- Atom labels: List of atom-type strings (e.g., `"C.3"`, `"N.ar"`)
- Features and labels: `pandas.DataFrame` or NumPy arrays

---

## Dependencies

- Python 3.8+
- RDKit
- NumPy
- Pandas
- scikit-learn

---

## Related Modules

| Module Path             | Purpose |
|--------------------------|---------|
| `atoMLtype/utils/`       | Provides shared dataset loaders and logging tools |
| `atoMLtype/models/GNN/`  | Alternative GNN-based modeling pipeline |
| `data/`                  | Stores `.sdf` and `.json` inputs used in training |

---

This Random Forest implementation provides a simple and interpretable baseline for atom typing performance benchmarking, feature ablation studies, and early-stage model development.
