University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025 

# Utilities Module

This module contains shared utilities used across both Random Forest and GNN-based workflows. It includes tools for dataset preparation, atom label processing, feature standardization, and evaluation metrics. These scripts provide foundational functionality to standardize input data and support consistent training and evaluation across models.

## Module Contents

| File | Description |
|------|-------------|
| `SDFdataset.py` | Loads and validates molecules from SDF files, aligns them with per-atom JSON labels, and filters valid entries. |
| `featurizer.py` | Defines an abstract base class for molecular featurizers used by both GNN and RF modules. |
| `label_utils.py` | Handles atom label collapsing, alternating label recovery, and utilities for structured label management. |
| `metrics.py` | Computes accuracy metrics and generates confusion matrices and heatmaps for atom type prediction models. |

## Input Requirements

- `SDFdataset.py` expects:
  - An SDF file containing molecules with 3D coordinates and valid atom orderings.
  - A JSON file containing atom type annotations structured per molecule (`{"Name": ..., "Atom_types": [...]}`).

- `label_utils.py` and `metrics.py` expect:
  - Flat or nested lists of atom type strings.
  - In some cases, RDKit `Mol` objects for structure-aware operations (e.g., alternating label assignment).

## Dependencies

- Python 3.x
- RDKit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Related Modules

- `atoMLtype/RF`: Uses `featurizer.py`, `SDFdataset.py`, and `label_utils.py` for preprocessing and training.
- `atoMLtype/GNN`: Integrates `SDFdataset.py`, `label_utils.py`, and `metrics.py` into dataset construction, model evaluation, and visualization pipelines.
- `data/`: Source for SDF files and JSON label files processed by these utilities.
