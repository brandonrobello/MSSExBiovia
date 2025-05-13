University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: May 12th 2025  
Last Updated: May 12th 2025  

# Modeling Infrastructure

This directory contains the full modeling infrastructure used for atom type prediction. It includes reusable components for model training, prediction, evaluation, and encoding. These are used across both classical machine learning (Random Forest) and deep learning (GNN) pipelines, ensuring consistency, modularity, and scalability.

---

## Subdirectories

| Folder | Description |
|--------|-------------|
| `RF/`  | Random Forest-based atom classification using hand-engineered features. |
| `GNN/` | Graph neural networks using PyTorch Geometric for atom-level classification. |

---

## Core Modules

| File              | Purpose |
|-------------------|---------|
| `ModelOutput.py`  | Handles formatting, storing, and exporting model predictions. |
| `ModelTrainer.py` | Unified training wrapper for both RF and GNN models. Manages splits, callbacks, and metrics. |
| `ModelEngine.py`  | Centralizes model lifecycle (training, evaluation, prediction). Compatible with all supported models. |
| `ModelEncoder.py` | Encodes and decodes atom-type labels using `LabelEncoder`. Supports label normalization and remapping. |
| `BaseGNNModel.py` | Abstract base class for all GNN architectures. Provides shared GNN behaviors (forward pass, dropout, loss). |

---

## Design Philosophy

- **Modularity**: All components can be reused and extended independently.
- **Compatibility**: Engineered to work seamlessly with both RF and GNN workflows.
- **Maintainability**: Abstracts common logic across pipelines to avoid code duplication.
- **Reproducibility**: Encoders, outputs, and training procedures are version-controlled and serializable.

---

## Related Modules

| Module Path              | Purpose |
|---------------------------|---------|
| `datasets/`               | Supplies `.sdf` molecules and `.json` labels for training |
| `utils/`                  | Shared utilities for logging, metrics, and featurization |
| `analysis/`               | Visualization tools for interpretability and model inspection |
| `tests/`                  | End-to-end functionality tests for training, inference, and prediction accuracy |

---

This directory serves as the architectural backbone for the ML pipeline and enforces a consistent interface across modeling paradigms.
