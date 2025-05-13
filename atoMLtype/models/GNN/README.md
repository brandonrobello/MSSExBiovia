University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025  

# GNN Models for Atom Type Classification

This module implements graph neural network (GNN) architectures for atom typing based on molecular graph representations. All models are built using PyTorch Geometric and support flexible configuration, training, and inference for per-atom classification tasks.

---

## Module Contents

| File                  | Description |
|-----------------------|-------------|
| `BaselineGNNs.py`     | Defines baseline GNN models: GCN, GAT, GIN with 2- and 4-layer variants. |
| `DMPNNmodel.py`       | Implements an edge-aware Directed Message Passing Neural Network (D-MPNN) with optional attention. |
| `MultilayerGNNs_TBD.py` | Experimental/placeholder script for future multi-layer GNN variants (work in progress). |

---

## Implemented Architectures

### From `BaselineGNNs.py`

| Class          | Type | Description |
|----------------|------|-------------|
| `BaselineGCN`  | GCN  | Two-layer Graph Convolutional Network |
| `GCN_4Layer`   | GCN  | Four-layer deep GCN |
| `BaselineGAT`  | GAT  | Two-layer Graph Attention Network |
| `GAT_4L`       | GAT  | Four-layer attention-based GNN |
| `BaselineGIN`  | GIN  | Graph Isomorphism Network using MLP-based aggregators |

### From `DMPNNmodel.py`

| Class              | Type     | Description |
|--------------------|----------|-------------|
| `Att_AtomBondMPNN` | D-MPNN   | Edge-aware, bond-directed message passing with optional attention weighting for atom typing |

---

## Model Input

All GNN models operate on `torch_geometric.data.Data` objects with the following fields:

- `x`: Node (atom) features
- `edge_index`: Graph connectivity
- `edge_attr`: Bond (edge) features
- `y`: Atom-type labels (per node)

---

## Integration

These models are called by the training engine defined in:

- `atoMLtype/models/ModelEngine.py`
- `GNNTrainer` class
- Dataset creation in `atoMLtype/datasets/GNNdataset.py`

Each model is compatible with label encoders and evaluation utilities provided in the broader project.

---

## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy, Pandas

---

## Related Modules

| Path                    | Purpose |
|-------------------------|---------|
| `datasets/GNNdataset.py`| Loads `.sdf` and `.json` inputs into GNN-compatible graphs |
| `models/ModelEngine.py` | Unifies training and inference logic |
| `utils/metrics.py`      | Accuracy, F1-score, confusion matrix, and other evaluators |
| `analysis/`             | Interpretability tools including saliency and attention maps |

---

This directory provides the core architectures for deep learning-based atom classification. The models are modular and designed for benchmarking, ablation, and extension to new GNN paradigms.
