University of California Berkeley x BIOVIA  
MSSE Capstone Project - Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: Wednesday April 9th 2025  
Last Updated: Wednesday April 9th 2025  

# GNN Module for Atom Type Classification

This module implements a graph neural network (GNN) pipeline for atom typing using molecular structure data. It processes SDF-formatted molecules into graph representations, extracts chemically meaningful features, and applies GNN-based models to classify atom types at the node level.

## Module Contents

| File | Description |
|------|-------------|
| `GNNdataset.py` | Loads molecular structures and labels, and converts them into PyTorch Geometric graph datasets. Includes support for 1-hop subgraph generation. |
| `GNNfeaturizer.py` | Extracts atom and bond features using RDKit, supporting both standard and message-passing (D-MPNN-style) featurization. |
| `GNNmodel.py` | Defines multiple GNN architectures including GCN, GAT, GIN, MPNN, and attention-based models. Includes a flexible training and evaluation framework. |

## Functionality Overview

### Dataset and Featurization

- Molecules are loaded using a custom `SDFdataset` class and labeled with atom typing annotations.
- Features are generated using `GraphFeaturizer` or `GraphFeaturizer_DMPNN`.
- Each molecule is represented as a PyTorch Geometric `Data` object with:
  - `x`: atom (node) features
  - `edge_index`: connectivity (bonds)
  - `edge_attr`: bond features
  - `y`: per-atom classification labels

### Subgraph Construction

- `GNNdataset_subgraphs` supports extraction of per-atom 1-hop subgraphs.
- These subgraphs enable training models that focus on local chemical environments.

## Model Architectures

The module includes the following GNN variants:

| Model | Type | Description |
|-------|------|-------------|
| `BaselineGCN`, `GCN_4Layer` | GCN | Graph convolutional networks for node-level prediction |
| `BaselineGAT`, `GAT_4L` | GAT | Attention-based networks that weigh neighbor contributions |
| `BaselineGIN` | GIN | Graph isomorphism network using MLP-based aggregation |
| `MPNN_4L` | MPNN | Message passing neural network with standard aggregation |
| `Att_AtomBondMPNN` | Attention MPNN | Atom-bond message passing with attention-based edge weighting |

## Training and Evaluation

- The `GNNTrainer` class manages k-fold cross-validation, training, validation, and loss visualization.
- Supports classification and regression tasks.
- Evaluation includes accuracy, F1-score, and MSE (based on task).

## Input Requirements

- `.sdf` file with molecular structures
- `.json` file containing atom-level labels

Set `collapse=True` to reduce fine-grained atom types into broader categories during label encoding.

### Dependencies
- Python 3.x

- RDKit

- PyTorch

- PyTorch Geometric

- NumPy, Pandas, scikit-learn

- Matplotlib

### Related Modules
- atoMLtype/utils: Shared utilities for loading and labeling molecular data

- atoMLtype/RF: Random Forest-based atom typing implementation

- data/: Contains molecular datasets and reference label files

## Example Usage

```python
from atoMLtype.GNN.GNNdataset import GNNdataset
from atoMLtype.GNN.GNNmodel import BaselineGAT, GNNTrainer

dataset = GNNdataset("data/parm_at_Frosst/zinc.sdf", "data/antechamber/atomLabels_gaff2.json")

model = BaselineGAT(
    num_node_features=dataset[0].x.shape[1],
    num_atom_types=len(dataset.label_encoder.classes_)
)

trainer = GNNTrainer(model=model, dataset=dataset, epochs=20)
trainer.train()


