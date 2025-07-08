University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025  

# Machine Learning for Atom Type Assignment in Force Fields

This repository contains the full implementation of a collaborative capstone project between the UC Berkeley MSSE program and Dassault Systèmes BIOVIA. The goal is to automate atom type assignment in molecular force fields using both interpretable machine learning and advanced graph neural network (GNN) architectures.

---

## Project Objectives

- Develop scalable, high-accuracy models for atom type classification.
- Compare traditional models (Random Forest) with GNNs in terms of performance and interpretability.
- Design a reproducible, modular software framework for scientific modeling.

---

## Repository Structure
## 📁 Full Project Structure

The full folder structure of this repository is shown below, with comments explaining the purpose of each file and directory:

```text
MSSExBiovia/
├── README.md                         # Project overview, installation, usage, and structure
├── environment.yml                   # Conda environment with all dependencies
├── Tutorial_quickstart.ipynb        # Jupyter tutorial: loading data and training a model
├── Tutorial_saveModel.ipynb         # Jupyter tutorial: saving and loading trained models
├── Tutorial_analysis.ipynb          # Jupyter tutorial: using analysis tools
├── atoMLtype/                       # Core Python package
│   ├── __init__.py
│   ├── README.md                    # Package-level summary
│   ├── analysis/                    # Post-training interpretability and evaluation tools
│   │   ├── accuracy_counts.py       # Accuracy summaries by label or molecule
│   │   ├── confusionMatrices.py     # Confusion matrix generation and plotting
│   │   ├── discrepancies.py         # Misclassification tracking
│   │   ├── heatmaps.py              # Heatmaps for attention or prediction scores
│   │   └── molecule_embeddings.py   # 2D projection of GNN embeddings
│   ├── datasets/                   # Dataset loaders and featurizers
│   │   ├── BaseDataset.py           # Abstract class for dataset management
│   │   ├── SDFdataset.py            # Loader for RF-compatible datasets
│   │   ├── GNNdataset.py            # Loader for GNN-compatible graph datasets
│   │   ├── BaseFeaturizer.py        # Abstract base class for feature extractors
│   │   ├── GNNfeaturizer.py         # Atom/bond featurization for GNNs
│   │   └── README.md
│   ├── models/                     # Modeling architectures and training engines
│   │   ├── BaseGNNModel.py          # Shared GNN base class
│   │   ├── ModelEncoder.py          # Label encoder/decoder utilities
│   │   ├── ModelEngine.py           # Model lifecycle manager (train/eval/predict)
│   │   ├── ModelOutput.py           # Output formatting, storage, and summaries
│   │   ├── ModelTrainer.py          # Unified training logic
│   │   ├── GNN/                     # Graph neural network models
│   │   │   ├── BaselineGNNs.py      # GCN, GAT, and GIN implementations
│   │   │   ├── DMPNNmodel.py        # Attention-weighted directed MPNN
│   │   │   ├── MultilayerGNNs_TBD.py# WIP: Extended GNN variants
│   │   │   └── README.md
│   │   ├── RF/                      # Random Forest model and featurizer
│   │   │   ├── RFmodel.py           # Random Forest training and prediction
│   │   │   ├── RFfeaturizer.py      # Atom-level feature engineering for RF
│   │   │   └── README.md
│   │   └── README.md
│   └── utils/                      # Shared logging and helper functions
│       ├── logging_utils.py         # Logger setup for scripts and training
│       └── README.md
├── tests/                          # Test suite and Docker test runner
│   ├── Dockerfile                   # Container for isolated test runs
│   ├── build_and_test.sh           # Test execution script
│   ├── requirements.txt            # Test dependencies
│   ├── test_functionality.py       # Unit and integration tests
│   ├── test_data/                  # Sample molecules and labels for testing
│   │   ├── zinc_subsampled.sdf
│   │   └── atomLabels_gaff2_subsampled.json
│   └── README.MD
```
---

## Setup Instructions

### 1. Install Dependencies

Use the included `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate atomtyping
```

### 2. Data Format

Place your .sdf and .json files under a data/ directory. Ensure atom ordering matches between formats.
The data used in this repo can be found [HERE](https://zenodo.org/records/15369157?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhMzg5MjIwLTdhZWYtNDNmYS05ODk5LWI0MTRmOGRhMDA0YiIsImRhdGEiOnt9LCJyYW5kb20iOiIyOWVjYzE5ZjgyMzMzZDVmZmRhZDU1NzMyYWZiZDJmNSJ9.M7mfNuaggBtV1sJGNTf6U4XRc4xWVO-UB6cXSt7vLxzlvAVKhWtmoHG_7FvSTFi2C-oPXnzNAxvsrFMjbYp_TA)

### 3. Train Models

GNN Example:
```bash
from atoMLtype.datasets.GNNdataset import GNNdataset
from atoMLtype.models.GNN.BaselineGNNs import BaselineGAT
from atoMLtype.models.ModelTrainer import GNNTrainer

dataset = GNNdataset("data/zinc.sdf", "data/atom_labels.json")
model = BaselineGAT(num_node_features=dataset[0].x.shape[1], num_atom_types=len(dataset.label_encoder.classes_))
trainer = GNNTrainer(model=model, dataset=dataset, epochs=20)
trainer.train()
```

RF Example:
```bash
from atoMLtype.datasets.SDFdataset import SDFdataset
from atoMLtype.models.RF.RFmodel import RandomForestModel

dataset = SDFdataset("data/zinc.sdf", "data/atom_labels.json")
rf_model = RandomForestModel()
rf_model.train(dataset.features, dataset.labels)
```

## Evaluation Tools

Use the analysis/ scripts to:
- Generate confusion matrices
- Visualize embedding projections (PCA/t-SNE/UMAP)
- Analyze attention weights and accuracy by class
- Quantify discrepancies in label prediction

##  Testing
Run the full test suite:

```bash
cd tests
chmod +x build_and_test.sh
./build_and_test.sh
```
## Dependencies
- Python 3.8+
- RDKit
- PyTorch & PyTorch Geometric
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn
- UMAP-learn (optional for molecule_embeddings.py)

This repository provides a reproducible, modular, and research-grade platform for exploring atom typing through machine learning and GNNs.

### Contact
For questions or contributions, please reach out to the project authors or supervising faculty through the UC Berkeley MSSE program.

