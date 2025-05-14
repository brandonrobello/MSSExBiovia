University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: May 12th 2025  
Last Updated: May 13th 2025  

# Model Interpretability and Evaluation Tools

This module contains a suite of scripts for interpreting, visualizing, and evaluating atom typing models. These tools are primarily designed for post-training analysis of GNN and RF models and support both qualitative insights and quantitative summaries.

---

## Module Contents

| File                   | Description |
|------------------------|-------------|
| `molecule_embeddings.py` | Projects learned graph embeddings (from GNNs) into 2D space using dimensionality reduction techniques (e.g., t-SNE, PCA). |
| `confusionMatrices.py`   | Generates confusion matrices comparing true vs. predicted atom types. Can produce normalized, color-coded visualizations. |
| `discrepancies.py`       | Analyzes systematic misclassifications and highlights model confusion between similar atom types. |
| `heatmaps.py`            | Builds heatmaps from attention scores, prediction frequency, or class distributions. |
| `accuracy_counts.py`     | Aggregates accuracy statistics by atom type, molecule, or cluster and generates comparative visualizations. |

---

## Use Cases

- Identify clusters or label groups with high error rates
- Visualize the structure of learned embeddings
- Interpret model behavior via saliency, attention, or confusion heatmaps
- Report per-label or per-molecule performance for benchmarking

---

## Dependencies

- Python 3.8+
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- RDKit (if molecule visualization is needed)
- UMAP / t-SNE / PCA (used in `molecule_embeddings.py`)

---

## Related Modules

| Module Path               | Purpose |
|----------------------------|---------|
| `models/ModelOutput.py`    | Generates prediction data that these scripts consume |
| `utils/metrics.py`         | Computes supporting metrics used in visualization |
| `notebooks/`               | Demonstrates usage of these scripts in end-to-end experiments |

---

This module is essential for model diagnostics, performance reporting, and generating publication-ready visualizations.
