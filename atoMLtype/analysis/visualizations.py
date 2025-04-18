import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap


def plot_confidence_by_pred_label(pred_record, sort_by='alphabetical', showfliers=False, figsize=(10, 5)):
    """
    Plots confidence scores grouped by predicted label.

    Args:
        pred_record (PredictionRecord): A record of predictions (with confidence and pred_label).
        sort_by (str): How to sort x-axis labels ('frequency', 'alphabetical', or None).
        showfliers (bool): Whether to show outliers in the boxplot.
        figsize (tuple): Size of the plot.
    """
    # Build dataframe from AtomResult entries
    data = [{
        "pred_label": atom.pred_label,
        "confidence": atom.confidence
    } for atom in pred_record.atom_records if atom.confidence is not None]

    df = pd.DataFrame(data)

    # Sort predicted labels
    if sort_by == 'frequency':
        label_order = df['pred_label'].value_counts().index.tolist()
    elif sort_by == 'alphabetical':
        label_order = sorted(df['pred_label'].unique())
    else:
        label_order = None

    # Plot
    plt.figure(figsize=figsize)
    sns.boxplot(x='pred_label', y='confidence', data=df, order=label_order, showfliers=showfliers)
    plt.xticks(rotation=90)
    plt.title("Confidence by Predicted Label")
    plt.xlabel("Predicted Label")
    plt.ylabel("Confidence (Softmax)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def visualize_embeddings(embeddings, labels, method='umap', label_type='True Label', max_points=5000):
    """
    Reduce and visualize atom embeddings with UMAP or t-SNE.

    Args:
        embeddings (np.ndarray): Shape [num_atoms, hidden_dim]
        labels (List[str or int]): Length [num_atoms]
        method (str): 'umap' or 'tsne'
        label_type (str): Label for legend (e.g. 'true_label' or 'pred_label')
        max_points (int): Subsample if too large
    """
    if len(embeddings) > max_points:
        embeddings = embeddings[:max_points]
        labels = labels[:max_points]

    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("husl", len(set(labels)))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette=palette, s=10, alpha=0.7)
    plt.title(f"{method.upper()} projection of atom embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title=label_type, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def visualize_embeddings_mismatches(embeddings, labels, mismatches_mask=None, pred_labels=None,
                         method='umap', label_type='True Label', max_points=5000):
    """
    Reduce and visualize atom embeddings with optional mismatch overlay colored by predicted label.

    Args:
        embeddings (np.ndarray): [num_atoms, hidden_dim]
        labels (List[str]): True labels
        mismatches_mask (List[bool]): Optional mask of misclassified atoms
        pred_labels (List[str]): Optional predicted labels (used for mismatches)
        method (str): 'umap' or 'tsne'
        label_type (str): Label for legend title
        max_points (int): Limit number of plotted points
    """
    if len(embeddings) > max_points:
        embeddings = embeddings[:max_points]
        labels = labels[:max_points]
        if mismatches_mask is not None:
            mismatches_mask = mismatches_mask[:max_points]
        if pred_labels is not None:
            pred_labels = pred_labels[:max_points]

    # Build label-to-index mapping from all true and predicted labels
    all_labels = set(labels)
    if pred_labels is not None:
        all_labels.update(pred_labels)
    unique_labels = sorted(all_labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    palette = sns.color_palette("husl", len(unique_labels))

    numeric_labels = [label_to_index[l] for l in labels]

    # Dimension reduction
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)

    # Plot base layer (all atoms, true label color)
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("husl", len(unique_labels))
    colors = [palette[label_to_index[l]] for l in labels]
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=8, alpha=0.6)

    # Overlay mismatches with pred label color + "X"
    if mismatches_mask is not None and pred_labels is not None:
        mismatches = np.array(mismatches_mask)
        mismatch_coords = reduced[mismatches]
        mismatch_colors = [palette[label_to_index[pred_labels[i]]] for i in range(len(labels)) if mismatches[i]]

        plt.scatter(mismatch_coords[:, 0], mismatch_coords[:, 1], c=mismatch_colors,
                    marker='x', s=40, linewidths=1.5, edgecolors='red', label='Mismatches')

    # Create sorted legend
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=palette[i], label=label)
        for label, i in label_to_index.items()
    ]
    if mismatches_mask is not None:
        handles.append(plt.Line2D([], [], marker='x', linestyle='', color='gray', label='Mismatches'))

    plt.legend(handles=handles, title=label_type, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{method.upper()} projection of atom embeddings colored by {label_type}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()




from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
from typing import List
from atoMLtype.models.ModelOutput import AtomResult
def draw_molecule_with_mismatches(mol: Chem.Mol, atom_predictions: List[AtomResult], image_size=(400, 300)):
    """
    Draws a molecule highlighting mismatched atoms in red.

    Args:
        mol (Chem.Mol): RDKit molecule
        atom_predictions (List[AtomResult]): List of AtomResult for this molecule
        image_size (tuple): Output image size (width, height)
    
    Returns:
        PIL.Image: Rendered molecule image
    """
    highlight_atoms = []
    atom_colors = {}

    for ap in atom_predictions:
        if ap.true_label != ap.pred_label:
            highlight_atoms.append(ap.atom_idx_in_mol)
            atom_colors[ap.atom_idx_in_mol] = (1.0, 0.0, 0.0)  # red for mismatches

    drawer = rdMolDraw2D.MolDraw2DCairo(image_size[0], image_size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    return Image.open(io.BytesIO(png_data))