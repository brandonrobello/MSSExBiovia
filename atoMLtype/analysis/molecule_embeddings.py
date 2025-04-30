import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap

from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import os
from typing import List, Optional

from atoMLtype.models.ModelOutput import PredictionRecord, AtomResult

def visualize_prediction_embeddings(
    pred_record: PredictionRecord,
    key: str = "clf_embeddings",
    method: str = 'umap',
    color_by: str = 'true_label',
    highlight_mismatches: bool = True,
    mismatch_style: str = 'red',
    max_points: int = None,
    show_legend: bool = True,
    figsize: tuple = (10, 6),
    title: str = None
):
    """
    Visualizes atom embeddings from a PredictionRecord using UMAP or TSNE.

    Args:
        pred_record (PredictionRecord): Contains embeddings, labels, and mismatch info.
        key (str): Which embedding in `.analysis` to visualize.
        method (str): 'umap' or 'tsne'.
        color_by (str): 'true_label' or 'pred_label' for coloring base points.
        highlight_mismatches (bool): Whether to overlay mismatched atoms.
        mismatch_style (str): 'pred' (color by predicted label) or 'red'.
        max_points (int): Subsample to this number of atoms.
        show_legend (bool): Whether to display the legend.
        figsize (tuple): Size of the plot.
        title (str): Title for plot.
    """
    all_atoms = pred_record.atom_records

    # Filter out missing embeddings
    atoms_with_key = [a for a in all_atoms if key in a.analysis]
    if not atoms_with_key:
        raise ValueError(f"No atoms have analysis key '{key}'")

    if max_points and len(atoms_with_key) > max_points:
        atoms_with_key = atoms_with_key[:max_points]

    embeddings = np.stack([a.analysis[key] for a in atoms_with_key])
    labels = [getattr(a, color_by) for a in atoms_with_key]
    pred_labels = [a.pred_label for a in atoms_with_key]
    mismatches_mask = [a.true_label != a.pred_label for a in atoms_with_key]

    # Encode labels
    all_labels = sorted(set(labels + pred_labels))
    label_to_index = {l: i for i, l in enumerate(all_labels)}
    palette = sns.color_palette("husl", len(all_labels))
    label_colors = [palette[label_to_index[l]] for l in labels]

    # Reduce dimension
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42) if method == 'umap' \
        else TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=label_colors, s=10, alpha=0.6)

    if highlight_mismatches:
        mismatches = np.array(mismatches_mask)
        mismatch_coords = reduced[mismatches]

        if mismatch_style == 'pred':
            mismatch_colors = [palette[label_to_index[pred_labels[i]]]
                               for i in range(len(pred_labels)) if mismatches[i]]
        else:
            mismatch_colors = ['red'] * np.sum(mismatches)

        plt.scatter(mismatch_coords[:, 0], mismatch_coords[:, 1], c=mismatch_colors,
                    marker='x', s=50, linewidths=1.5, label='Mismatches')

    # Build legend
    if show_legend:
        handles = [
            plt.Line2D([], [], marker='o', linestyle='', color=palette[i], label=l)
            for l, i in label_to_index.items()
        ]
        if highlight_mismatches:
            mismatch_handle_color = 'gray' if mismatch_style == 'pred' else 'red'
            handles.append(plt.Line2D([], [], marker='x', linestyle='', color=mismatch_handle_color, label='Mismatches'))

        plt.legend(handles=handles, title=color_by.replace("_", " ").title(), 
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if title:
        plt.title(title)
    else:
        plt.title(f"{method.upper()} projection of atom embeddings colored by {color_by}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

def draw_molecule_with_mismatches_labeled(
    mol: Chem.Mol,
    atom_predictions: List[AtomResult],
    image_size=(400, 300),
    save_dir: Optional[str] = None
) -> Image.Image:
    """
    Draws a molecule highlighting mismatched atoms in red and annotating them with 
    predicted / true atom types using RDKit's atomNote system.

    Args:
        mol (Chem.Mol): RDKit molecule
        atom_predictions (List[AtomResult]): List of AtomResult for this molecule
        image_size (tuple): Output image size (width, height)
        save_dir (str, optional): Optional directory to save the image

    Returns:
        PIL.Image: Rendered molecule image with highlights and notes
    """
    mol = Chem.Mol(mol)  # Copy to avoid mutating original
    rdDepictor.Compute2DCoords(mol)

    # Highlight mismatched atoms in red
    highlight_atoms = []
    atom_colors = {}
    for ap in atom_predictions:
        if ap.true_label != ap.pred_label:
            idx = ap.atom_idx_in_mol
            highlight_atoms.append(idx)
            atom_colors[idx] = (0.0, 1.0, 0.0)  # red
            note = f"{ap.pred_label} != {ap.true_label}"
            mol.GetAtomWithIdx(idx).SetProp("atomNote", note)

    # Set up the drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(*image_size)
    opts = drawer.drawOptions()
    opts.annotationFontScale = 0.8
    opts.addStereoAnnotation = True
    opts.addAtomIndices = False

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[], 
        legend=mol.GetProp("_Name") 
    )
    drawer.FinishDrawing()

    # Convert to PIL image
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # Optionally save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{mol.GetProp('_Name') }_misclassATs_annotated.png")
        img.save(path)

    return img