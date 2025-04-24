import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Union, DefaultDict
from sklearn.metrics import accuracy_score

from atoMLtype.models.ModelOutput import PredictionRecord, AtomResult
from atoMLtype.analysis.accuracy_counts import extract_element

### DEFINE GLOBAL DEFAULT ELEMENTS
DEFAULT_ELEMENTS = {"c", "h", "o", "n", "s", "p", "other"}


def _compute_heatmap_matrix(pred_record: PredictionRecord) -> pd.DataFrame:
    """
    Computes a normalized heatmap matrix (confusion matrix) from a PredictionRecord.
    Each cell [i,j] shows the percentage of times true label i was predicted as j.

    Args:
        pred_record (PredictionRecord): The prediction record containing true and predicted labels.

    Returns:
        pd.DataFrame: Normalized confusion matrix (%), rows = true labels, cols = predicted labels.
    """
    y_true = pred_record.true_labels
    y_pred = pred_record.pred_labels

    # Ensure valid input
    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch in length of true and predicted labels.")

    # Get all unique atom types across both true and predicted labels
    all_atom_types = sorted(set(y_true) | set(y_pred))

    # Initialize raw count matrix
    heatmap_matrix = pd.DataFrame(0, index=all_atom_types, columns=all_atom_types, dtype=float)

    # Populate counts
    for true_label, pred_label in zip(y_true, y_pred):
        heatmap_matrix.loc[true_label, pred_label] += 1

    # Normalize by row (i.e., true label counts)
    row_sums = heatmap_matrix.sum(axis=1).replace(0, 1)  # Avoid division by zero
    normalized_matrix = heatmap_matrix.div(row_sums, axis=0) * 100

    return normalized_matrix

def _plot_heatmap(matrix, title, xlabel="Reference Types", ylabel="Predicted Types", cmap="Reds", figsize=(12, 10), annot=False):
    """
    Plots a heatmap for a given Heatmap matrix.

    Args:
        matrix (pd.DataFrame): Heatmap matrix data.
        title (str): Title of the heatmap.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        cmap (str): Color map.
        figsize (tuple): Size of the figure.

    Returns:
        None (Displays the heatmap).
    """
    if matrix.empty or matrix.sum().sum() == 0:
        print(f"Skipping empty heatmap: {title}")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix.T, annot=annot, fmt=".1f", cmap=cmap,
        linewidths=0.5, square=True, cbar=True, cbar_kws={"label": "Percentage"},
        xticklabels=matrix.index, yticklabels=matrix.columns,
        annot_kws={"size": 8}
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.show()

def plot_full_heatmap(pred_record: PredictionRecord, annot: bool = False) -> None:
    """
    Generates and displays a full normalized heatmap of all atom type predictions.

    This heatmap shows the percentage of times each true atom type was predicted
    as each other atom type. Includes overall misclassification statistics in the title.

    Args:
        pred_record (PredictionRecord): Atom-level prediction record containing true and predicted labels.
        annot (bool): Whether to annotate the heatmap cells with percentage values.

    Returns:
        None. Displays a confusion-style heatmap with row-normalized percentages.
    """
    # Compute normalized confusion matrix from prediction record
    normalized_matrix = _compute_heatmap_matrix(pred_record)

    # Get total number of atoms and number misclassified
    total_atoms = len(pred_record.true_labels)
    total_misclassified = sum(
        t != p for t, p in zip(pred_record.true_labels, pred_record.pred_labels)
    )

    # Compute misclassification rate
    misclassification_rate = (total_misclassified / total_atoms) * 100

    # Format dynamic title with stats
    title = (
        f"Heatmap Matrix for All Atom Types\n"
        f"Total Atoms: {total_atoms} | "
        f"Misclassified: {total_misclassified} ({misclassification_rate:.2f}%)"
    )

    # Plot the heatmap using shared utility
    _plot_heatmap(matrix=normalized_matrix, title=title, annot=annot)


def plot_element_heatmap(pred_record: PredictionRecord, 
                         elements: set = DEFAULT_ELEMENTS, 
                         annot: bool = True) -> None:
    """
    Generates and displays row-normalized heatmaps for each chemical element based 
    on atom type predictions from a PredictionRecord.

    Each heatmap shows the percentage of times atom types within the same element
    were confused for one another. Displays misclassification statistics in the title.

    Args:
        pred_record (PredictionRecord): The atom-level prediction results.
        elements (set): Set of valid chemical element symbols (e.g., {"C", "N", "O", ...}).
        annot (bool): Whether to annotate heatmap cells with percentage values.

    Returns:
        None. Displays one heatmap per element with intra-element confusion.
    """
    # Compute normalized confusion matrix for all atom types
    normalized_matrix = _compute_heatmap_matrix(pred_record)

    # Identify all unique atom types from the prediction record
    all_atom_types = sorted(set(pred_record.true_labels) | set(pred_record.pred_labels))

    for element in elements:
        # Filter atom types that belong to the current element
        element_rows = [label for label in all_atom_types if extract_element(label) == element]

        if not element_rows:
            continue  # Skip if no atom types for this element

        # Slice matrix to only include atom types of this element
        element_matrix = normalized_matrix.loc[element_rows, element_rows]

        # Count total atoms of this element across the dataset
        total_element_atoms = sum(
            1 for t in pred_record.true_labels if extract_element(t) == element
        )

        # Count misclassified atoms where true label belongs to this element
        misclassified_in_element = sum(
            1 for t, p in zip(pred_record.true_labels, pred_record.pred_labels)
            if extract_element(t) == element and t != p
        )

        # Compute misclassification percentage for this element
        misclassification_rate = (misclassified_in_element / total_element_atoms) * 100

        # Construct title with stats
        title = (
            f"Heatmap Matrix for {element} Atom Types\n"
            f"Total {element} Atoms: {total_element_atoms} | "
            f"Misclassified: {misclassified_in_element} ({misclassification_rate:.2f}%)"
        )

        # Plot heatmap for this element
        _plot_heatmap(element_matrix, title, annot=annot)


def plot_cross_element_heatmap(pred_record: PredictionRecord, elements=DEFAULT_ELEMENTS, annot=True, figsize=(10, 8)):
    """
    Generates a heatmap showing only the cross-element misclassified atom types from a PredictionRecord.
    Only cases where the predicted element differs from the true element are shown.

    Args:
        pred_record (PredictionRecord): Prediction record object.
        elements (set, optional): Set of valid elements (for filtering/mapping). Default includes all seen.
        annot (bool): Whether to annotate the heatmap.
        figsize (tuple): Size of the plot.

    Returns:
        None (Displays heatmap).
    """
    y_true = pred_record.true_labels
    y_pred = pred_record.pred_labels

    # Initialize heatmap matrix
    all_labels = sorted(set(y_true + y_pred))
    matrix = pd.DataFrame(0, index=all_labels, columns=all_labels, dtype=float)

    total_misclassified = 0
    total_atoms = len(y_true)

    for t, p in zip(y_true, y_pred):
        if extract_element(t, elements) != extract_element(p, elements):
            matrix.loc[t, p] += 1
            total_misclassified += 1

    # Normalize by row (true label count)
    row_totals = Counter(y_true)
    for label in matrix.index:
        total = row_totals[label]
        if total > 0:
            matrix.loc[label] = matrix.loc[label] / total * 100

    # Filter non-zero rows/columns
    matrix = matrix.loc[(matrix.sum(axis=1) > 0), (matrix.sum(axis=0) > 0)]

    # Report and Plot
    misclassification_rate = total_misclassified / total_atoms * 100
    title = (f"Cross-element Misclassification Heatmap\n"
             f"Total Atoms: {total_atoms} | Misclassified: {total_misclassified} "
             f"({misclassification_rate:.2f}%)")

    if matrix.empty:
        print("No cross-element misclassifications found.")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap="OrRd", annot=annot, fmt=".1f", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Atom Type")
    plt.ylabel("True Atom Type")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()