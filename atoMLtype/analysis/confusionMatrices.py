import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Optional, Set, Tuple, Dict
from atoMLtype.models.ModelOutput import PredictionRecord
from atoMLtype.analysis.accuracy_counts import extract_element

DEFAULT_ELEMENTS = {"c", "h", "o", "n", "s", "p", "other"}

def _conf_matrix_from_pairs(
        pairs: List[Tuple[str, str]],
        true_label_counts: Dict[str, int],
        title: str,
        cmap: str,
        annot: bool = True) -> None:    
    """
    Plots a confusion matrix from (true, predicted) atom type pairs with non-zero filtering.
    
    Each cell shows the count of times a predicted label was assigned to a given true label.
    X-axis includes true label with its total abundance as (n=...).

    Args:
        pairs (List[Tuple[str, str]]): List of (true_label, pred_label) pairs.
        true_label_counts (dict): Mapping of true label -> count in dataset, for annotation.
        title (str): Title of the plot.
        cmap (str): Matplotlib colormap name for the heatmap.
        annot (bool): Whether to annotate cells with their values.

    Returns:
        None. Displays the confusion matrix heatmap.
    """
    # Create DataFrame of (true, predicted) values
    df = pd.DataFrame(pairs, columns=["True", "Predicted"])

    # Compute raw confusion matrix (Predicted vs True)
    matrix = pd.crosstab(df["Predicted"], df["True"])

    # Create annotation matrix with blank entries for zeros
    annot_matrix = matrix.astype(str)
    annot_matrix[matrix == 0] = ""

    # Append atom type abundance to column labels
    xtick_labels = [f"{col}\n(n={true_label_counts.get(col, 0)})" for col in matrix.columns]
    matrix.columns = xtick_labels
    annot_matrix.columns = xtick_labels

    # Mask to hide zero cells from rendering
    mask = matrix == 0

    # Plot heatmap
    plt.figure(figsize=(max(12, 0.5 * matrix.shape[1]), max(10, 0.5 * matrix.shape[0])))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=annot_matrix if annot else False,
        fmt="",
        cmap=cmap,
        cbar_kws={"label": "Count"}
    )

    plt.title(title)
    plt.xlabel("True Atom Type")
    plt.ylabel("Predicted Atom Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_full_confusion_matrices(
    pred_record: PredictionRecord,
    elements: Set[str] = DEFAULT_ELEMENTS,
    annot: bool = True
) -> None:
    """
    Generates and displays two confusion matrices using mismatched atom predictions from a PredictionRecord:
        1. Cross-element mismatches: True and predicted labels belong to different elements.
        2. Intra-element mismatches: True and predicted labels belong to the same element.

    Confusion matrices display non-zero misclassification counts and include true label abundances.

    Args:
        pred_record (PredictionRecord): Record of all atom predictions and true labels.
        elements (Set[str]): Set of valid atomic element symbols (used for parsing from atom types).
        annot (bool): Whether to annotate each heatmap cell with the count.

    Returns:
        None. Displays two heatmaps (or prints messages if no mismatches are found).
    """
    y_true = pred_record.true_labels
    y_pred = pred_record.pred_labels

    # Pair up all predicted and true labels
    pairs = np.array(list(zip(y_true, y_pred)))

    # Filter out matching predictions — only consider mismatches
    mismatches = pairs[pairs[:, 0] != pairs[:, 1]]

    # Count how many times each true label appears (for x-axis annotations)
    true_label_counts = Counter(y_true)

    # Separate mismatches by cross-element vs intra-element
    cross_cat = [
        (t, p) for t, p in mismatches
        if extract_element(t, elements) != extract_element(p, elements)
    ]
    intra_cat = [
        (t, p) for t, p in mismatches
        if extract_element(t, elements) == extract_element(p, elements)
    ]

    # Plot heatmaps for cross-element mismatches
    if cross_cat:
        _conf_matrix_from_pairs(
            cross_cat,
            true_label_counts,
            title="Cross-element Confusion Matrix (Non-Zero Only)",
            cmap="OrRd",
            annot=annot
        )
    else:
        print("No cross-element mismatches found.")

    # Plot heatmaps for intra-element mismatches
    if intra_cat:
        _conf_matrix_from_pairs(
            intra_cat,
            true_label_counts,
            title="Intra-element Confusion Matrix (Non-Zero Only)",
            cmap="BuPu",
            annot=annot
        )
    else:
        print("No intra-element mismatches found.")


def plot_element_confusion_matrices(
    pred_record: PredictionRecord,
    elements: Set[str] = DEFAULT_ELEMENTS,
    annot: bool = True
) -> None:
    """
    Generates and displays per-element confusion matrices for intra-element mismatches.

    Each matrix shows how atom types within the same element are misclassified with one another.
    Only mismatches where the true and predicted atom types belong to the same element are considered.

    Args:
        pred_record (PredictionRecord): Record containing atom-level predictions and true labels.
        elements (Set[str]): Set of valid atomic element symbols used for parsing element types.
        annot (bool): Whether to annotate the heatmap with raw counts.

    Returns:
        None. Displays one confusion matrix per element if mismatches exist.
    """
    y_true = pred_record.true_labels
    y_pred = pred_record.pred_labels

    # Pair true and predicted labels, then filter to only mismatches
    pairs = np.array(list(zip(y_true, y_pred)))
    mismatches = pairs[pairs[:, 0] != pairs[:, 1]]

    # Collect all atom types across true and predicted labels
    all_atom_types = sorted(set(y_true + y_pred))

    # Pre-compute counts of all true labels (used for bar annotations)
    full_true_label_counts = Counter(y_true)

    # Iterate over each element and generate intra-element mismatch matrix
    for element in elements:
        # Atom types belonging to the current element
        element_types = [
            atom_type for atom_type in all_atom_types
            if extract_element(atom_type, elements) == element
        ]

        # Filter mismatches where both true and predicted types belong to this element
        filtered = [
            (t, p) for (t, p) in mismatches
            if t in element_types and p in element_types
        ]

        if not filtered:
            continue  # Skip if no intra-element mismatches for this element

        # Subset of true label counts for atom types in this element
        label_counts = {
            atom_type: full_true_label_counts[atom_type]
            for atom_type in element_types
        }

        title = f"{element.capitalize()} Element Confusion Matrix (Intra-element Only)"

        _conf_matrix_from_pairs(
            pairs=filtered,
            true_label_counts=label_counts,
            title=title,
            cmap="YlGnBu",
            annot=annot
        )
