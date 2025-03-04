import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Union
from sklearn.metrics import accuracy_score

### DEFINE GLOBAL DEFAULT CATEGORIES
DEFAULT_CATEGORIES = {"c", "h", "o", "n", "other"}

# Function to categorize atom types
def categorize_atom_type(atom_type: str, categories=DEFAULT_CATEGORIES) -> str:
    """Returns the elemental category based on the first letter or known prefixes of the atom type."""
    if not atom_type:
        return "unknown"

    atom_type_lower = atom_type.lower()

    # Check if atom type starts with a known elemental category
    for cat in categories:
        if atom_type_lower.startswith(cat):  # Prefix-based check
            return cat

    return "other"  # Default if no match


# Count Atom Type Occurrences
def count_atom_types(y_labels: Union[Dict[str, List[str]], np.ndarray]) -> Counter:
    """
    Counts occurrences of each atom type in a given dataset.

    Args:
        y_labels: A dictionary {molecule_name: [atom_types]} or a NumPy array of atom types.

    Returns:
        Counter: A dictionary-like object with atom type counts.
    """
    # Convert input to a flattened list of atom types
    if isinstance(y_labels, np.ndarray):
        all_atom_types = y_labels.flatten().tolist() if y_labels.ndim > 1 else y_labels.tolist()
    elif isinstance(y_labels, dict):
        all_atom_types = [atom for labels in y_labels.values() for atom in labels]
    else:
        raise TypeError("y_labels must be a dictionary or a numpy array.")

    return Counter(all_atom_types)

def get_accuracies(y_true, y_pred):
    """
    Computes hierarchical accuracy metrics:
    - Atom-wise accuracy (matches `accuracy_score`)
    - Atom-type accuracy
    - Category-wise accuracy (correct predictions within category)
    - Cross-category misclassification rate
    - Per-category atom-type accuracy

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.

    Returns:
        dict: Accuracy results for different hierarchical levels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length")

    # Compute overall atom-wise accuracy (matches sklearn's accuracy_score)
    correct_atom_wise = accuracy_score(y_true, y_pred)

    # Compute atom-type accuracy
    atom_type_counts = count_atom_types(y_true)  # Get total occurrences of each atom type
    atom_type_accuracy = {
        atom: sum(1 for t, p in zip(y_true, y_pred) if t == p == atom) / atom_type_counts[atom]
        if atom_type_counts[atom] > 0 else 0.0
        for atom in atom_type_counts
    }

    # Compute category-wise accuracy (correct predictions staying within category)
    category_counts = Counter(categorize_atom_type(atom) for atom in y_true)  # Count of true labels per category
    category_correct = Counter()  # Track correct predictions per category
    cross_category_misclassifications = Counter()  # Track misclassifications per category

    for true_label, pred_label in zip(y_true, y_pred):
        true_cat = categorize_atom_type(true_label)
        pred_cat = categorize_atom_type(pred_label)

        if true_cat == pred_cat:  # Correct within category
            category_correct[true_cat] += 1
        else:  # Went into another category
            cross_category_misclassifications[true_cat] += 1

    # Normalize category accuracy
    category_accuracy = {
        cat: category_correct[cat] / category_counts[cat] if category_counts[cat] > 0 else 0.0
        for cat in category_counts
    }

    # Compute cross-category misclassification rate (percent of misclassified atoms per category)
    cross_category_accuracy = {
        cat: cross_category_misclassifications[cat] / category_counts[cat] if category_counts[cat] > 0 else 0.0
        for cat in category_counts
    }

    # Compute per-category atom-type accuracy
    category_atom_type_accuracy = {}
    for cat in category_counts:
        # Get all atom types that belong to this category
        atom_types_in_category = [atom for atom in atom_type_counts if categorize_atom_type(atom) == cat]

        # Compute accuracy per atom type within this category
        category_atom_type_accuracy[cat] = {
            atom: atom_type_accuracy[atom] for atom in atom_types_in_category
        }

    return {
        "atom_wise_accuracy": correct_atom_wise,
        "atom_type_accuracy": atom_type_accuracy,
        "category_accuracy": category_accuracy,
        "cross_category_misclassification_rate": cross_category_accuracy,
        "category_atom_type_accuracy": category_atom_type_accuracy
    }



# Plot Atom Type Distribution
def plot_atom_distribution(y_labels, order=DEFAULT_CATEGORIES, plot=True):
    """
    Generates a bar plot showing the distribution of atom types and returns a DataFrame.

    Args:
        y_labels: Dictionary of {mol_name: [atom_types]} or a NumPy array.
        order: Optional custom sorting order for atom types.
        plot: If True, displays a bar plot.

    Returns:
        pd.DataFrame: Dataframe with atom type counts.
    """
    atom_type_counts = count_atom_types(y_labels)

    # Categorize atom types
    categorized = {cat: [] for cat in order}
    for atom_type, count in atom_type_counts.items():
        category = categorize_atom_type(atom_type)
        categorized.setdefault(category, []).append((atom_type, count))

    # Sort categories
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x[1], reverse=True)

    # Reassemble sorted list
    sorted_atom_types = [(atom, count, cat) for cat in order if cat in categorized for atom, count in categorized[cat]]

    # Convert to DataFrame
    atom_type_df = pd.DataFrame(sorted_atom_types, columns=["Atom Type", "Count", "Category"])

    if plot:
        plt.figure(figsize=(18, 6))
        sns.barplot(x="Atom Type", y="Count", hue="Category", palette="bright", data=atom_type_df)
        plt.xlabel("Atom Type")
        plt.ylabel("Frequency")
        plt.title("Distribution of Atom Types")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=90)
        plt.legend(title="Category")
        plt.show()

    return atom_type_df

# Compute Confusion Matrix
def compute_confusion_matrix(y_true: List[str], y_pred: List[str]):
    """
    Computes the full confusion matrix for atom types.
    
    Args:
        y_true: List of true atom types.
        y_pred: List of predicted atom types.
    
    Returns:
        pd.DataFrame: A DataFrame containing the confusion matrix.
    """
<<<<<<< HEAD
=======

        # Flatten lists if they contain nested lists
    if isinstance(y_true[0], list): 
        y_true = sum(y_true, [])
    if isinstance(y_pred[0], list):
        y_pred = sum(y_pred, [])

    # Convert NumPy arrays to lists of integers (if not already lists)
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()

    # Flatten lists (handles cases where labels are stored as nested lists)
    y_true = [int(item) if isinstance(item, (list, np.ndarray)) else item for item in y_true]
    y_pred = [int(item) if isinstance(item, (list, np.ndarray)) else item for item in y_pred]

>>>>>>> b80bc37 (GNN CODE: Implementations for GNN Dev and testing)
    all_atom_types = sorted(set(y_true) | set(y_pred))
    confusion_matrix = pd.DataFrame(0, index=all_atom_types, columns=all_atom_types, dtype=float)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix.loc[true_label, pred_label] += 1

    # Normalize matrix
    row_sums = confusion_matrix.sum(axis=1).replace(0, 1)  # Avoid division by zero
    normalized_matrix = confusion_matrix.div(row_sums, axis=0) * 100

    return normalized_matrix

# Function to plot a heatmap
def plot_heatmap(matrix, title, xlabel="Reference Types", ylabel="Predicted Types", cmap="BuPu", figsize=(12, 10), annot=False):
    """
    Plots a heatmap for a given confusion matrix.

    Args:
        matrix (pd.DataFrame): Confusion matrix data.
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

# Plot Full Confusion Matrix
def plot_full_confusion_matrix(y_true, y_pred, annot=False):
    """
    Generates a full confusion matrix heatmap for atom types.
    """
    normalized_matrix = compute_confusion_matrix(y_true, y_pred)
    
    total_atoms = len(y_true)
    total_misclassified = sum(t != p for t, p in zip(y_true, y_pred))

    # Compute misclassification Rate
    overall_misclassification_rate = (total_misclassified / total_atoms) * 100

    title = f"Confusion Matrix for All Atom Types\n\
        Total Atoms: {total_atoms} | Total Misclassified: {total_misclassified} ({overall_misclassification_rate:.2f}%)"

    plot_heatmap(normalized_matrix, title, annot)

# Plot Category-Based Confusion Matrices
def plot_category_confusion_matrix(y_true, y_pred, categories=DEFAULT_CATEGORIES, annot=True):
    """
    Generates category-based confusion matrix heatmaps.
    """
    normalized_matrix = compute_confusion_matrix(y_true, y_pred)
    all_atom_types = sorted(set(y_true) | set(y_pred))

    for category in categories:
        category_rows = [label for label in all_atom_types if categorize_atom_type(label) == category]

        if not category_rows:
            continue

        category_matrix = normalized_matrix.loc[category_rows, category_rows]

        # Count total atoms and misclassifications for this category
        total_category_atoms = sum(1 for t in y_true if categorize_atom_type(t) == category)
        misclassified_in_category = sum(1 for t, p in zip(y_true, y_pred) if categorize_atom_type(t) == category and t != p)

        # Compute misclassification Rate
        overall_misclassification_rate = (misclassified_in_category / total_category_atoms) * 100

        title = f"Confusion Matrix for {category} Atom Types\n\
            Total {category} Atoms: {total_category_atoms} | Total Misclassified: {misclassified_in_category} ({overall_misclassification_rate:.2f}%)"

        plot_heatmap(category_matrix, title, annot=annot)


# Plot Cross-Category Confusion Matrix
def plot_detailed_cross_category_confusion(y_true, y_pred, categories=DEFAULT_CATEGORIES, annot=True):
    """
    Generates a heatmap showing only the cross-category misclassified atom types.
    The title includes misclassification statistics.

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.
        categories (set): Set of known atom categories.
        annot (bool): Whether to annotate the heatmap.

    Returns:
        None (Displays heatmap).
    """
    # Compute Misclassification Matrix (Atom-level across categories)
    cross_category_conf_matrix = pd.DataFrame(0, index=list(set(y_true)), columns=list(set(y_pred)), dtype=float)
    
    total_misclassified = 0
    total_atoms = len(y_true)  # Total instances
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_category = categorize_atom_type(true_label)
        pred_category = categorize_atom_type(pred_label)

        if true_category != pred_category:  # Only count cross-category misclassifications
            cross_category_conf_matrix.loc[true_label, pred_label] += 1
            total_misclassified += 1

    # Normalize by total occurrences of each atom type
    atom_type_totals = Counter(y_true)
    for atom in cross_category_conf_matrix.index:
        if atom in atom_type_totals and atom_type_totals[atom] > 0:
            cross_category_conf_matrix.loc[atom] = (
                cross_category_conf_matrix.loc[atom] / atom_type_totals[atom] * 100
            )

    # Remove rows and columns with only zeros (no misclassifications)
    cross_category_conf_matrix = cross_category_conf_matrix.loc[(cross_category_conf_matrix.sum(axis=1) > 0), :]
    cross_category_conf_matrix = cross_category_conf_matrix.loc[:, (cross_category_conf_matrix.sum(axis=0) > 0)]

    # Compute Overall Misclassification Rate
    overall_misclassification_rate = (total_misclassified / total_atoms) * 100

    # Generate Heatmap Title with Statistics
    heatmap_title = f"Cross-Category Misclassification Heatmap\n\
        Total Atoms: {total_atoms} | Misclassified: {total_misclassified} ({overall_misclassification_rate:.2f}%)"

    # Plot Heatmap (Only if there are misclassifications)
    if not cross_category_conf_matrix.empty:
        plot_heatmap(
            cross_category_conf_matrix, 
            heatmap_title, 
            cmap="OrRd", 
            annot=annot
        )
    else:
        print("No cross-category misclassifications found!")
