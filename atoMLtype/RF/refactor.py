import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Union

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


# #  Get Accuracy Metrics
# def get_accuracies(y_true, y_pred):
#     """
#     Computes hierarchical accuracy metrics:
#     - Atom-wise accuracy
#     - Atom-type accuracy
#     - Category-wise accuracy

#     Args:
#         y_true (list): List of true atom types.
#         y_pred (list): List of predicted atom types.

#     Returns:
#         dict: Accuracy results for different hierarchical levels.
#     """
#     if len(y_true) != len(y_pred):
#         raise ValueError("y_true and y_pred must be of the same length")

#     correct_atom_wise = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

#     atom_type_counts = count_atom_types(y_true)
#     atom_type_accuracy = {atom: sum(1 for t, p in zip(y_true, y_pred) if t == p == atom) / atom_type_counts[atom]
#                           for atom in atom_type_counts}

#     category_counts = Counter(categorize_atom_type(atom) for atom in y_true)
#     category_accuracy = {cat: sum(1 for t, p in zip(y_true, y_pred) if categorize_atom_type(t) == categorize_atom_type(p) == cat)
#                          / category_counts[cat] for cat in category_counts}

#     return {
#         "atom_wise_accuracy": correct_atom_wise,
#         "atom_type_accuracy": atom_type_accuracy,
#         "category_accuracy": category_accuracy
#     }


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

# Plot Complete Heatmap for Confusion Matrix
def plot_category_confusion_matrix(y_true, y_pred):
    """
    Generates confusion matrix heatmap.

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.

    Returns:
        None (Displays heatmaps).
    """
    # Compute confusion matrix
    all_atom_types = sorted(set(y_true) | set(y_pred))
    confusion_matrix = pd.DataFrame(0, index=all_atom_types, columns=all_atom_types, dtype=float)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix.loc[true_label, pred_label] += 1

    # Normalize matrix
    row_sums = confusion_matrix.sum(axis=1)
    normalized_matrix = confusion_matrix.div(row_sums, axis=0) * 100
    normalized_matrix = normalized_matrix.fillna(0)

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        normalized_matrix.T, fmt=".1f", cmap="BuPu",
        linewidths=0.5, square=True, cbar=True, cbar_kws={"label": "Percentage"},
        annot_kws={"size": 8}
    )

    plt.xlabel("Reference Types")
    plt.ylabel("Predicted Types")
    plt.title(f"Confusion Matrix Atom Types")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()

# Plot Heatmap for Category-Based Confusion Matrix
def plot_category_confusion_matrix(y_true, y_pred, categories=DEFAULT_CATEGORIES):
    """
    Generates category-based confusion matrix heatmaps.

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.

    Returns:
        None (Displays heatmaps).
    """
    # Compute confusion matrix
    all_atom_types = sorted(set(y_true) | set(y_pred))
    confusion_matrix = pd.DataFrame(0, index=all_atom_types, columns=all_atom_types, dtype=float)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix.loc[true_label, pred_label] += 1

    # Normalize matrix
    row_sums = confusion_matrix.sum(axis=1)
    normalized_matrix = confusion_matrix.div(row_sums, axis=0) * 100
    normalized_matrix = normalized_matrix.fillna(0)

    # Generate heatmaps for categories
    for category in categories:
        category_rows = [label for label in all_atom_types if categorize_atom_type(label) == category]

        if not category_rows:
            continue

        category_matrix = normalized_matrix.loc[category_rows, category_rows]

        if category_matrix.empty or category_matrix.sum().sum() == 0:
            continue

        # Plot Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            category_matrix.T, annot=True, fmt=".1f", cmap="BuPu",
            linewidths=0.5, square=True, cbar=True, cbar_kws={"label": "Percentage"},
            xticklabels=category_matrix.index, yticklabels=category_matrix.columns,
            annot_kws={"size": 8}
        )

        plt.xlabel("Reference Types")
        plt.ylabel("Predicted Types")
        plt.title(f"Confusion Matrix for {category} Atom Types")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.show()

# Plot Heatmap for Cross-Category Confusion Matrix
def plot_cross_category_confusion_matrix(y_true, y_pred, categories=DEFAULT_CATEGORIES):
    """
    Generates a cross-category confusion matrix heatmap showing misclassification percentages
    as a proportion of the total references for each category.

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.

    Returns:
        None (Displays the heatmap).
    """
    # Initialize confusion matrix (categories x categories)
    category_conf_matrix = pd.DataFrame(0, index=categories, columns=categories, dtype=float)

    # Step 1: Count Cross-Category Misclassifications
    category_totals = Counter(categorize_atom_type(label) for label in y_true)  # Total true instances per category

    for true_label, pred_label in zip(y_true, y_pred):
        true_category = categorize_atom_type(true_label)
        pred_category = categorize_atom_type(pred_label)

        if true_category != pred_category:  # Only track cross-category misclassifications
            category_conf_matrix.loc[true_category, pred_category] += 1

    # Step 2: Normalize by total occurrences of each category in y_true
    for category in categories:
        if category_totals[category] > 0:  # Prevent division by zero
            category_conf_matrix.loc[category] = (
                category_conf_matrix.loc[category] / category_totals[category] * 100
            )

    # Step 3: Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        category_conf_matrix, annot=True, fmt=".1f", cmap="OrRd",
        linewidths=0.5, square=True, cbar=True, cbar_kws={"label": "Percentage of Total Reference"},
        xticklabels=category_conf_matrix.columns, yticklabels=category_conf_matrix.index,
        annot_kws={"size": 10}
    )

    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.title("Cross-Category Misclassification Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()





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
    plot_heatmap(normalized_matrix, "Confusion Matrix for All Atom Types", annot)

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
        plot_heatmap(category_matrix, f"Confusion Matrix for {category} Atom Types", annot=annot)

# Plot Cross-Category Confusion Matrix
def plot_detailed_cross_category_confusion(y_true, y_pred, categories=DEFAULT_CATEGORIES, annot=True):
    """
    Generates a heatmap showing only the cross-category misclassified atom types.
    Also prints the overall misclassification rate.

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.
        categories (set): Set of known atom categories.
        annot (bool): Whether to annotate the heatmap.

    Returns:
        None (Displays heatmap and prints misclassification summary).
    """
    # Step 1: Compute Misclassification Matrix (Atom-level across categories)
    cross_category_conf_matrix = pd.DataFrame(0, index=list(set(y_true)), columns=list(set(y_pred)), dtype=float)
    
    # Track total misclassified atoms
    total_misclassified = 0
    total_atoms = len(y_true)  # Total instances
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_category = categorize_atom_type(true_label)
        pred_category = categorize_atom_type(pred_label)

        if true_category != pred_category:  # **Only count cross-category misclassifications**
            cross_category_conf_matrix.loc[true_label, pred_label] += 1
            total_misclassified += 1

    # Normalize by total occurrences of each atom type
    atom_type_totals = Counter(y_true)  # Count total occurrences of each atom type
    for atom in cross_category_conf_matrix.index:
        if atom in atom_type_totals and atom_type_totals[atom] > 0:
            cross_category_conf_matrix.loc[atom] = (
                cross_category_conf_matrix.loc[atom] / atom_type_totals[atom] * 100
            )

    # **Remove rows and columns with only zeros (no misclassifications)**
    cross_category_conf_matrix = cross_category_conf_matrix.loc[(cross_category_conf_matrix.sum(axis=1) > 0), :]
    cross_category_conf_matrix = cross_category_conf_matrix.loc[:, (cross_category_conf_matrix.sum(axis=0) > 0)]

    # Step 2: Compute Overall Misclassification Rate
    overall_misclassification_rate = (total_misclassified / total_atoms) * 100

    # Step 3: Print Summary Statistics
    print(f"🔹 Total Atoms: {total_atoms}")
    print(f"🔹 Total Misclassified Atoms: {total_misclassified}")
    print(f"🔹 Overall Misclassification Rate: {overall_misclassification_rate:.2f}%\n")

    # Step 4: Plot Heatmap (Only if there are misclassifications)
    if not cross_category_conf_matrix.empty:
        plot_heatmap(
            cross_category_conf_matrix, 
            "Detailed Cross-Category Misclassification Heatmap", 
            cmap="OrRd", 
            annot=annot
        )
    else:
        print("✅ No significant cross-category misclassifications found!")


def get_accuracies_from_confusion(y_true, y_pred, categories=DEFAULT_CATEGORIES):
    """
    Computes hierarchical accuracy metrics using the confusion matrix:
    - Atom-wise accuracy
    - Atom-type accuracy
    - Category-wise accuracy

    Args:
        y_true (list): List of true atom types.
        y_pred (list): List of predicted atom types.
        categories (set): Default set of atom categories.

    Returns:
        dict: Accuracy results at different hierarchical levels.
    """
    # Compute Confusion Matrix
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)

    # Total correct predictions (sum of diagonal elements)
    total_correct = confusion_matrix.values.diagonal().sum()
    total_instances = confusion_matrix.values.sum()

    # Compute Atom-wise Accuracy
    atom_wise_accuracy = total_correct / total_instances if total_instances > 0 else 0.0

    # Compute Atom-type Accuracy (Correct per type / Total per type)
    atom_type_accuracy = {
        atom: confusion_matrix.loc[atom, atom] / confusion_matrix.loc[atom].sum()
        if confusion_matrix.loc[atom].sum() > 0 else 0.0
        for atom in confusion_matrix.index
    }

    # Compute Category-wise Accuracy
    category_totals = Counter(categorize_atom_type(label) for label in y_true)

    category_correct = Counter()
    for true_label in confusion_matrix.index:
        for pred_label in confusion_matrix.columns:
            if categorize_atom_type(true_label) == categorize_atom_type(pred_label):
                category_correct[categorize_atom_type(true_label)] += confusion_matrix.loc[true_label, pred_label]

    category_accuracy = {
        cat: category_correct[cat] / category_totals[cat] if category_totals[cat] > 0 else 0.0
        for cat in categories
    }

    return {
        "atom_wise_accuracy": atom_wise_accuracy,
        "atom_type_accuracy": atom_type_accuracy,
        "category_accuracy": category_accuracy
    }
