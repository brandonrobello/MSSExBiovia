import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Union, DefaultDict
from sklearn.metrics import accuracy_score

from atoMLtype.models.ModelOutput import PredictionRecord, AtomResult


### DEFINE GLOBAL DEFAULT CATEGORIES
DEFAULT_ELEMENTS = {"c", "h", "o", "n", "s", "p", "other"}

def extract_element(atom_type: str, elements: set = DEFAULT_ELEMENTS) -> str:
    """
    Categorizes an atom type based on its elemental prefix.

    Args:
        atom_type (str): The atom type to categorize.
        elements (set): Set of known atom elements.

    Returns:
        str: The element of the atom type, or "other" if no match is found.
    """
    if not atom_type:
        return "unknown"

    atom_type_lower = atom_type.lower()

    # Check if atom type starts with a known elemental element
    for cat in elements:
        if atom_type_lower.startswith(cat):  # Prefix-based check
            return cat

    return "other"  # Default if no match


def count_atom_types(y_labels: Union[Dict[str, List[str]], np.ndarray]) -> Counter:
    """
    Counts occurrences of each atom type in a dataset.

    Args:
        y_labels (Union[Dict[str, List[str]], np.ndarray]): A dictionary {molecule_name: [atom_types]} 
            or a NumPy array of atom types.

    Returns:
        Counter: A dictionary-like object with atom type counts.
    """
    # Flatten input to a list of atom types
    if isinstance(y_labels, np.ndarray):
        # Handle NumPy arrays (flatten if multidimensional)
        all_atom_types = y_labels.flatten().tolist() if y_labels.ndim > 1 else y_labels.tolist()
    elif isinstance(y_labels, dict):
        # Handle dictionaries by flattening all atom types
        all_atom_types = [atom for labels in y_labels.values() for atom in labels]
    else:
        raise TypeError("y_labels must be a dictionary or a numpy array.")

    return Counter(all_atom_types)


def get_accuracies_and_counts(pred_record: PredictionRecord, 
                              elements: set = DEFAULT_ELEMENTS) -> Dict:
    """
    Computes hierarchical accuracy metrics and abundances from a PredictionRecord.

    Returns:
        dict with:
            - accuracy (float): Overall accuracy across all atoms.
            - atom_type_accuracy (dict): Accuracy per atom type.
            - atom_type_counts (dict): Count per atom type.
            - element_accuracy (dict): Accuracy per chemical element.
            - element_counts (dict): Count per chemical element.
            - cross_element_misclassification_rate (dict): % misclassified outside true element category.
            - element_atom_type_accuracy (dict of dicts): Per-atom-type accuracy within each element.
    """
    y_true = pred_record.true_labels
    y_pred = pred_record.pred_labels

    if len(y_true) != len(y_pred):
        raise ValueError("PredictionRecord length mismatch between true and predicted labels")

    # Atom-wise accuracy
    atom_wise_acc = accuracy_score(y_true, y_pred)

    # Atom-type accuracy
    atom_type_counts = Counter(y_true)
    atom_type_accuracy = {
        at: sum(1 for t, p in zip(y_true, y_pred) if t == p == at) / atom_type_counts[at]
        for at in atom_type_counts
    }

    # element-wise metrics
    element_counts = Counter(extract_element(t, elements) for t in y_true)
    element_correct = Counter()
    cross_element_misclassified = Counter()

    for t, p in zip(y_true, y_pred):
        true_elem = extract_element(t)
        pred_elem = extract_element(p)
        if true_elem == pred_elem:
            element_correct[true_elem] += 1
        else:
            cross_element_misclassified[true_elem] += 1

    element_accuracy = {
        elem: element_correct[elem] / element_counts[elem]
        for elem in element_counts
    }

    cross_element_rate = {
        elem: cross_element_misclassified[elem] / element_counts[elem]
        for elem in element_counts
    }

    # Per-element atom type accuracy
    element_atom_type_accuracy = DefaultDict(dict)
    for at in atom_type_counts:
        elem = extract_element(at)
        element_atom_type_accuracy[elem][at] = atom_type_accuracy[at]

    return {
        "accuracy": atom_wise_acc,
        "atom_type_accuracy": atom_type_accuracy,
        "atom_type_counts": dict(atom_type_counts),
        "element_accuracy": element_accuracy,
        "element_counts": dict(element_counts),
        "cross_element_misclassification_rate": cross_element_rate,
        "element_atom_type_accuracy": element_atom_type_accuracy
    }

def plot_atom_distribution(y_labels: Union[List[str], Dict[str, List[str]], np.ndarray],
                           sort_by: str = 'alphabetical',
                           plot: bool = True) -> pd.DataFrame:
    """
    Generates a bar plot showing the distribution of atom types.

    Args:
        y_labels (Union[List[str], Dict[str, List[str]], np.ndarray]):
            A flat list/array of atom types, or a dictionary of {mol_name: [atom_types]}.
        sort_by (str): Sorting method for atom types. One of ['alphabetical', 'frequency'].
        plot (bool): If True, displays a bar plot.
    """
    # Flatten label list if passed as a dict of mols
    if isinstance(y_labels, dict):
        all_labels = [at for ats in y_labels.values() for at in ats]
    elif isinstance(y_labels, (list, np.ndarray)):
        all_labels = list(y_labels)
    else:
        raise TypeError("y_labels must be a list, np.ndarray, or dict of lists")

    # Count and classify atom types
    atom_type_counts = Counter(all_labels)
    rows = [(atom_type, count, extract_element(atom_type))
            for atom_type, count in atom_type_counts.items()]

    df = pd.DataFrame(rows, columns=["Atom Type", "Count", "Element"])

    # Sorting
    if sort_by == 'frequency':
        df = df.sort_values("Count", ascending=False)
    elif sort_by == 'alphabetical':
        df = df.sort_values("Atom Type")
    else:
        raise ValueError("sort_by must be 'alphabetical' or 'frequency'")

    if plot:
        plt.figure(figsize=(18, 6))
        sns.barplot(data=df, x="Atom Type", y="Count", hue="Element", dodge=False, palette="tab10")
        plt.xlabel("Atom Type")
        plt.ylabel("Frequency")
        plt.title("Distribution of Atom Types")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=90)
        plt.legend(title="Element", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        plt.show()