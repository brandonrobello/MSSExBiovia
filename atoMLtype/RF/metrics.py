import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Union

# DEFINE GLOBALLY THE DEFAULT CATAGORIES
def categorize_atom_type(atom_type, categories):
    """Returns the elemental category based on the first letter of the atom type."""
    if not atom_type:
        return "unknown"  # Handle empty values
    return atom_type[0].lower() if atom_type[0].lower() in categories else "other"


def atom_type_counts(y_labels: Union[Dict[str, List[str]], np.ndarray], order=None, plot=True):
    """
    Generates a bar plot showing the distribution of atom types in the dataset
    and returns the data as a Pandas DataFrame.

    This function can handle:
    1. A dictionary where each molecule name is mapped to a list of atom types.
    2. A NumPy array containing atom types.

    Args:
        y_labels (Dict[str, List[str]] or np.ndarray): 
            - If a dictionary: {molecule_name (str): atom_types (list[str])}
            - If a numpy array: An array of atom types.

        order (List[str] or None, optional): 
            A custom order for atom types in the plot. If provided, the atom types will be 
            displayed in the specified order. If None, a default ordering is used:
                1. Atom types starting with 'h'
                2. Atom types starting with 'c'
                3. Atom types starting with 'n'
                4. All other atom types are grouped under "other" and sorted by count.

        plot (bool, optional): 
            If True (default), the function generates a bar plot. If False, it only returns the DataFrame.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the atom type counts with columns:
                - "Atom Type" (str): The atom type.
                - "Count" (int): The frequency of the atom type in the dataset.

    Notes:
        - If `order` is provided, it overrides the default categorization and sorting.
        - Atom types not explicitly listed in `order` will be included under "other".
    """

    # Handle numpy array input by converting it to a list
    if isinstance(y_labels, np.ndarray):
        if y_labels.ndim > 1:
            all_atom_types = y_labels.flatten().tolist()  # Flatten 2D arrays
        else:
            all_atom_types = y_labels.tolist()
    elif isinstance(y_labels, dict):
        # Flatten all atom types from y_labels dictionary
        all_atom_types = []
        for labels in y_labels.values():
            all_atom_types.extend(labels)
    else:
        raise TypeError("y_labels must be a dictionary or a numpy array.")

    # Count occurrences
    atom_type_counts = Counter(all_atom_types)

    # Define default ordering: ["h", "c", "n"], but allow custom orders
    ordering = order if order else ["h", "c", "n"]
    if "other" not in ordering:
        ordering.append("other")  # Ensure "other" category is always included

    # Categorize atom types
    categorized = {cat: [] for cat in ordering}

    for atom_type, count in atom_type_counts.items():
        prefix = atom_type[0].lower()
        if prefix in ordering:
            categorized.setdefault(prefix, []).append((atom_type, count))
        else:
            categorized.setdefault("other", []).append((atom_type, count))

    # Sort categories: first 3 (h, c, n) by natural order, "other" by count
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x[1], reverse=True)

    # Reassemble sorted atom types based on `ordering`
    sorted_atom_types = []
    for cat in ordering:
        if cat in categorized:
            sorted_atom_types.extend(categorized[cat])

    # Convert to DataFrame for plotting
    atom_type_df = pd.DataFrame(sorted_atom_types, columns=["Atom Type", "Count"])

    # Number of unique atom types
    print(f"Total number of unique atom types = {len(atom_type_df['Atom Type'])}")

    # Plot distribution
    if plot:
        plt.figure(figsize=(18, 6))
        sns.barplot(x="Atom Type", y="Count", data=atom_type_df)
        plt.xlabel("Atom Type")
        plt.ylabel("Frequency")
        plt.title("Distribution of Atom Types")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=90)
        plt.show()

    return atom_type_df  # Return the DataFrame for further use

