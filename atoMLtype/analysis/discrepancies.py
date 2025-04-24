import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap

from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import os
from typing import List, Optional

from atoMLtype.models.ModelOutput import PredictionRecord, AtomResult
from atoMLtype.analysis.accuracy_counts import extract_element


def plot_confidence_by_pred_label(pred_record: PredictionRecord, show_mismatch: bool = True,
                                  sort_by: str = 'alphabetical', showfliers: bool = False, 
                                  figsize: tuple = (10, 5)):
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
        "confidence": atom.confidence,
        "is_mismatch": atom.pred_label != atom.true_label
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

    # Overlay mismatches as red dots
    if show_mismatch:
        mismatches = df[df['is_mismatch']]
        sns.stripplot(x='pred_label', y='confidence', data=mismatches, 
                    order=label_order, color='red', size=4, jitter=True, alpha=0.7)
    
    plt.xticks(rotation=90)
    plt.title("Confidence by Predicted Label")
    plt.xlabel("Predicted Label")
    plt.ylabel("Confidence (Softmax)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_discrepancy_distribution(pred_record: PredictionRecord, figsize=(5, 5)):
    """
    Plots a donut chart of how many molecules have 0, 1, or 2+ mismatches.

    Args:
        pred_record (PredictionRecord): A record of predictions.
        figsize (tuple): Figure size for the donut chart.
    """
    # Count mismatch groups
    counts = [len(v) for v in pred_record.mismatched_molecules.values()]
    total_mols = len(pred_record.by_mol_name)

    zero = total_mols - len(counts)
    one = sum(1 for c in counts if c == 1)
    two_or_more = sum(1 for c in counts if c >= 2)

    sizes = [zero, one, two_or_more]
    labels = ['0 discrepancy', '1 discrepancy', '≥ 2 discrepancies']
    colors = ['#A3F7B5', '#FFA07A', '#CD5C5C']
    explode = (0.02, 0.05, 0.1)

    # Donut chart
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, wedgeprops=dict(width=0.4)
    )
    ax.set_title("Number of Discrepancies per Molecule", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_element_discrepancy_rate(pred_record: PredictionRecord, valid_elements: set, figsize=(12, 6)):
    """
    Plots discrepancy rate per element and shows total abundance under x-axis.

    Args:
        pred_record (PredictionRecord): Atom-level prediction results.
        valid_elements (set): Chemical element symbols (e.g., {"C", "N", "Cl", "Br"}).
        figsize (tuple): Size of the figure.
    """

    # Count per element
    total_by_element = defaultdict(int)
    mismatches_by_element = defaultdict(int)

    for atom in pred_record.atom_records:
        if atom.true_label and atom.pred_label:
            true_elem = extract_element(atom.true_label, valid_elements)
            total_by_element[true_elem] += 1
            if atom.true_label != atom.pred_label:
                mismatches_by_element[true_elem] += 1

    elements = sorted(total_by_element.keys())
    rates = [(mismatches_by_element[e] / total_by_element[e]) * 100 for e in elements]
    abundances = [total_by_element[e] for e in elements]
    xtick_labels = [f"{e}\n(n={a})" for e, a in zip(elements, abundances)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(elements, rates, color='steelblue')

    # Annotate bar height
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ax.set_ylabel("Discrepancy Rate (%)")
    ax.set_xlabel("Element\n(Abundance below)")
    ax.set_title("Discrepancy Rate per Element")
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(xtick_labels)
    plt.tight_layout()
    plt.show()

def plot_discrepancy_rate_by_atom_type(pred_record: PredictionRecord, top_n: int = 20, figsize=(12, 6)):
    """
    Plots discrepancy rate by atom type (true label), with abundance under each bar.

    Args:
        pred_record (PredictionRecord): Atom-level prediction results.
        top_n (int): Maximum number of atom types to display.
        figsize (tuple): Size of the figure.
    """
    total_counts = Counter(pred_record.true_labels)
    mismatch_counts = Counter(a.true_label for a in pred_record.mismatches if a.true_label)

    rates = {
        label: 100 * mismatch_counts[label] / total_counts[label]
        for label in total_counts if total_counts[label] > 0
    }

    sorted_items = sorted(rates.items(), key=lambda x: -x[1])[:top_n]
    labels, percentages = zip(*sorted_items)
    abundances = [total_counts[l] for l in labels]
    xtick_labels = [f"{l}\n(n={a})" for l, a in zip(labels, abundances)]

    plt.figure(figsize=figsize)
    sns.barplot(x=list(labels), y=list(percentages), color="steelblue")

    # Add value annotations
    for i, pct in enumerate(percentages):
        plt.text(i, pct, f"{pct:.2f}%", ha='center', va='bottom', fontsize=8)

    plt.xticks(ticks=range(len(labels)), labels=xtick_labels, rotation=45)
    plt.title("Discrepancy Rate per Atom Type")
    plt.xlabel("Atom Type\n(Abundance below)")
    plt.ylabel("Discrepancy Rate (%)")
    plt.ylim(0, max(percentages) + 10)
    plt.tight_layout()
    plt.show()

