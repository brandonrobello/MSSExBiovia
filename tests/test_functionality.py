import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import logging
import importlib
from collections import Counter
from torch.utils.data import random_split

from atoMLtype.datasets.GNNdataset import GNNdataset
from atoMLtype.models.ModelEncoder import ModelEncoder
from atoMLtype.models.GNN.DMPNNmodel import AtomBondMPNN
from atoMLtype.models.ModelTrainer import GNNTrainer
from atoMLtype.models.ModelEngine import ModelEngine


# ------------------------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------------------------

def build_test_model(train_dataset, encoder):
    """
    Constructs a test model using the dimensions from the training dataset.

    Args:
        train_dataset (Dataset): A dataset containing PyG molecular graphs.
        encoder (ModelEncoder): The label encoder used for atom types.

    Returns:
        AtomBondMPNN: Initialized GNN model.
    """
    return AtomBondMPNN(
        atom_input_dim=train_dataset[0].x.shape[1],
        bond_input_dim=train_dataset[0].edge_attr.shape[1],
        hidden_dim=128,
        encoder=encoder,
        num_layers=5,
        use_attention=True
    )

def get_first_valid_atom(prediction_record):
    """
    Returns the first valid predicted atom from the record.

    Args:
        prediction_record (PredictionRecord): Object holding all atom-level predictions.

    Returns:
        Tuple[str, int, str]: (mol_name, atom_idx, pred_label)

    Raises:
        ValueError: If no valid atoms are found in the record.
    """
    for atom in prediction_record.atom_records:
        if atom.mol_name and atom.pred_label:
            return atom.mol_name, atom.atom_idx_in_mol, atom.pred_label
    raise ValueError("No valid atoms found in prediction record")


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup_data():
    """
    Loads the dataset, initializes the encoder, and splits into train/test sets.

    Returns:
        Tuple: (ModelEncoder, train_dataset, test_dataset)
    """
    encoder = ModelEncoder(collapse=True)
    sdf_path = "tests/test_data/zinc_subsampled.sdf"
    json_labels = "tests/test_data/atomLabels_gaff2_subsampled.json"

    dirEdge_dataset = GNNdataset(
        sdf_path,
        json_labels,
        directed_graph=True,
        labeled=True,
        encoder=encoder
    )

    train_size = int(0.90 * len(dirEdge_dataset))
    test_size = len(dirEdge_dataset) - train_size
    train_dataset, test_dataset = random_split(dirEdge_dataset, [train_size, test_size])
    return encoder, train_dataset, test_dataset

@pytest.fixture(scope="module")
def trained_model_and_predictions(setup_data):
    """
    Trains the GNN model and runs inference to obtain predictions.

    Returns:
        Tuple: (trained_model, PredictionRecord)
    """
    encoder, train_dataset, test_dataset = setup_data
    model = build_test_model(train_dataset, encoder)

    trainer = GNNTrainer(
        model=model,
        dataset=train_dataset,
        batch_size=32,
        learning_rate=0.001,
        epochs=5,
        k_folds=2,
        random_seed=21
    )
    trainer.train(draw_curve=False, verbose=False)

    engine = ModelEngine(model=model, dataset=test_dataset, device="cpu", batch_size=32)
    prediction_record = engine.predict(analysis=True)

    return model, prediction_record


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_dataset_loading(setup_data):
    """
    Ensures that the dataset loads and splits correctly.
    """
    _, train_dataset, test_dataset = setup_data
    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"

def test_model_initialization(setup_data):
    """
    Verifies model structure and input dimensions.
    """
    encoder, train_dataset, _ = setup_data
    assert len(train_dataset) > 0, "Train dataset is empty"

    graph = train_dataset[0]
    assert hasattr(graph, "x"), "Graph missing node features"
    assert hasattr(graph, "edge_attr"), "Graph missing edge attributes"

    model = build_test_model(train_dataset, encoder)

    assert model is not None, "Model is None"
    assert hasattr(model, 'forward'), "Model does not implement forward()"
    assert hasattr(model, 'encoder'), "Model missing encoder"

def test_training_runs(trained_model_and_predictions):
    """
    Confirms that training and inference complete successfully.
    """
    model, prediction_record = trained_model_and_predictions
    assert model is not None
    assert prediction_record is not None

def test_prediction_atom_types(trained_model_and_predictions):
    """
    Ensures common atom types are predicted.
    """
    _, prediction_record = trained_model_and_predictions
    pred_counter = prediction_record.pred_label_counter

    expected_major_types = ['ca', 'hc', 'ha', 'c3', 'h1']
    for atom_type in expected_major_types:
        assert atom_type in pred_counter, f"Missing expected atom type: {atom_type}"
        assert pred_counter[atom_type] > 0, f"Predicted count for {atom_type} is zero"

def test_prediction_accuracy(trained_model_and_predictions):
    """
    Checks if model accuracy is above a minimum threshold.
    """
    _, prediction_record = trained_model_and_predictions
    accuracy = prediction_record.accuracy
    assert accuracy > 0.70, f"Prediction accuracy too low: {accuracy:.2f}"

def test_label_diversity(trained_model_and_predictions):
    """
    Asserts that the model predicts a diverse set of atom types.
    """
    _, prediction_record = trained_model_and_predictions
    num_labels = len(prediction_record.pred_label_counter)
    assert num_labels >= 30, f"Too few predicted atom types: {num_labels}"

def test_confidence_statistics(trained_model_and_predictions):
    """
    Validates that prediction confidence scores are within reasonable bounds.
    """
    _, prediction_record = trained_model_and_predictions
    min_conf = prediction_record.min_confidence
    mean_conf = prediction_record.mean_confidence
    max_conf = prediction_record.max_confidence

    assert min_conf > 0.1, f"Minimum confidence too low: {min_conf}"
    assert mean_conf > 0.7, f"Mean confidence too low: {mean_conf}"
    assert max_conf <= 1.0, f"Maximum confidence should not exceed 1.0, got {max_conf}"

def test_multiple_atom_predictions(trained_model_and_predictions, N=3):
    """
    Verifies that at least N atoms are correctly predicted.

    Args:
        N (int): Number of atoms to check for prediction accuracy.
    """
    _, prediction_record = trained_model_and_predictions
    checked = 0

    for atom in prediction_record.atom_records:
        if atom.true_label is not None:
            assert atom.pred_label == atom.true_label, (
                f"[{atom.mol_name}] Atom {atom.atom_idx_in_mol}: predicted '{atom.pred_label}', expected '{atom.true_label}'"
            )
            checked += 1
            if checked >= N:
                break

    assert checked == N, f"Checked only {checked} atoms (expected {N})"
