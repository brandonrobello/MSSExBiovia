# Biovia Functionality Tests

This directory (`tests/`) contains automated unit tests for verifying the functionality of the GNN-based molecular property prediction pipeline.

##  What This Tests

The test suite checks:

- Correct loading and preprocessing of molecular data from `.sdf` and `.json` files.
- Proper initialization and architecture of the GNN model (`AtomBondMPNN`).
- Successful training and inference with `GNNTrainer` and `ModelEngine`.
- Prediction performance and statistics (accuracy, confidence, label diversity).
- Atom-level correctness for specific predictions.

##  How to Run the Tests

All tests are containerized and executed using the script:

```bash
cd tests
```

```bash
chmod +x ./build_and_test.sh
````


```bash
./build_and_test.sh
```

(make sure you are in tests directory)

# Example output of test

```
============================= test session starts ==============================
platform linux -- Python 3.11.12, pytest-7.4.0, pluggy-1.5.0 -- /usr/local/bin/python3.11
cachedir: .pytest_cache
rootdir: /app
plugins: anyio-4.9.0
collecting ... collected 8 items

tests/test_functionality.py::test_dataset_loading PASSED                 [ 12%]
tests/test_functionality.py::test_model_initialization PASSED            [ 25%]
tests/test_functionality.py::test_training_runs PASSED                   [ 37%]
tests/test_functionality.py::test_prediction_atom_types PASSED           [ 50%]
tests/test_functionality.py::test_prediction_accuracy PASSED             [ 62%]
tests/test_functionality.py::test_label_diversity PASSED                 [ 75%]
tests/test_functionality.py::test_confidence_statistics PASSED           [ 87%]
tests/test_functionality.py::test_multiple_atom_predictions PASSED       [100%]

============================== 8 passed in 43.81s ==============================
```