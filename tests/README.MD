University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025  

# Test Suite for GNN-Based Atom Typing Pipeline

This directory contains a comprehensive suite of automated tests designed to validate the functionality and robustness of the Graph Neural Network (GNN)-based molecular property prediction system developed for this capstone project.

## Purpose of the Test Suite

These tests are intended to ensure that all major components of the model pipeline behave as expected under a variety of conditions. 
They cover:

- Correct loading and preprocessing of molecular datasets (`.sdf`, `.json`)
- Proper initialization of the `AtomBondMPNN` model architecture
- Training execution via the `GNNTrainer` engine
- Inference correctness and label prediction accuracy
- Atom-level label mapping consistency
- Diversity of predicted labels and confidence statistics

Each test is structured to isolate potential points of failure in the pipeline and to ensure the integrity of both input processing and model outputs.

## Running the Tests

The test suite is containerized and designed to be run in a Linux-based environment using Docker. Follow the steps below to execute all tests:

```bash
cd tests
chmod +x ./build_and_test.sh
./build_and_test.sh
```

Ensure you are located in the tests/ directory before executing the script.

Prerequisites: 
- Docker must be installed and running

- Python 3.11 or higher is used inside the container

- All dependencies are pre-configured in the Docker environment

## Test Script Details
The script build_and_test.sh performs the following:

1. Builds the Docker image for the testing environment.

2. Mounts the codebase inside the container.

3. Executes all test cases using pytest.

## Sample Output
```bash
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

## Notes
- Tests are designed for modular execution and can be adapted to alternative models or datasets.

- All statistical checks use reference metrics defined in utils/metrics.py.

This test suite is a core component of the project's CI/validation framework and reflects the engineering discipline required for reproducible scientific software development.




