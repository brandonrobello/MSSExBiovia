University of California Berkeley x BIOVIA  
MSSE Capstone Project - Spring 2025  
Authors: Brandon Robello, Jeremy Millford, Yara Khoury  
Created: Wednesday April 9th 2025  
Last Updated: Wednesday April 9th 2025  

# Atom Type Classification Project

This repository contains code, data, and experiments for predicting atom types using both traditional machine learning and graph-based methods.

## Project Structure

| Folder | Description |
|--------|-------------|
| `atoMLtype/` | Core module containing all code related to data processing, featurization, and modeling. |
| `data/` | Contains molecular datasets and atom-level labels used during training and evaluation. |
| `.ipynb` files | Notebooks used to test, analyze, or demonstrate model behavior. |

## Dependencies

This project uses:

- Python 3.x
- RDKit
- NumPy
- Pandas
- scikit-learn
- PyTorch
- PyTorch Geometric
- Matplotlib
- Seaborn

## Usage

To use this repository:
1. Clone the repo and set up your environment.
2. Run one of the notebooks for testing or model evaluation.
3. Refer to the individual `README.md` files in each subfolder for specific usage and structure.

## Notes

- Molecules are stored in SDF format.
- Atom type labels are stored in JSON format.
- Models are trained and evaluated on a per-atom basis.

