# HW1: Binary Classification with Gradient Descent (PyTorch)

This project implements gradient-descent-based optimization using PyTorch automatic differentiation
for a binary classification problem.

## Project Structure
- `src/neonunicorn/deepl/two_layer_binary_classification.py`  
  Contains the function:
  `binary_classification(d, n, epochs=10000, h=0.001)`

- `scripts/binaryclassification_impl.py`  
  Demonstrates how to call the function, plot the loss vs epochs,
  and save the loss plot as a timestamped PDF.

## Setup
From the project root directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch matplotlib

