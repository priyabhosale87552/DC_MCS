# EDI-1 Project

Simple project for training and evaluating models on the BBC dataset.

Repository structure

- backend/: training and evaluation scripts
- dataset/: data files (bbc.csv)
- frontend/: minimal app

Quick start

1. Create a virtual environment and activate it (Windows PowerShell):

   python -m venv .venv
   .venv\\Scripts\\Activate.ps1

2. Install dependencies:

   pip install -r requirements.txt

3. Run training or evaluation (examples):

   python backend/train_model.py
   python backend/evaluate.py

4. Run the frontend app:

   python frontend/app.py

Data

The dataset is at `dataset/bbc.csv`.

Notes

- Adjust commands for your OS or environment as needed.
