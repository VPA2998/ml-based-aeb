# ML-based Automatic Emergency Braking (AEB)

This project trains machine learning models for Automatic Emergency Braking using the MetaDrive simulator.

## What this project does

- Collects driving scenarios with a lead vehicle (ego speed, relative speed, distance, TTC).
- Trains:
  - A **classification model** (brake vs. no-brake).
  - A **regression model** (brake intensity between 0 and 1).
- Evaluates models with metrics, confusion matrices, and plots inside the notebook.
- Prepares for closed-loop AEB behavior where the model controls braking in simulation.

## Repository structure

- `notebooks/ML_Based_AEB_Project_with_MetaDrive_Simulator.ipynb` – full experiment in Colab style.
- `src/train_models.py` – script where the training pipeline will be extracted and cleaned.
---

## How to run

### Option 1 – Recommended: virtual environment (local)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn pandas matplotlib seaborn joblib
python src/train_models.py
 ```

when you're done:

```bash
deactivate
 ```
---
### Option 2 – Using conda

```bash
conda create -n aeb python=3.8
conda activate aeb
pip install scikit-learn pandas matplotlib seaborn joblib
jupyter notebook notebooks/ML_Based_AEB_Project_with_MetaDrive_Simulator.ipynb
 ```

Then run the notebook cells in order to:

- Collect simulation data (MetaDrive / Colab setup).

- Train models.

- View evaluation metrics and plots.