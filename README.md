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
---
- (optional but recommended):

text
### (Recommended) Use virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # will be added later
text

For now you can skip committing this if you’re tired; the key is:

- Always run project commands like:
  ```bash
  cd ~/ml-based-aeb
  source .venv/bin/activate
  python src/train_models.py
When done, deactivate venv with:

bash
deactivate
If you want to continue later, next steps will be:

export a CSV from the notebook,

point train_models.py to it,

move actual training code over.

---

1. Create a Python environment (example with conda):

   ```bash
   conda create -n aeb python=3.8
   conda activate aeb

2. Install basic dependencies 

   ```bash
    pip install scikit-learn pandas matplotlib seaborn joblib

3. Open the notebook

   ```bash
    jupyter notebook 
    notebooks/ML_Based_AEB_Project_with_MetaDrive_Simulator.ipynb

4. Run the cells in order to:

- Collect simulation data (in Colab / MetaDrive setup).

- Train models.

- View evaluation metrics and plots.
