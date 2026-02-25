import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from pathlib import Path


def main():
    data_path = Path("data") / "aeb_dataset.csv"

    if not data_path.exists():
        print(f"[WARN] {data_path} not found yet. Export it from the notebook later.")
        return

    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
