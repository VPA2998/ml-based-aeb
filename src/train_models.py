import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


def main():
    data_path = Path("data") / "aeb_dataset.csv"

    if not data_path.exists():
        print(f"[WARN] {data_path} not found yet. Export it from the notebook later.")
        return

    # 1) Load data
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    # 2) Clean data
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["ego_speed", "rel_speed", "distance", "brake_flag"])
    print("After cleaning:", df.shape)

    # 3) Features and target
    feature_cols = ["ego_speed", "rel_speed", "distance"]
    target_col = "brake_flag"
    X = df[feature_cols]
    y = df[target_col]

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

    # 5) Model definition + training
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 6) Evaluation
    y_pred = clf.predict(X_test)
    print("\n=== Classification report (RandomForest) ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 7) Save model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "rf_brake_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved RandomForest classifier to {model_path}")
    

if __name__ == "__main__":
    main()

