import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib


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

    # 3) Features
    feature_cols = ["ego_speed", "rel_speed", "distance"]

    # === Classification: brake_flag ===
    target_col = "brake_flag"
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Classification report (RandomForest) ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # === Regression: brake_value ===
    reg_target_col = "brake_value"
    X_reg = df[feature_cols]
    y_reg = df[reg_target_col]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    regr = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    regr.fit(X_reg_train, y_reg_train)

    y_reg_pred = regr.predict(X_reg_test)

    print("\n=== Regression report (RandomForestRegressor for brake_value) ===")
    print(f"MSE: {mean_squared_error(y_reg_test, y_reg_pred):.4f}")
    print(f"R2 : {r2_score(y_reg_test, y_reg_pred):.4f}")

    # === Save models ===
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    clf_path = models_dir / "rf_brake_classifier.joblib"
    joblib.dump(clf, clf_path)
    print(f"Saved RandomForest classifier to {clf_path}")

    regr_path = models_dir / "rf_brake_regressor.joblib"
    joblib.dump(regr, regr_path)
    print(f"Saved RandomForest regressor to {regr_path}")


if __name__ == "__main__":
    main()
