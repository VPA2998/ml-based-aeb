import joblib
import pandas as pd
from pathlib import Path

def main():
    model_path = Path("models") / "rf_brake_classifier.joblib"
    clf = joblib.load(model_path)

    # Example input: ego_speed, rel_speed, distance
    sample_df = pd.DataFrame(
        [[20.0, -5.0, 15.0]],
        columns=["ego_speed", "rel_speed", "distance"]
    )

    pred = clf.predict(sample_df)[0]
    proba = clf.predict_proba(sample_df)[0]

    print("Prediction (0=no brake, 1=brake):", pred)
    print("Probabilities [no_brake, brake]:", proba)

if __name__ == "__main__":
    main()
import joblib
import pandas as pd
from pathlib import Path


FEATURE_COLS = ["ego_speed", "rel_speed", "distance"]


def load_models():
    models_dir = Path("models")
    clf_path = models_dir / "rf_brake_classifier.joblib"
    regr_path = models_dir / "rf_brake_regressor.joblib"

    clf = joblib.load(clf_path)
    regr = joblib.load(regr_path)

    return clf, regr


def run_demo_scenario(ego_speed, rel_speed, distance):
    clf, regr = load_models()

    sample_df = pd.DataFrame(
        [[ego_speed, rel_speed, distance]],
        columns=FEATURE_COLS,
    )

    # Classification: should we brake?
    pred_class = clf.predict(sample_df)[0]
    proba = clf.predict_proba(sample_df)[0]

    # Regression: how strong to brake?
    brake_value = regr.predict(sample_df)[0]

    print("Input scenario:")
    print(f"  ego_speed = {ego_speed}")
    print(f"  rel_speed = {rel_speed}")
    print(f"  distance  = {distance}")

    print("\nClassifier output:")
    print(f"  Prediction (0=no brake, 1=brake): {pred_class}")
    print(f"  Probabilities [no_brake, brake]: {proba}")

    print("\nRegressor output:")
    print(f"  Predicted brake_value: {brake_value:.3f}")


def main():
    # Example scenario – you can tweak these values freely
    run_demo_scenario(
        ego_speed=20.0,
        rel_speed=-5.0,
        distance=15.0,
    )


if __name__ == "__main__":
    main()
