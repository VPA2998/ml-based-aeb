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
