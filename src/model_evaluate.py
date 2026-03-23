import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MLP on val or test split.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    args = parser.parse_args()

    split_path = PROCESSED_DIR / f"{args.split}_data.csv"
    model_path = ARTIFACTS_DIR / "mlp_model.pkl"

    data = pd.read_csv(split_path)
    X = data.drop(columns=["label"])
    y = data["label"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    print(f"\nEvaluating split: {args.split}")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y, y_pred, average='macro'):.4f}")

    print("\nPredicted Classes (first 10):")
    print(y_pred[:10])

    print("\nSoftmax Probabilities (first 5):")
    print(y_proba[:5])

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    main()
