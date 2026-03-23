import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_split(path):
    split_df = pd.read_csv(path)
    X = split_df.drop(columns=["label"])
    y = split_df["label"]
    return X, y


def main():
    X_train, y_train = load_split(PROCESSED_DIR / "train_data.csv")
    X_val, y_val = load_split(PROCESSED_DIR / "val_data.csv")
    X_test, y_test = load_split(PROCESSED_DIR / "test_data.csv")

    # Tune on validation set only.
    search_space = [
        {"hidden_layer_sizes": (6,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (12,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (24,), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (12, 6), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (24, 12), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (12,), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (12,), "alpha": 0.0001, "learning_rate_init": 0.005},
    ]

    best_params = None
    best_val_f1 = -1.0
    best_val_acc = -1.0

    print("Validation tuning results:")
    for params in search_space:
        candidate = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        candidate.fit(X_train, y_train)
        y_val_pred = candidate.predict(X_val)

        val_f1 = f1_score(y_val, y_val_pred, average="macro")
        val_acc = accuracy_score(y_val, y_val_pred)

        print(
            f"  params={params} | val_macro_f1={val_f1:.4f} | val_acc={val_acc:.4f}"
        )

        if (val_f1 > best_val_f1) or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_params = params

    print("\nSelected params from validation set:")
    print(best_params)
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Retrain best model on train+val before final test evaluation.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_model = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=800,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    final_model.fit(X_train_val, y_train_val)

    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)

    print("\nFinal test evaluation (unseen test set):")
    print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test macro-F1: {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\nSoftmax Probabilities (first 5 samples):")
    print(y_test_proba[:5])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACTS_DIR / "mlp_model_final.pkl")
    # Keep legacy filename for runtime compatibility.
    joblib.dump(final_model, ARTIFACTS_DIR / "mlp_model.pkl")
    with open(ARTIFACTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nSaved: artifacts/mlp_model_final.pkl")
    print("Saved: artifacts/mlp_model.pkl")
    print("Saved: artifacts/best_params.json")


if __name__ == "__main__":
    main()
