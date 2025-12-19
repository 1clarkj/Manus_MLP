import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# === Load Validation Data ===
validate_data = pd.read_csv("val_data.csv")  # Replace with your actual file
X_val = validate_data.drop(columns=["label"])
y_val = validate_data["label"]

# === Load Trained MLP Model ===
mlp_model = joblib.load("mlp_model.pkl")

# === Make Predictions ===
y_pred = mlp_model.predict(X_val)
y_proba = mlp_model.predict_proba(X_val)

# === Output Results ===
print("\nPredicted Classes (first 10):")
print(y_pred[:10])

print("\nSoftmax Probabilities (first 5):")
print(y_proba[:5])

# === Evaluate the Model ===
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))