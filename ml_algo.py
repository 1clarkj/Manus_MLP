import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from sklearn.neural_network import MLPClassifier
from scipy.special import softmax

# Load the training and test datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and labels
X_train = train_data.drop(columns=["label"])
y_train = train_data["label"]

X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]

# Step 1: Train an MLP Classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)

# Step 2: Test the Model
y_pred = mlp_model.predict(X_test)

# Step 3: Get Softmax Probabilities
y_proba = mlp_model.predict_proba(X_test)  # This gives the softmax probabilities
print("\nSoftmax Probabilities (first 5 samples):")
print(y_proba[:5])

# Step 4: Evaluate the Model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(mlp_model, "mlp_model1.pkl")
