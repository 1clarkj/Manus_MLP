import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the datasets
df1 = pd.read_csv('position1_pointing.csv')
df2 = pd.read_csv('position2.csv')

# Add labels
df1['label'] = 0
df2['label'] = 1

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)


# Separate features and labels
features = combined_df.drop(columns=['label'])
labels = combined_df['label']

# Split the data into training (70%), test (15%), and validation (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the training features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Scale the test and validation features using the same scaler
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, "minmax_scaler.pkl")

# Convert the scaled features back into DataFrames
train_df = pd.DataFrame(X_train_scaled, columns=features.columns)
test_df = pd.DataFrame(X_test_scaled, columns=features.columns)
val_df = pd.DataFrame(X_val_scaled, columns=features.columns)

# Add the labels back to the scaled DataFrames
train_df['label'] = y_train.values
test_df['label'] = y_test.values
val_df['label'] = y_val.values

# Save the datasets to CSV files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)



