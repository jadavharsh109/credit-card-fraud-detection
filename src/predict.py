import os
import joblib
import pandas as pd
import numpy as np

# Load serialized model and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'random_forest_fraud_model.joblib')
scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
sample_path = os.path.join(base_dir, 'data', 'creditcard_sample.csv')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("Error: Trained model or scaler not found in models/. Run train_and_evaluate.py first.")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Loaded Random Forest Model and StandardScaler successfully.\n")

if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    X_sample = sample_df.drop('Class', axis=1, errors='ignore')
    y_actual = sample_df['Class'] if 'Class' in sample_df.columns else None

    # Scale Time and Amount
    X_scaled = X_sample.copy()
    X_scaled[['Time', 'Amount']] = scaler.transform(X_sample[['Time', 'Amount']])

    # Predict
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print(f"Inference on {len(sample_df)} sample transactions:")
    print("-" * 50)
    fraud_detected = sum(predictions == 1)
    print(f"Total Transactions Scanned : {len(sample_df)}")
    print(f"Flagged as Fraudulent      : {fraud_detected}")
    print(f"Approved as Genuine        : {len(sample_df) - fraud_detected}")
    if y_actual is not None:
        actual_fraud = sum(y_actual == 1)
        print(f"Ground Truth Frauds in Set : {actual_fraud}")
    print("-" * 50)
