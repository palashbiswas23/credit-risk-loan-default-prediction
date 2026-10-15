
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("credit_risk_model.joblib")

# Load test data
df = pd.read_csv(
    r"C:\Users\pbgvh\Downloads\credit_risk_cleaned.csv"
)

# Select one record and remove target/leakage columns
X = df.drop(
    columns=["id", "loan_status", "default_flag", "issue_d"]
)

sample = X.iloc[[0]]

# Predict loan default
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "Default Risk" if prediction == 1 else "No Default Risk")
print(f"Estimated Default Probability: {probability:.2%}")
