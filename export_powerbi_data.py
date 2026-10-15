
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

data_path = Path(
    r"C:\Users\pbgvh\Downloads\credit_risk_cleaned.csv"
)

df = pd.read_csv(data_path)

X = df.drop(
    columns=["id", "loan_status", "default_flag", "issue_d"]
)
y = df["default_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = joblib.load("credit_risk_model.joblib")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

report = df.loc[
    X_test.index,
    [
        "id", "loan_amnt", "int_rate", "grade",
        "purpose", "annual_inc", "loan_status", "default_flag"
    ]
].copy()

report["Predicted_Default"] = predictions
report["Default_Probability"] = probabilities

report["Risk_Category"] = pd.cut(
    probabilities,
    bins=[-0.001, 0.20, 0.50, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

output_path = Path(__file__).resolve().parent / "powerbi_credit_risk_data.csv"
report.to_csv(output_path, index=False)

print("Power BI data exported successfully!")
print("Rows:", len(report))
print("Columns:", len(report.columns))
print("Saved to:", output_path)
