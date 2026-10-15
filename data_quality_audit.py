
import pandas as pd
from pathlib import Path

base = Path(r"C:\Users\pbgvh\Downloads\CreditRiskProject")

files = {
    "Cleaned training data": (
        Path(r"C:\Users\pbgvh\Downloads")
        / "credit_risk_cleaned.csv"
    ),
    "Power BI predictions": base / "powerbi_credit_risk_data.csv",
}

for name, path in files.items():
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    try:
        df = pd.read_csv(path, on_bad_lines="error")
    except Exception as e:
        print("CSV READ ERROR:", repr(e))
        continue

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate rows:", df.duplicated().sum())

    if "id" in df.columns:
        print("Duplicate loan IDs:", df["id"].duplicated().sum())

    print("Missing cells:", int(df.isna().sum().sum()))

    if "default_flag" in df.columns:
        print("\nActual target counts:")
        print(df["default_flag"].value_counts(dropna=False))

    if "Predicted_Default" in df.columns:
        print("\nPrediction counts:")
        print(df["Predicted_Default"].value_counts(dropna=False))

    if "Default_Probability" in df.columns:
        p = df["Default_Probability"]
        print("Invalid probabilities:", int((~p.between(0, 1)).sum()))
        print("Missing probabilities:", int(p.isna().sum()))

print("\nAudit finished.")
