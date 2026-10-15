import joblib as joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
DATA_PATH = r"C:\Users\pbgvh\Downloads\credit_risk_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("CREDIT RISK & LOAN DEFAULT PREDICTION")
print("=" * 55)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# 2. Select features and target
X = df.drop(
    columns=["id", "loan_status", "default_flag", "issue_d"]
)

y = df["default_flag"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Identify numeric and categorical columns
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 5. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []
trained_models = {}

# 7. Train and evaluate models
for name, classifier in models.items():

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Default_Precision": round(precision, 4),
        "Default_Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(auc, 4)
    })

    trained_models[name] = model

    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 8. Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv("model_comparison.csv", index=False)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

# 9. Save test predictions for analysis
prediction_output = pd.DataFrame({
    "actual_default": y_test.to_numpy(),
    "logistic_probability": trained_models[
        "Logistic Regression"
    ].predict_proba(X_test)[:, 1],
    "random_forest_probability": trained_models[
        "Random Forest"
    ].predict_proba(X_test)[:, 1]
})

prediction_output.to_csv("test_predictions.csv", index=False)

print("\nFiles saved:")
print("- model_comparison.csv")
print("- test_predictions.csv")
print("\nModel training and evaluation completed!")

# 10. Save trained Logistic Regression model
joblib.dump(
    trained_models["Logistic Regression"],
    "credit_risk_model.joblib"
)

print("Saved model: credit_risk_model.joblib")