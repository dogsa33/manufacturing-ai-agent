from pathlib import Path

import duckdb
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "database" / "manufacturing.duckdb"
MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.pkl"


# =========================================================
# 1. Load data
# =========================================================

con = duckdb.connect(str(DB_PATH), read_only=True)

query = """
SELECT
    Type,
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    Torque,
    "Tool wear",
    "Machine failure"
FROM manufacturing_data
"""

df = con.execute(query).fetchdf()
con.close()


print("=" * 60)
print("1. DATA LOAD")
print("=" * 60)

print("Shape:", df.shape)

print("\nTarget distribution:")
print(df["Machine failure"].value_counts())


# =========================================================
# 2. Feature / Target split
# =========================================================

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

TARGET = "Machine failure"

X = df[FEATURES]
y = df[TARGET]


# =========================================================
# 3. Train / Test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


print("\n" + "=" * 60)
print("2. TRAIN / TEST SPLIT")
print("=" * 60)

print("Train samples:", len(X_train))
print("Test samples :", len(X_test))

print("\nTrain failure rate:")
print(round(y_train.mean() * 100, 2), "%")

print("Test failure rate:")
print(round(y_test.mean() * 100, 2), "%")


# =========================================================
# 4. Preprocessing
# =========================================================

categorical_features = ["Type"]

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


# =========================================================
# 5. Model
# =========================================================

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# =========================================================
# 6. Train
# =========================================================

print("\n" + "=" * 60)
print("3. MODEL TRAINING")
print("=" * 60)

pipeline.fit(X_train, y_train)

print("Training complete.")


# =========================================================
# 7. Prediction
# =========================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# =========================================================
# 8. Evaluation
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


print("\n" + "=" * 60)
print("4. MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0,
    )
)


# =========================================================
# 9. Save model
# =========================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)


print("\n" + "=" * 60)
print("5. MODEL SAVED")
print("=" * 60)

print("Model path:", MODEL_PATH)