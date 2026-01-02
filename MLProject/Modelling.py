import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ======================
# MLflow Autolog
# ======================
mlflow.sklearn.autolog()

# ======================
# Load data (aman untuk lokal & CI)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_preprocessed.csv")
df = pd.read_csv(DATA_PATH)

# ======================
# Target Engineering
# ======================
median_charge = df["charges"].median()
df["target"] = (df["charges"] >= median_charge).astype(int)

X = df.drop(columns=["charges", "target"])
y = df["target"]

# ======================
# Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# Preprocess
# ======================
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

# ======================
# Model pipeline
# ======================
model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

# ======================
# Train + log
# ======================
with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy_manual", acc)
    print("Accuracy:", acc)

    # ======================
    # Confusion Matrix -> Artifact
    # ======================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Insurance Charges Classification")

    cm_path = os.path.join(BASE_DIR, "training_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(cm_path)

    # ======================
    # Simpan model ke MLflow
    # ======================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # ======================
    # Metric Info JSON
    # ======================
    metric_info = {
        "accuracy": acc,
        "model_type": "LogisticRegression",
        "target_definition": "charges >= median",
        "median_charges": float(median_charge),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
    }

    metric_path = os.path.join(BASE_DIR, "metric_info.json")
    with open(metric_path, "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact(metric_path)