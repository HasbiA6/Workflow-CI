import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def build_pipeline() -> Pipeline:
    """Build preprocessing + regression pipeline."""
    num_features = ["age", "bmi", "children"]
    cat_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_preprocessed.csv",
        help="Path to preprocessed dataset CSV",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="charges",
        help="Target column name",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split",
    )
    args = parser.parse_args()

    # -------------------------
    # MLflow setup (CI-friendly)
    # -------------------------
    # Jangan set_tracking_uri di sini. Biarkan env MLFLOW_TRACKING_URI dari workflow yang mengatur.
    mlflow.set_experiment("Basic_Insurance_Regression")

    # Autolog yang stabil untuk MLflow Projects (CI)
    # log_models=True penting agar folder model/ muncul
    mlflow.autolog(log_models=True)

    # -------------------------
    # Load data
    # -------------------------
    data = pd.read_csv(args.data_path)

    if args.target_col not in data.columns:
        raise ValueError(
            f"Target column '{args.target_col}' tidak ditemukan. Kolom tersedia: {list(data.columns)}"
        )

    X = data.drop(args.target_col, axis=1)
    y = data[args.target_col]

    # -------------------------
    # Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # -------------------------
    # Build + train model
    # -------------------------
    model = build_pipeline()
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluate
    # -------------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Manual logging (opsional tapi bagus untuk reviewer)
    mlflow.log_metric("mse_test", float(mse))
    mlflow.log_metric("r2_test", float(r2))

    print("Training selesai.")
    print(f"MSE Test: {mse:.4f}")
    print(f"R2  Test: {r2:.4f}")


if __name__ == "__main__":
    main()
