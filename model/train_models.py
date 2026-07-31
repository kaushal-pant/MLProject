import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "model"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42
        ),
    }


def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def main():
    # Load Breast Cancer dataset
    data = load_breast_cancer(as_frame=True)
    x = data.data.copy()
    y = data.target.copy()
    
    print(f"Dataset loaded: {len(x)} instances, {len(x.columns)} features")

    # Keep a reproducible test set that can be uploaded in Streamlit app.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    test_df = x_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    full_df = x.copy()
    full_df["target"] = y.values
    full_df.to_csv(ROOT / "dataset_used.csv", index=False)

    models = build_models()
    all_metrics = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test)
        all_metrics.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
            }
        )

        model_payload = {
            "model": model,
            "feature_names": list(x.columns),
            "target_name": "target",
        }
        model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_filename = model_filename.replace("__", "_")
        joblib.dump(model_payload, MODELS_DIR / f"{model_filename}.joblib")

        with open(ARTIFACTS_DIR / f"{model_filename}_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    metrics_df = pd.DataFrame(all_metrics).sort_values(by="f1", ascending=False)
    metrics_df.to_csv(ROOT / "model_metrics.csv", index=False)

    with open(ROOT / "dataset_info.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
                "source": "UCI Machine Learning Repository (available via scikit-learn)",
                "instances": int(full_df.shape[0]),
                "features": int(x.shape[1]),
                "target_classes": ["malignant", "benign"],
                "train_size": int(x_train.shape[0]),
                "test_size": int(x_test.shape[0]),
            },
            fp,
            indent=2,
        )

    print("Training complete.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    np.random.seed(42)
    main()
