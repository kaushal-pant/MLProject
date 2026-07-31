import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "model"
METRICS_PATH = ROOT / "model_metrics.csv"
DATASET_INFO_PATH = ROOT / "dataset_info.json"


@st.cache_data
def load_dataset_info():
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data
def load_training_metrics():
    return pd.read_csv(METRICS_PATH)


def model_file_map():
    return {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Decision Tree": MODELS_DIR / "decision_tree.joblib",
        "KNN": MODELS_DIR / "knn.joblib",
        "Naive Bayes": MODELS_DIR / "naive_bayes.joblib",
        "Random Forest (Ensemble)": MODELS_DIR / "random_forest_ensemble.joblib",
    }


def compute_metrics(y_true, y_pred, y_proba=None):
    auc_value = roc_auc_score(y_true, y_proba) if y_proba is not None else roc_auc_score(y_true, y_pred)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classification App", layout="wide")
    st.title("Machine Learning Assignment 2")
    st.caption("Breast Cancer Classification using multiple ML models")

    dataset_info = load_dataset_info()
    training_metrics_df = load_training_metrics()

    with st.expander("Dataset Summary", expanded=True):
        st.write(
            f"Dataset: {dataset_info['dataset_name']} | Instances: {dataset_info['instances']} | "
            f"Features: {dataset_info['features']}"
        )
        st.write(f"Source: {dataset_info['source']}")

    st.subheader("1) Upload Test Data (CSV)")
    st.write("Use the provided test_data.csv (contains all required feature columns).")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.subheader("2) Select Model")
    selected_model = st.selectbox("Choose a model", list(model_file_map().keys()))

    st.subheader("3) Metrics on Training Split (Reference)")
    st.dataframe(training_metrics_df.style.format({
        "accuracy": "{:.4f}",
        "auc": "{:.4f}",
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1": "{:.4f}",
        "mcc": "{:.4f}",
    }))

    if uploaded_file is None:
        st.info("Upload test_data.csv to run predictions and see confusion matrix/report.")
        return

    payload = joblib.load(model_file_map()[selected_model])
    model = payload["model"]
    feature_names = payload["feature_names"]

    test_df = pd.read_csv(uploaded_file)

    missing = [feature for feature in feature_names if feature not in test_df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        return

    x_test = test_df[feature_names].copy()
    y_test = test_df["target"] if "target" in test_df.columns else None

    y_pred = model.predict(x_test)
    result_df = test_df.copy()
    result_df["prediction"] = y_pred

    st.subheader("4) Predictions")
    st.dataframe(result_df.head(20))

    if y_test is None:
        st.warning("No target column found in uploaded file. Metrics and confusion matrix need true labels.")
        return

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metric_cols = st.columns(6)
    for idx, (name, value) in enumerate(metrics.items()):
        metric_cols[idx].metric(name, f"{value:.4f}")

    st.subheader("5) Confusion Matrix")
    draw_confusion_matrix(y_test, y_pred)

    st.subheader("6) Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"))


if __name__ == "__main__":
    main()
