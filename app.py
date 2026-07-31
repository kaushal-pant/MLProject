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

# Class labels
CLASS_LABELS = ["Malignant", "Benign"]


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


def draw_confusion_matrix(y_true, y_pred, model_name, title_override=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=12, fontweight='bold')
    if title_override:
        ax.set_title(title_override, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)


def show_home_page():
    st.title("Machine Learning Classification Models")
    st.markdown("### Breast Cancer Classification System - ML Assignment")
    
    st.divider()
    
    # Overview
    st.subheader("Application Overview")
    st.markdown("""
    This application demonstrates multiple machine learning classification models trained on the 
    **Breast Cancer Wisconsin (Diagnostic)** dataset. The system allows you to:
    - Compare performance metrics across 5 different models
    - Upload test data and get real-time predictions
    - Visualize results with confusion matrices and classification reports
    """)
    
    st.divider()
    
    # Models and Metrics Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Models: 5 Classification Algorithms")
        st.markdown("""
        1. **Logistic Regression**
        2. **Decision Tree Classifier**
        3. **K-Nearest Neighbors (KNN) Classifier**
        4. **Naive Bayes Classifier (GaussianNB)**
        5. **Random Forest Classifier (Ensemble)**
        """)
    
    with col2:
        st.subheader("Metrics: 6 Evaluation Metrics")
        st.markdown("""
        - **Accuracy**, **AUC**, **Precision**
        - **Recall**, **F1**, **MCC**
        """)
    
    st.divider()
    
    # Dataset Information
    st.subheader("Dataset Information")
    dataset_info = load_dataset_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Instances", dataset_info['instances'])
    with col2:
        st.metric("Number of Features", dataset_info['features'])
    with col3:
        st.metric("Classes", len(dataset_info['target_classes']))
    
    st.markdown(f"""
    - **Dataset Name:** {dataset_info['dataset_name']}
    - **Source:** {dataset_info['source']}
    - **Target Classes:** {', '.join(dataset_info['target_classes']).title()}
    - **Train/Test Split:** {dataset_info['train_size']}/{dataset_info['test_size']}
    """)
    
    st.divider()
    
    # Models Training Performance
    st.subheader("Models Training Performance")
    training_metrics_df = load_training_metrics()
    
    # Display metrics table
    st.dataframe(
        training_metrics_df.style.format({
            "accuracy": "{:.4f}",
            "auc": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "mcc": "{:.4f}",
        }).background_gradient(subset=['f1'], cmap='YlGn'),
        use_container_width=True
    )
    
    # Best model highlight
    best_model = training_metrics_df.loc[training_metrics_df['f1'].idxmax(), 'model']
    best_f1 = training_metrics_df['f1'].max()
    st.success(f"**Best Performing Model:** {best_model} (F1 Score: {best_f1:.4f})")
    
    st.divider()
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. Navigate to **"Prediction"** to evaluate individual models with uploaded test data
    2. Navigate to **"Result Comparison"** to see all models compared with performance charts and confusion matrices
    3. Upload your test data CSV file (must include the target column)
    4. View predictions, metrics, confusion matrices, and classification reports
    
    **Tip:** Use the provided `test_data.csv` file for testing
    """)


def show_prediction_page():
    st.title("Model Prediction & Evaluation")
    
    dataset_info = load_dataset_info()
    
    # File Upload Section
    st.markdown("### Step 1: Upload Test Data")
    st.info("Upload a CSV file containing all required feature columns and the target column")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
    
    # Model Selection
    st.markdown("### Step 2: Select Classification Model")
    selected_model = st.selectbox(
        "Choose a model",
        list(model_file_map().keys()),
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.warning("Please upload test_data.csv to continue")
        st.stop()
    
    # Load model
    payload = joblib.load(model_file_map()[selected_model])
    model = payload["model"]
    feature_names = payload["feature_names"]
    
    # Load and validate data
    test_df = pd.read_csv(uploaded_file)
    
    missing = [feature for feature in feature_names if feature not in test_df.columns]
    if missing:
        st.error("Missing required feature columns. Please load correct test data.")
        st.stop()
    
    x_test = test_df[feature_names].copy()
    y_test = test_df["target"] if "target" in test_df.columns else None
    
    if y_test is None:
        st.error("Missing required target column. Please load correct test data.")
        st.stop()
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = None
    
    # Display Results
    st.divider()
    st.markdown("### Step 3: Results")
    
    # Metrics
    st.markdown(f"#### Evaluation Metrics for **{selected_model}**")
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics_cols = [col1, col2, col3, col4, col5, col6]
    
    for idx, (name, value) in enumerate(metrics.items()):
        with metrics_cols[idx]:
            st.metric(name, f"{value:.4f}")
    
    st.divider()
    
    # Predictions Table
    with st.expander("View Predictions (First 20 rows)", expanded=False):
        result_df = test_df.copy()
        result_df["prediction"] = y_pred
        result_df["prediction_label"] = result_df["prediction"].map({0: "Malignant", 1: "Benign"})
        result_df["actual_label"] = result_df["target"].map({0: "Malignant", 1: "Benign"})
        st.dataframe(result_df[["actual_label", "prediction_label"] + list(feature_names[:5])].head(20))
    
    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    draw_confusion_matrix(y_test, y_pred, selected_model)
    
    st.divider()
    
    # Classification Report
    st.markdown(f"#### Detailed Classification Report - {selected_model}")
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.dataframe(
        report_df.style.format("{:.4f}").background_gradient(subset=['f1-score'], cmap='RdYlGn'),
        use_container_width=True
    )


def show_comparison_page():
    st.title("Result Comparison")
    st.markdown("### Compare all models performance on the same test dataset")
    
    st.divider()
    
    # File Upload Section
    st.markdown("#### Upload Test Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], key="comparison_upload")
    
    if uploaded_file is None:
        st.warning("Please upload test_data.csv to see model comparison")
        st.stop()
    
    # Load test data
    test_df = pd.read_csv(uploaded_file)
    
    if "target" not in test_df.columns:
        st.error("Missing required target column. Please load correct test data.")
        st.stop()
    
    y_test = test_df["target"]
    
    # Dictionary to store results and predictions
    all_results = {}
    all_predictions = {}
    
    # Run predictions for all models
    with st.spinner("Running predictions on all models..."):
        for model_name, model_path in model_file_map().items():
            try:
                payload = joblib.load(model_path)
                model = payload["model"]
                feature_names = payload["feature_names"]
                
                # Check if all features are present
                missing = [f for f in feature_names if f not in test_df.columns]
                if missing:
                    st.error("Missing required feature columns. Please load correct test data.")
                    st.stop()
                
                x_test = test_df[feature_names].copy()
                y_pred = model.predict(x_test)
                
                # Get probabilities if available
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(x_test)[:, 1]
                else:
                    y_proba = None
                
                # Compute metrics
                metrics = compute_metrics(y_test, y_pred, y_proba)
                all_results[model_name] = metrics
                all_predictions[model_name] = y_pred
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
    
    if not all_results:
        st.error("No models could be evaluated. Please check your data.")
        st.stop()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    comparison_df.index.name = "Model"
    comparison_df = comparison_df.reset_index()
    
    st.divider()
    
    # Display metrics table
    st.markdown("#### Performance Metrics Comparison")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }).background_gradient(subset=['F1'], cmap='YlGn'),
        use_container_width=True
    )
    
    st.divider()
    
    # Bar charts for each metric
    st.markdown("#### Performance Comparison Charts")
    
    metrics_to_plot = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    
    # Create 2x3 grid for plots with soothing colors
    for i in range(0, len(metrics_to_plot), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(metrics_to_plot):
                metric_name = metrics_to_plot[i + j]
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    
                    x_pos = range(len(comparison_df))
                    values = comparison_df[metric_name]
                    
                    # Soothing pastel colors
                    colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3']
                    
                    bars = ax.bar(x_pos, values, color=colors[:len(comparison_df)], 
                                  edgecolor='gray', linewidth=1, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    ax.set_xlabel('Model', fontsize=10, fontweight='bold')
                    ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
                    ax.set_title(f'{metric_name} Comparison', fontsize=11, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=8)
                    ax.set_ylim(0, min(1.0, values.max() * 1.15))
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    st.divider()
    
    # Confusion Matrices for all models
    st.markdown("#### Confusion Matrices for All Models")
    
    # Create grid for confusion matrices (3 per row)
    model_names = list(all_predictions.keys())
    for i in range(0, len(model_names), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(model_names):
                model_name = model_names[i + j]
                y_pred = all_predictions[model_name]
                
                with cols[j]:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax,
                                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
                    ax.set_xlabel("Predicted", fontsize=10, fontweight='bold')
                    ax.set_ylabel("Actual", fontsize=10, fontweight='bold')
                    ax.set_title(f"{model_name}", fontsize=11, fontweight='bold', pad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    st.divider()
    
    # Best model summary
    st.markdown("#### Summary")
    best_f1_idx = comparison_df['F1'].idxmax()
    best_model = comparison_df.loc[best_f1_idx, 'Model']
    best_f1 = comparison_df.loc[best_f1_idx, 'F1']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Best Model (by F1 Score):** {best_model} (F1: {best_f1:.4f})")
    with col2:
        best_acc_idx = comparison_df['Accuracy'].idxmax()
        best_acc_model = comparison_df.loc[best_acc_idx, 'Model']
        best_acc = comparison_df.loc[best_acc_idx, 'Accuracy']
        st.info(f"**Best Model (by Accuracy):** {best_acc_model} (Acc: {best_acc:.4f})")


def main():
    st.set_page_config(
        page_title="ML Assignment 2 - Breast Cancer Classification",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Prediction", "Result Comparison"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Machine Learning Assignment 2**
    
    Name: Kaushal Pant  
    BITS ID: 2025AC05981
    
    Dataset: Breast Cancer Wisconsin
    """)
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Prediction":
        show_prediction_page()
    else:
        show_comparison_page()


if __name__ == "__main__":
    main()
