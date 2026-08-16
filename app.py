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


@st.cache_data
def load_full_dataset():
    dataset_path = ROOT / "dataset_used.csv"
    return pd.read_csv(dataset_path)


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
    
    # Colored boxes for Dataset, Models, and Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #D6EAF8; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB;">
            <h4 style="color: #1F618D; margin-top: 0;">Dataset: Breast Cancer Classification</h4>
            <ul style="color: #1F618D;">
                <li><b>Instances:</b> 569</li>
                <li><b>Features:</b> 30</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #FEF9E7; padding: 20px; border-radius: 10px; border-left: 5px solid #F39C12;">
            <h4 style="color: #9A7D0A; margin-top: 0;">Models: 5 Classification Algorithms</h4>
            <ul style="color: #9A7D0A;">
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>K-Nearest Neighbors (KNN)</li>
                <li>Naive Bayes (GaussianNB)</li>
                <li>Random Forest (Ensemble)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #D5F4E6; padding: 20px; border-radius: 10px; border-left: 5px solid #28B463;">
            <h4 style="color: #186A3B; margin-top: 0;">Metrics: 6 Evaluation Metrics</h4>
            <ul style="color: #186A3B;">
                <li>Accuracy, AUC, Precision</li>
                <li>Recall, F1, MCC</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Models Training Performance
    st.subheader("Models Training Performance")
    training_metrics_df = load_training_metrics()
    
    # Display metrics table with 1-based numbering, no color coding
    display_df = training_metrics_df.copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(
        display_df.style.format({
            "accuracy": "{:.4f}",
            "auc": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "mcc": "{:.4f}",
        }),
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
    1. Navigate to **"Dataset Overview"** to explore the dataset with samples and statistics
    2. Navigate to **"Prediction"** to evaluate individual models with uploaded test data
    3. Navigate to **"Result Comparison"** to see all models compared with performance charts and confusion matrices
    
    **Tip:** Use the provided `test_data.csv` file for testing
    """)


def show_dataset_page():
    st.title("Dataset Overview")
    st.markdown("### Breast Cancer Wisconsin (Diagnostic) Dataset")
    
    st.divider()
    
    # Load dataset
    dataset_df = load_full_dataset()
    dataset_info = load_dataset_info()
    
    # Dataset Overview
    st.subheader("Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Instances", dataset_info['instances'])
    with col2:
        st.metric("Features", dataset_info['features'])
    with col3:
        st.metric("Classes", len(dataset_info['target_classes']))
    with col4:
        st.metric("Missing Values", 0)
    
    st.markdown(f"""
    - **Dataset Name:** {dataset_info['dataset_name']}
    - **Source:** {dataset_info['source']}
    - **Task Type:** Binary Classification
    - **Target Classes:** {', '.join(dataset_info['target_classes']).title()}
    - **Train/Test Split:** {dataset_info['train_size']}/{dataset_info['test_size']}
    """)
    
    st.divider()
    
    # Class Distribution
    st.subheader("Class Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        class_counts = dataset_df['target'].value_counts().sort_index()
        st.markdown(f"""
        - **Malignant (0):** {class_counts[0]} instances ({class_counts[0]/len(dataset_df)*100:.1f}%)
        - **Benign (1):** {class_counts[1]} instances ({class_counts[1]/len(dataset_df)*100:.1f}%)
        """)
    
    with col2:
        # Simple bar chart for class distribution
        fig, ax = plt.subplots(figsize=(5, 3))
        class_counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'], ax=ax, edgecolor='black')
        ax.set_xticklabels(['Malignant', 'Benign'], rotation=0)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Class Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.divider()
    
    # Dataset Sample
    st.subheader("Dataset Sample")
    st.markdown("**First 10 rows of the dataset:**")
    
    # Show first 10 rows with target label
    sample_df = dataset_df.head(10).copy()
    sample_df['target_label'] = sample_df['target'].map({0: 'Malignant', 1: 'Benign'})
    
    # Reorder columns to show target_label first
    cols = ['target_label'] + [col for col in sample_df.columns if col not in ['target', 'target_label']]
    st.dataframe(sample_df[cols], use_container_width=True, height=400)
    
    st.divider()
    
    # Feature Statistics
    st.subheader("Feature Statistics")
    st.markdown("**Summary statistics for all features:**")
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in dataset_df.columns if col != 'target']
    stats_df = dataset_df[feature_cols].describe().T
    
    # Round to 4 decimal places
    stats_df = stats_df.round(4)
    
    # Display with color gradient
    st.dataframe(
        stats_df.style.background_gradient(subset=['mean'], cmap='Blues'),
        use_container_width=True,
        height=600
    )
    
    st.divider()
    
    # Feature Categories
    st.subheader("Feature Categories")
    st.markdown("""
    The 30 features are organized into three categories based on the cell nucleus measurements:
    
    **1. Mean Features (10 features):**
    - mean radius, mean texture, mean perimeter, mean area, mean smoothness
    - mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension
    
    **2. Standard Error Features (10 features):**
    - radius error, texture error, perimeter error, area error, smoothness error
    - compactness error, concavity error, concave points error, symmetry error, fractal dimension error
    
    **3. Worst/Largest Features (10 features):**
    - worst radius, worst texture, worst perimeter, worst area, worst smoothness
    - worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension
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


@st.cache_data
def load_default_test_data():
    test_path = ROOT / "test_data.csv"
    return pd.read_csv(test_path)


def show_comparison_page():
    st.title("Result Comparison")
    st.markdown("### Compare all models performance on the same test dataset")
    
    st.divider()
    
    # Data source selection
    st.markdown("#### Select Test Data Source")
    data_source = st.radio(
        "Data source",
        ["Use pre-loaded test data", "Upload custom CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    if data_source == "Upload custom CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], key="comparison_upload")
        if uploaded_file is None:
            st.info("Please upload a CSV file to see model comparison, or switch to the pre-loaded test data option.")
            st.stop()
        test_df = pd.read_csv(uploaded_file)
    else:
        test_df = load_default_test_data()
        st.success(f"Using pre-loaded test data — {len(test_df)} samples (test_data.csv)")
    
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
    
    # Display metrics table — plain, no color coding
    st.markdown("#### Performance Metrics Comparison")
    plain_df = comparison_df.copy()
    plain_df.index = range(1, len(plain_df) + 1)
    st.dataframe(
        plain_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }),
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
        ["Home", "Dataset Overview", "Prediction", "Result Comparison"],
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
    elif page == "Dataset Overview":
        show_dataset_page()
    elif page == "Prediction":
        show_prediction_page()
    else:
        show_comparison_page()


if __name__ == "__main__":
    main()
