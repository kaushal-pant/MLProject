# Machine Learning Assignment 2 - Classification and Streamlit Deployment

**Name:** Kaushal Pant  
**BITS ID:** 2025AC05981

## a) Problem statement
Build and evaluate multiple machine learning classification models on a single dataset, then deploy an interactive Streamlit app to visualize predictions and model performance metrics.

## b) Dataset description
- **Dataset name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (available via scikit-learn)
- **Task type:** Binary classification (`malignant` vs `benign`)
- **Instances:** 569
- **Features:** 30 (mean radius, mean texture, mean perimeter, mean area, mean smoothness, and 25 other continuous features computed from digitized images of breast mass)
- **Why this dataset:** It satisfies assignment constraints (minimum 12 features and 500+ instances), is widely recognized in ML literature, contains no missing values, and is ideal for comparing classification algorithms.

## c) GitHub Repository Link
- **Repository:** ADD_YOUR_GITHUB_REPO_LINK_HERE

## d) Models used and performance comparison
### Implemented models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier (GaussianNB)
5. Random Forest Classifier (Ensemble)

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Decision Tree | 0.9123 | 0.9147 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model with highest accuracy (0.9825), F1 (0.9861), and MCC (0.9623); excellent linear separator for this dataset with perfect precision-recall balance. |
| KNN | Exceptional performance with perfect recall (1.0000) and strong F1 (0.9796); benefits significantly from feature scaling in the pipeline. |
| Random Forest (Ensemble) | Robust ensemble method with balanced precision and recall (0.9583); highest AUC (0.9937) shows excellent ranking ability. |
| Naive Bayes | Solid baseline with good performance (F1: 0.9517); Gaussian assumption works reasonably well for continuous features. |
| Decision Tree | Lowest performer but still respectable (0.9123 accuracy); single tree is less stable than ensemble methods. |
| **Overall Winner** | **Logistic Regression** (highest F1, accuracy, and MCC; most balanced and reliable classifier for breast cancer diagnosis). |

## Project Structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- dataset_used.csv
|-- dataset_info.json
|-- model_metrics.csv
|-- model/
|   |-- train_models.py
|   |-- ML_Assignment_2_Classification.ipynb
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest_ensemble.joblib
|-- artifacts/
|   |-- *_metrics.json
```

## How to run locally
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## Streamlit app requirements coverage
The app implements all required assignment features:
1. CSV upload option for test data.
2. Model selection dropdown.
3. Display of evaluation metrics.
4. Confusion matrix and classification report.

## Mandatory submission items to include in final PDF
1. GitHub repository link.
2. Live Streamlit app link: ADD_YOUR_STREAMLIT_APP_LINK_HERE
3. Screenshot of execution on BITS Virtual Lab.
4. This README content copied into the submission PDF.
