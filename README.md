# Machine Learning Assignment 2 - Classification Models

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
- **Repository:** https://github.com/kaushal-pant/MLProject

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
| Logistic Regression | **Top performer across all metrics.** Achieves 98.25% accuracy with perfect metric alignment (precision: 98.61%, recall: 98.61%, F1: 98.61%, AUC: 99.54%). The symmetric precision-recall indicates no systematic bias toward false positives or negatives—both classes are equally well-identified. Highest MCC (0.9623) signals excellent overall correlation between predictions and reality. Simplicity and interpretability make this the primary candidate for clinical deployment where model reasoning must be auditable. |
| Decision Tree | **Poorest model performance.** Accuracy of 91.23% drops notably compared to competitors; AUC of 91.47% suggests weaker ranking ability for borderline cases. Notably, precision (95.59%) substantially exceeds recall (90.28%), implying the model over-conservatively labels cases, missing some actual malignant instances—a risky behavior in medical diagnosis. Lowest MCC (0.8174) reflects poor overall predictive correlation. Despite max_depth=10, evidence of training-specific overfitting remains; ensemble approaches (e.g., Random Forest) would better generalize this tree-based logic. |
| K-Nearest Neighbors | **Competitive alternative with distinctive strengths.** Accuracy (97.37%) marginally trails Logistic Regression but F1 (97.96%) is second-best. Perfect recall (100%) is exceptional—all truly malignant cases are caught—though precision (96.00%) is slightly lower, meaning ~4% false alarms. Feature scaling via pipeline is critical for this distance-based method; without it, large-magnitude features would dominate. Prediction latency grows with dataset size; cross-validation across k values could further optimize the k=5 choice. |
| Naive Bayes | **Reliable baseline despite simplifying assumptions.** At 93.86% accuracy and 95.17% F1, it underperforms the top models but outpaces Decision Tree. Strong AUC (98.78%)—second only to Logistic Regression—indicates excellent probabilistic ranking even with the naive feature-independence assumption. Symmetric precision/recall (94.52%/95.83%) shows balanced error distribution. Computationally fastest model; useful as a quick diagnostic or when interpretability of probabilistic outputs is valued over raw accuracy. |
| Random Forest (Ensemble) | **Second-best overall with ensemble robustness.** Matches KNN on accuracy and F1 (94.74%, 95.83%) but leads with highest AUC (99.37%), showing superior ability to rank confidence on boundary cases. The ensemble of 100 trees averages out individual tree noise, dramatically reducing overfitting compared to single Decision Tree. Feature importance rankings provide explainability; variable contributions help clinicians understand which tissue properties drive predictions. Production-viable alternative to Logistic Regression with slightly lower accuracy but greater flexibility for non-linear patterns. |
| **Overall Winner** | **Logistic Regression** — Highest performance on 5 of 6 metrics (accuracy, AUC, precision, recall, F1, MCC). Combines superior metrics with maximum interpretability and minimal computational overhead, making it optimal for clinical decision-support systems. Recommended for production deployment on breast cancer diagnosis where both accuracy and explainability are non-negotiable. |

## Project Structure
```text
MLProject/
|-- app.py                                    -(Streamlit app)
|-- requirements.txt                          -(Dependencies)
|-- README.md                                 -(Documentation)
|-- test_data.csv                             -(Test dataset)
|-- dataset_info.json                         -(Dataset metadata)
|-- model_metrics.csv                         -(Model metrics)
|-- model/
|   |-- 2025AC05981_ML_Assignment_2.ipynb     -(Notebook for artifacts)
```

## Streamlit app functionality
The app implements all required assignment features:
1. CSV upload option for test data.
2. Model selection dropdown.
3. Display of evaluation metrics.
4. Confusion matrix and classification report.

## Streamlit app link
https://mlproject-2025ac05981.streamlit.app/