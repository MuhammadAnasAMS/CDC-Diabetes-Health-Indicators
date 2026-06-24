# CDC Diabetes Health Indicators - Machine Learning Predictive Pipeline

This repository contains an end-to-end Machine Learning project that leverages the **CDC Diabetes Health Indicators Dataset** to predict whether an individual has pre-diabetes/diabetes based on healthcare statistics and lifestyle survey information. 

The goal of this project is to benchmark various classification algorithms to find the most accurate and reliable predictive model for preventative healthcare analytics.

## 📊 Dataset Overview
The dataset is fetched programmatically from the **UCI Machine Learning Repository** (UCI ID: 891). It represents survey responses collected by the CDC's Behavioral Risk Factor Surveillance System (BRFSS).

- **Total Instances:** 253,680 records
- **Features (21 columns):** Demographic factors (Age, Sex, Income, Education), medical history (High Blood Pressure, High Cholesterol, Stroke, Heart Disease), and lifestyle choices (BMI, Smoking, Physical Activity, Diet/Alcohol habits).
- **Target Variable (`Diabetes_binary`):** - `0` = No Diabetes / Healthy
  - `1` = Prediabetes or Diabetes

## 🛠️ Project Architecture & Workflow
The workflow is completely structured inside a Jupyter Notebook pipeline:

1. **Environment Setup & Data Retrieval:** Dynamic fetching of the live tabular dataset using the `ucimlrepo` framework.
2. **Data Exploration & Preprocessing:** Data integrity validation (checking data shapes, identifying column types, verifying null values).
3. **Feature-Target Mapping:** Separating patient attributes from the diagnosis indicator and generating structured pandas DataFrames.
4. **Model Benchmarking:** Implement, train, and test multiple foundational and ensemble classification algorithms:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
   - K-Nearest Neighbors (KNN)
   - Support Vector Classifier (SVC / LinearSVC)
   - Multi-layer Perceptron (MLP Neural Networks)
5. **Optimization & Performance Evaluation:** - **Cross-Validation:** Multi-fold cross-validation to guarantee generalization.
   - **Hyperparameter Tuning:** Systematic search optimization using `GridSearchCV`.
   - **Metrics Suite:** Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curve rendering.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries. You can set them up using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost ucimlrepo
