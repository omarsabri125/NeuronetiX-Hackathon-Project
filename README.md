# Customer Churn Prediction - Telecom Dataset

## 📄 Project Overview

This project focuses on predicting customer churn using machine learning techniques on a telecom dataset. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and hyperparameter tuning.
In addition, **MLflow** was integrated for **experiment tracking**, **model registry**, and **deployment preparation**, enabling better reproducibility and collaboration.

📌 This project was developed as part of a **Hackathon**, where our team proudly achieved **3rd place** 🏆.

## 📂 Dataset

* **Source**: `Telecom Customers Churn.csv`
* **Target variable**: `Churn` (whether the customer churned or not)
* **Features**: Includes customer demographics, service usage patterns, account information, and charges.

## 🚀 Key Steps

1. **Import Libraries**

   * Data handling: `pandas`, `numpy`
   * Visualization: `matplotlib`, `seaborn`, `plotly`
   * Preprocessing & modeling: `scikit-learn`, `xgboost`
   * Experiment tracking: `mlflow`

2. **Data Loading & Cleaning**

   * Loaded dataset from CSV file.
   * Dropped irrelevant columns (`customerID`).
   * Handled missing values.

3. **Exploratory Data Analysis (EDA)**

   * Visualized feature distributions.
   * Checked class balance for the target variable.
   * Explored relationships between features.

4. **Data Preprocessing**

   * Encoding categorical variables (`LabelEncoder`, `OneHotEncoder`).
   * Feature scaling (`StandardScaler`, `MinMaxScaler`).
   * Applied imputation strategies (`SimpleImputer`, `KNNImputer`, `IterativeImputer`).

5. **Model Building**
   Implemented and compared multiple classification algorithms:

   * Logistic Regression
   * K-Nearest Neighbors
   * Support Vector Machine
   * Decision Tree
   * Random Forest
   * XGBoost
   * Naive Bayes
   * Extra Trees Classifier
   * SGD Classifier
   * Ensemble models (AdaBoost, Gradient Boosting)

6. **Experiment Tracking with MLflow**

   * Logged parameters, metrics, confusion matrices, and ROC curves using MLflow.
   * Tracked multiple models and compared their performance.
   * Registered the best-performing model into the **MLflow Model Registry**.
   * Added versioning, tags, and aliases for easier model management.

7. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   * Plots: ROC curves, Precision-Recall curves.
   * Cross-validation and hyperparameter tuning with `GridSearchCV`.
   * Visualized and logged results via MLflow.

## 🔧 Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
plotly
scikit-learn
xgboost
mlflow
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 💡 Results

* Compared multiple models and tracked their performance with MLflow.
* Best-performing model was logged and versioned in the MLflow Model Registry.
* Confusion matrices, ROC curves, and evaluation metrics were visualized and stored as MLflow artifacts.
* 🏆 The project secured **3rd place in the Hackathon competition**, highlighting its practical impact and robustness.

## 📊 MLflow Usage

* **Tracking Server**: Run `mlflow ui` and visit [http://localhost:5000](http://localhost:5000) to view experiments.
* **Model Registry**: Promoted the best model and managed versions/aliases.
* **Model Loading**:

```python
import mlflow.sklearn

model_name = "forest_best"
model_version = "1"  # or "latest" / alias
model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.sklearn.load_model(model_uri)
print("✅ Model loaded successfully")
```

## 🧩 MLflow Project

You can run this project as an **MLflow Project**, making experiments fully reproducible and parameterized.

### 🔹 Run with default entry point

```bash
mlflow run . --experiment-name churn-detection
```

### 🔹 Run with multiple entry points

```bash
mlflow run -e forest . --experiment-name churn-detection
mlflow run -e logistic . --experiment-name churn-detection
mlflow run -e xgboost . --experiment-name churn-detection
```

### 🔹 Run with custom parameters

```bash
# Logistic Regression
mlflow run -e logistic . --experiment-name churn-detection -P c=3.5 -P p="l2"

# XGBoost
mlflow run -e xgboost . --experiment-name churn-detection -P n=250 -P lr=0.15 -P d=22
```

This allows you to easily reproduce experiments, tune hyperparameters, and track results in MLflow.

## 🚀 Deployment with MLflow

You can serve the registered model as a REST API using MLflow’s built-in serving functionality:

1. **Serve the model locally**:

```bash
mlflow models serve -m "models:/forest_best/1" -p 1234 --no-conda
```

This will start a REST API server on `http://127.0.0.1:1234/predict`.

2. **Send a request to the API**:

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"instances": [[65, 1, 29.85, 0, 29.85, "Female", "Yes", "DSL"]]}' \
http://127.0.0.1:1234/invocations
```

3. **Production Deployment**:

   * Promote the model to the `Production` stage in MLflow UI.
   * Use aliases like `models:/forest_best@Production` for stable loading.
   * Integrate with tools like Docker, Kubernetes, or cloud platforms (AWS Sagemaker, Azure ML, GCP Vertex AI).

