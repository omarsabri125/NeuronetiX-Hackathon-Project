
# Customer Churn Prediction - Telecom Dataset

## 📄 Project Overview

This project focuses on predicting customer churn using machine learning techniques on a telecom dataset. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and hyperparameter tuning to identify customers likely to discontinue the service.

## 📂 Dataset

- **Source**: `Telecom Customers Churn.csv`
- **Target variable**: `Churn` (whether the customer churned or not)
- **Features**: Includes customer demographics, service usage patterns, account information, and charges.

## 🚀 Key Steps

1. **Import Libraries**  
   Python libraries for data handling (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`, `plotly`), preprocessing (`scikit-learn`), and modeling (classification algorithms like `RandomForestClassifier`, `XGBoost`, `SVC`, etc.).

2. **Data Loading & Cleaning**  
   - Loaded dataset from CSV file.
   - Dropped irrelevant columns (`customerID`).
   - Identified and handled missing values.

3. **Exploratory Data Analysis (EDA)**  
   - Visualized feature distributions.
   - Checked class balance for the target variable.
   - Analyzed relationships between features.

4. **Data Preprocessing**  
   - Encoding categorical variables using `LabelEncoder` and `OneHotEncoder`.
   - Feature scaling using `StandardScaler` and `MinMaxScaler`.
   - Applied various imputation strategies (`SimpleImputer`, `KNNImputer`, `IterativeImputer`).

5. **Model Building**  
   Implemented and compared multiple classification algorithms:
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine
   - Decision Tree
   - Random Forest
   - XGBoost
   - Naive Bayes
   - Extra Trees Classifier
   - Stochastic Gradient Descent (SGD) Classifier
   - Ensemble models (AdaBoost, Gradient Boosting)

6. **Model Evaluation**  
   - Evaluated models using metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - Visualized ROC and Precision-Recall curves.
   - Performed cross-validation and hyperparameter tuning (`GridSearchCV`).

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
```

Install all dependencies using:

```
pip install -r requirements.txt
```

## 💡 Results

- Compared multiple models to find the best-performing algorithm for churn prediction.
- Provided evaluation metrics and confusion matrices.
- Visualized the performance of models through curves and plots.

## 🤝 Contribution

Feel free to fork this repo, raise issues, or submit pull requests.

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
