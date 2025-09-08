# Loan Default Risk Prediction

## Overview
This project predicts loan default risk using historical LendingClub loan data. The goal is to help banks and lending institutions make better credit risk decisions, reduce exposure to bad loans, and improve profitability.

We developed two models:
- Logistic Regression – interpretable baseline model
- XGBoost – advanced model with stronger predictive performance

To enhance trust and explainability, SHAP was used to identify the most important risk drivers.

## Business Problem
Loan defaults represent a major risk for banks and lenders. A machine learning model can:
- Flag high-risk borrowers before loan approval
- Support compliance with regulatory frameworks such as Basel III
- Improve transparency by explaining why a prediction was made

## Tech Stack
- Python (Pandas, Numpy, Matplotlib, Seaborn)
- Scikit-learn (Logistic Regression, pipelines)
- XGBoost (gradient boosted trees for tabular data)
- SHAP (explainable AI)
- Jupyter Notebook
- Streamlit (for interactive web app)

## Dataset
- Source: Kaggle – [LendingClub Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
- Features: loan amount, interest rate, loan term, annual income, debt-to-income ratio, credit history, loan grade, and others
- Target:
  - 1 → Loan Default (Charged Off)
  - 0 → Fully Paid

## Project Pipeline
**1. Data Preparation**
- Removed irrelevant identifiers (IDs, URLs, text descriptions)
- Handled missing values and outliers
- Encoded categorical variables
- Split into training and testing sets

**2. Exploratory Data Analysis (EDA)**
- Distribution of loan status (imbalanced dataset)
- Trends in annual income, interest rates, and debt-to-income ratios
- Correlation analysis

**3. Modeling**
- Logistic Regression (baseline performance, ROC-AUC ~0.71)
- XGBoost with imbalance handling (ROC-AUC ~0.76)

**4. Model Explainability**
- SHAP summary plots highlight key risk drivers: interest rate, loan grade, term, debt-to-income ratio, and annual income

## Results
**Logistic Regression**
- ROC-AUC: ~0.71
- Good interpretability

**XGBoost**
- ROC-AUC: ~0.76
- Better recall for defaults after handling class imbalance

**SHAP Insights**
- Top predictors: interest rate, loan grade, loan term, debt-to-income ratio, annual income

## Repository Structure
```
loan-default-prediction/
│── Loan-Default.ipynb              # Main notebook with EDA, preprocessing, and model training
│── app.py                          # Streamlit web app for interactive predictions
│── xgb_model.pkl                   # Trained XGBoost model saved with joblib
│── xgb_features.pkl                # Feature names used in training
│── README.md                       # Project documentation
│── requirements.txt                # Dependencies
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/h-rishi16/loan-default-prediction.git
cd loan-default-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Web App Usage
- Visit [Loan Default Predictor](https://h-rishi16-loan-default-prediction-app-uecn3j.streamlit.app)
- Enter borrower details (loan amount, interest rate, grade, employment length, etc.)
- The app predicts whether the borrower is **Likely to Default** or **Likely to Pay**, along with a probability score.

## Skills Demonstrated
- Data preprocessing and feature engineering
- Handling imbalanced datasets
- Model training and evaluation (ROC-AUC, classification report)
- Model explainability (SHAP values)
- Saving/loading ML models with joblib
- Deploying ML apps using Streamlit Cloud

## Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
jupyter
ipywidgets
streamlit
joblib
```

## Future Work
- Hyperparameter tuning with GridSearchCV or Optuna
- Improved handling of extreme outliers in income and DTI
- Additional model comparison (Random Forest, Neural Networks)
- Feature engineering based on SHAP insights

## Author
Hrishikesh Joshi
