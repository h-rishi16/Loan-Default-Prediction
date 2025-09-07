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

## Dataset
- Source: Kaggle – LendingClub Loan Data
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
- Logistic Regression (baseline performance, ROC-AUC ~0.70)
- XGBoost with imbalance handling (ROC-AUC ~0.71)

**4. Model Explainability**
- SHAP summary plots highlight key risk drivers: interest rate, loan grade, term, debt-to-income ratio, and annual income

## Results
**Logistic Regression**
- ROC-AUC: ~0.714
- Good interpretability

**XGBoost**
- ROC-AUC: ~0.761
- Better recall for defaults after handling class imbalance

**SHAP Insights**
- Top predictors: interest rate, loan grade, loan term, debt-to-income ratio, annual income

## Repository Structure
```
loan-default-prediction/
│── Loan_Default_Prediction.ipynb   # Main notebook
│── README.md                       # Project documentation
│── requirements.txt                # Dependencies
```

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Future Work
- Hyperparameter tuning with GridSearchCV or Optuna
- Improved handling of extreme outliers in income and DTI
- Deployment as a Streamlit web app for interactive predictions

## Author
Hrishikesh Joshi
