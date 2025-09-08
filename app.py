import streamlit as st
import joblib
import pandas as pd

# Load model and features
xgb = joblib.load("xgb_model.pkl")
features = joblib.load("xgb_features.pkl")

st.title("Loan Default Risk Predictor")

st.write("Enter borrower details to predict loan default risk:")

# Collect inputs
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=10000)
term = st.selectbox("Term", [" 36 months", " 60 months"])
int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0)
grade = st.selectbox("Grade", ["A","B","C","D","E","F","G"])
sub_grade = st.selectbox("Sub Grade", ["A1","A2","B1","C1","D1","E1"])
emp_length = st.selectbox("Employment Length", ["< 1 year","1 year","2 years","3 years","4 years","5 years",
                                                 "6 years","7 years","8 years","9 years","10+ years"])
home_ownership = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"])
annual_inc = st.number_input("Annual Income", min_value=1000, max_value=500000, value=60000)
purpose = st.selectbox("Purpose", ["credit_card","debt_consolidation","home_improvement","major_purchase","small_business"])
dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=40.0, value=10.0)
revol_util = st.number_input("Revolving Credit Utilization (%)", min_value=0.0, max_value=150.0, value=30.0)
total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100, value=20)

# Create dataframe for prediction
input_data = pd.DataFrame([{
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "grade": grade,
    "sub_grade": sub_grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "purpose": purpose,
    "dti": dti,
    "revol_util": revol_util,
    "total_acc": total_acc
}])

# One-hot encode like training
input_data = pd.get_dummies(input_data)
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[features]

# Predict
if st.button("Predict"):
    prob = xgb.predict_proba(input_data)[:, 1][0]
    prediction = "Likely to Default" if prob >= 0.5 else "Likely to Pay"
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Default Probability:** {prob:.2f}")