import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Customer Churn AI", page_icon="📊", layout="wide")

# Asset Loading
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(current_dir, 'churn_model.pkl')
scaler_file = os.path.join(current_dir, 'scaler.pkl')

@st.cache_resource
def load_assets():
    try:
        m = joblib.load(model_file)
        s = joblib.load(scaler_file)
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- CSS for Design ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetric"] { background-color: #1e2130; border: 1px solid #31333f; padding: 20px; border-radius: 12px; }
    [data-testid="stMetricLabel"] { color: #a3a8b4 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Telco Customer Churn Analysis")

if model is None:
    st.error("⚠️ Model files not found.")
    st.stop()

# 2. Sidebar - Basic Inputs
st.sidebar.header("📋 Customer Profile")
with st.sidebar:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)
    
    st.write("---")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.divider()
    st.caption("Advanced ML Predictor")

# 3. Main Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Tenure", f"{tenure} Mo")
col2.metric("Monthly", f"${monthly_charges}")
col3.metric("Total", f"${total_charges}")

if st.button("Start Risk Analysis"):
    try:
        # Step 1: Securely get column names
        expected_columns = []
        if hasattr(model, 'feature_names_in_'):
            expected_columns = list(model.feature_names_in_)
        elif hasattr(scaler, 'feature_names_in_'):
            expected_columns = list(scaler.feature_names_in_)
            
        if not expected_columns:
            st.error("Could not retrieve feature names from the model or scaler.")
            st.stop()

        # Step 2: Create a dictionary with user inputs
        user_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
            'InternetService_No': 1 if internet == "No" else 0
        }
        
        # Step 3: Build the DataFrame correctly
        # We create a dictionary where every expected column is 0
        full_data_dict = {col: 0 for col in expected_columns}
        
        # Update with our actual user values
        for col, value in user_data.items():
            if col in full_data_dict:
                full_data_dict[col] = value
        
        # Convert the dictionary to a DataFrame
        df_final = pd.DataFrame([full_data_dict])
        
        # Ensure column order matches the model exactly
        df_final = df_final[expected_columns]

        # Step 4: Scale and Predict
        scaled_data = scaler.transform(df_final)
        prob = model.predict_proba(scaled_data)[0][1]
        risk_score = int(prob * 100)

        # Result Display
        st.subheader("🎯 Analysis Results")
        if risk_score > 50:
            st.error(f"### High Risk: {risk_score}%")
            st.warning("Customer is likely to churn. Retention strategy recommended.")
        else:
            st.success(f"### Low Risk: {risk_score}%")
            st.info("Customer loyalty is stable.")
        
        st.progress(risk_score)

    except Exception as e:
        st.error(f"Prediction Error: {e}")