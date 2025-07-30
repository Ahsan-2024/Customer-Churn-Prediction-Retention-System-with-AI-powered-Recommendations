import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- Load Trained Model, Scaler, KMeans ----
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")

# ---- Recommendation Function ----
def get_recommendation(prob, cluster, tenure, charges):
    if prob > 0.7:
        return "High churn risk - Offer loyalty discounts"
    elif cluster == 1 and charges > 70:
        return "Suggest bundle services to reduce cost"
    elif tenure < 12:
        return "Provide onboarding assistance"
    else:
        return "Customer appears stable"

# ---- Page Configuration ----
st.set_page_config(page_title="Smart Churn Predictor", page_icon="🔍", layout="centered")

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            text-align: center;
        }
        .css-1cpxqw2 edgvbvh3 {
            background-color: #ffffff10 !important;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }
        .stTextInput>div>div>input, .stFileUploader>div>div>input {
            background-color: #1e1e1e;
            color: white;
        }
        .stDataFrame {
            background-color: #1e1e1e;
            color: white;
        }
        .block-container {
            padding: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
    <h1 style='font-size: 3rem;'>🔍 Smart Churn Prediction & Retention System</h1>
    <p style='font-size: 1.2rem;'>Upload customer data, predict churn, and get AI-powered retention strategies</p>
""", unsafe_allow_html=True)

# ---- File Upload ----
st.markdown("### ⬆️ Upload Customer Data (CSV)")
uploaded_file = st.file_uploader("Upload your CSV file (no 'Churn' column)", type="csv")

# ---- Prediction Logic ----
if uploaded_file is not None:
    try:
        st.success("✅ File uploaded successfully!")

        input_data = pd.read_csv(uploaded_file)

        # Encode categorical columns
        cat_cols = input_data.select_dtypes(include='object').columns.tolist()
        input_data_encoded = pd.get_dummies(input_data, columns=cat_cols)

        # Match training feature columns
        model_input_columns = model.get_booster().feature_names
        for col in model_input_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        input_data_encoded = input_data_encoded[model_input_columns]

        # Predict
        churn_probs = model.predict_proba(input_data_encoded)[:, 1]
        churn_preds = model.predict(input_data_encoded)

        cluster_features = scaler.transform(input_data[['tenure', 'MonthlyCharges', 'TotalCharges']])
        clusters = kmeans.predict(cluster_features)

        input_data['Churn_Probability'] = churn_probs
        input_data['Churn_Prediction'] = churn_preds
        input_data['Cluster'] = clusters
        input_data['Recommendation'] = input_data.apply(lambda row: get_recommendation(
            row['Churn_Probability'], row['Cluster'], row['tenure'], row['MonthlyCharges']), axis=1)

        # Display results
        st.markdown("### 📊 Prediction Results")
        st.dataframe(input_data[['tenure', 'MonthlyCharges', 'Churn_Probability', 'Churn_Prediction', 'Recommendation']])

        # Download
        st.download_button(
            label="📥 Download Results as CSV",
            data=input_data.to_csv(index=False),
            file_name="churn_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.warning("👆 Please upload a customer CSV file first.")
