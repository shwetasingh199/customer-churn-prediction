import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# =========================
# TITLE
# =========================
st.title("📊 Customer Churn Prediction Dashboard")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📥 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0, 200, 50)
total = st.sidebar.number_input("Total Charges", 0, 10000, 500)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# =========================
# ENCODING
# =========================
def encode_input(gender, contract):
    gender = 1 if gender == "Male" else 0

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    contract = contract_map[contract]

    return gender, contract

gender, contract = encode_input(gender, contract)

# =========================
# PREPARE INPUT DATA
# =========================
columns = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

input_data = np.array([[gender, senior, tenure, monthly, total, contract]])
input_df = pd.DataFrame(input_data, columns=columns)

# Scale input
input_scaled = scaler.transform(input_df)
predict_button = st.button("🔍 Predict Churn")
# =========================
# PREDICTION
# =========================
if predict_button:

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ Customer likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer likely to STAY (Probability: {1-prob:.2f})")

    # Risk level
    if prob > 0.7:
        st.error("🔥 High Risk Customer")
    elif prob > 0.4:
        st.warning("⚠️ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")

    # =========================
    # SHAP (INSIDE BUTTON)
    # =========================
    st.subheader("🧠 Model Explanation (SHAP)")

    background = pd.DataFrame(
        np.zeros((10, input_df.shape[1])),
        columns=input_df.columns
    )

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    impact_values = np.array(shap_values_class1).flatten()
    impact_values = impact_values[:len(input_df.columns)]

    # Table
    st.write("Top factors influencing prediction:")
    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": impact_values
    }).sort_values(by="Impact", ascending=False)

    st.dataframe(importance_df)

    # Chart
    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Impact"])
    st.pyplot(fig)

# =========================
# SHAP EXPLAINABILITY (FINAL STABLE VERSION)
# =========================
st.subheader("🧠 Model Explanation (SHAP)")

background = pd.DataFrame(
    np.zeros((10, input_df.shape[1])),
    columns=input_df.columns
)

explainer = shap.KernelExplainer(model.predict_proba, background)

shap_values = explainer.shap_values(input_df)

# Handle output format
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

# Convert to proper 1D
impact_values = np.array(shap_values_class1).flatten()
impact_values = impact_values[:len(input_df.columns)]

# =========================
# TABLE
# =========================
st.write("Top factors influencing prediction:")

importance_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Impact": impact_values
}).sort_values(by="Impact", ascending=False)

st.dataframe(importance_df)

# =========================
# BAR CHART (REPLACES WATERFALL)
# =========================
st.subheader("📊 Feature Impact Visualization")

fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Impact"])
ax.set_xlabel("Impact on Prediction")
ax.set_title("Feature Contribution")
st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE IMAGE
# =========================
st.subheader("🔍 Feature Importance")
st.image("outputs/feature_importance.png")

# =========================
# BUSINESS INSIGHTS
# =========================
st.subheader("💼 Business Insights")

st.write("""
- Customers with low tenure are more likely to churn  
- High monthly charges increase churn probability  
- Long-term customers are more stable  
- Contract type strongly impacts retention  
""")

# =========================
# DATA VISUALIZATION
# =========================
st.subheader("📈 Dataset Insights")

df = pd.read_csv("data/raw/churn.csv")

fig2, ax2 = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax2)
st.pyplot(fig2)