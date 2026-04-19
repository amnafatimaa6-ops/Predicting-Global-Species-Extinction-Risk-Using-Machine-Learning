import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, load_model, load_scaler, load_encoder

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Endangered Species AI", layout="wide")

st.title("🌍 Endangered Species Risk Prediction System")
st.write("Machine Learning-powered ecological risk analysis dashboard")

# -------------------------
# Load everything
# -------------------------
df = load_data()
model = load_model()
scaler = load_scaler()
encoder = load_encoder()

# -------------------------
# Feature engineering (same as notebook)
# -------------------------
df.columns = df.columns.map(str)

df["Change"] = df["2020"] - df["1970"]
df["Growth_Ratio"] = df["2020"] / (df["1970"] + 1)
df["Log_Change"] = np.log1p(df["2020"]) - np.log1p(df["1970"])

# Risk (if needed for exploration)
df["Risk"] = "Stable"
df.loc[df["Growth_Ratio"] < 0.5, "Risk"] = "Endangered"
df.loc[(df["Growth_Ratio"] >= 0.5) & (df["Growth_Ratio"] < 0.8), "Risk"] = "Vulnerable"

# -------------------------
# Sidebar navigation
# -------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Insights", "Country Analysis", "Predict Risk"]
)

# -------------------------
# 1. OVERVIEW
# -------------------------
if menu == "Overview":
    st.subheader("📊 Dataset Overview")
    
    st.write("Shape:", df.shape)
    st.write("Species count:", df["Binomial"].nunique())
    
    st.subheader("Class distribution")
    st.bar_chart(df["Class"].value_counts().head(10))

# -------------------------
# 2. MODEL INSIGHTS
# -------------------------
elif menu == "Model Insights":
    st.subheader("🧠 Model Insights")

    features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    df["Predicted_Risk"] = encoder.inverse_transform(preds)

    st.write("Prediction distribution:")
    st.bar_chart(df["Predicted_Risk"].value_counts())

    st.subheader("Feature importance (from trained model)")
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(features, importances)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------
# 3. COUNTRY ANALYSIS
# -------------------------
elif menu == "Country Analysis":
    st.subheader("🌍 Geographic Risk Distribution")

    endangered = df[df["Risk"] == "Endangered"]
    country_counts = endangered["Country"].value_counts().head(10)

    fig, ax = plt.subplots()
    country_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.write("Top countries showing high model-based risk concentration.")

# -------------------------
# 4. PREDICTION SYSTEM
# -------------------------
elif menu == "Predict Risk":
    st.subheader("🔮 Predict Species Risk")

    col1, col2 = st.columns(2)

    with col1:
        pop_1970 = st.number_input("Population 1970", min_value=0.0, value=100.0)

    with col2:
        pop_2020 = st.number_input("Population 2020", min_value=0.0, value=120.0)

    if st.button("Predict Risk"):

        change = pop_2020 - pop_1970
        growth_ratio = pop_2020 / (pop_1970 + 1)
        log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

        input_data = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)
        result = encoder.inverse_transform(pred)[0]

        st.success(f"Predicted Risk Level: {result}")
