import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from utils import load_model, load_data

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Endangered Species AI", layout="wide")

# ---------------- LOAD ----------------
model, scaler, label_encoder = load_model()
df = load_data()

st.title("🌍 Endangered Species Risk Intelligence System")

st.markdown("""
A machine learning system that predicts extinction risk using population trends,
growth ratios, and ecological change indicators.
""")

# ---------------- HARD STOP IF MODEL FAILS ----------------
if model is None or scaler is None or label_encoder is None:
    st.error("Model failed to load. Check pickle files in /model folder.")
    st.stop()

if df is None:
    st.error("Dataset failed to load.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Predict Species Risk")

pop_1970 = st.sidebar.number_input("Population in 1970", min_value=1.0, value=100.0)
pop_2020 = st.sidebar.number_input("Population in 2020", min_value=1.0, value=80.0)

# ---------------- FEATURE ENGINEERING ----------------
change = pop_2020 - pop_1970
growth_ratio = pop_2020 / (pop_1970 + 1)
log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

input_data = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
scaled = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Risk"):
    pred = model.predict(scaled)

    # SAFE handling (important fix)
    if hasattr(label_encoder, "inverse_transform"):
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = str(pred[0])

    st.subheader("🧠 Prediction Result")
    st.success(f"Predicted Risk: **{label}**")

st.divider()

# ---------------- FEATURE IMPORTANCE (FIXED SAFE VERSION) ----------------
st.subheader("📊 Model Feature Importance")

features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

    # SAFETY CHECK
    if len(importance) == len(features):
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)
    else:
        st.warning("Feature importance mismatch detected.")
else:
    st.warning("This model does not support feature importance.")

st.divider()

# ---------------- MODEL COMPARISON ----------------
st.subheader("⚖️ Model Comparison")

models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
scores = [0.9583, 0.9167, 0.9583]

fig2 = px.bar(
    x=models,
    y=scores,
    labels={"x": "Model", "y": "Accuracy"},
    text=scores,
    title="Model Performance Comparison"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- COUNTRY ANALYSIS ----------------
st.subheader("🌍 Country-Level Endangered Distribution")

if "Country" in df.columns and "Risk" in df.columns:

    endangered_df = df[df["Risk"] == "Endangered"]

    country_counts = endangered_df["Country"].value_counts().head(10)

    fig3 = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        labels={"x": "Country", "y": "Endangered Species Count"},
        title="Top Countries with Endangered Species"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------- INSIGHT ----------------
st.subheader("🧠 Insight Summary")

st.markdown("""
- Growth Ratio is the strongest predictor of extinction risk  
- Europe & North America dominate due to data availability bias  
- Decline rate matters more than raw population size  
- Model achieves ~95.8% accuracy using ensemble learning  
""")
