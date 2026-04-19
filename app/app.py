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

# ---------------- TITLE ----------------
st.title("🌍 Endangered Species Risk Intelligence System")

st.markdown("""
A machine learning system that predicts extinction risk using population trends,
growth ratios, and ecological change indicators.
""")

# ---------------- SAFE CHECK ----------------
if df is None or model is None:
    st.stop()

# ---------------- SIDEBAR INPUT ----------------
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
    label = label_encoder.inverse_transform(pred)[0]

    st.subheader("🧠 Prediction Result")
    st.success(f"Predicted Risk: **{label}**")

# ---------------- LAYOUT (SINGLE SCROLL PAGE) ----------------

st.divider()

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Model Feature Importance")

features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

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

if "Country" in df.columns:
    endangered_df = df.copy()

    if "Risk" in df.columns:
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

# ---------------- GLOBAL STORY ----------------
st.subheader("🧠 Insight Summary")

st.markdown("""
- Growth Ratio is the strongest predictor of extinction risk  
- Europe & North America dominate due to data availability bias  
- Decline rate matters more than raw population size  
- Model achieves ~95.8% accuracy using ensemble learning  
""")
