import streamlit as st
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

if df is None or model is None:
    st.error("Model or data failed to load.")
    st.stop()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🔍 Predict Species Risk")

pop_1970 = st.sidebar.number_input("Population in 1970", min_value=1.0, value=100.0)
pop_2020 = st.sidebar.number_input("Population in 2020", min_value=1.0, value=80.0)

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
    st.success(f"Predicted Risk: {label}")

st.divider()

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

fig, ax = plt.subplots()
ax.barh(features, model.feature_importances_)
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

st.divider()

# ---------------- MODEL COMPARISON ----------------
st.subheader("⚖️ Model Comparison")

models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
scores = [0.9583, 0.9167, 0.9583]

fig2 = px.bar(
    x=models,
    y=scores,
    text=scores,
    title="Model Accuracy Comparison"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- COUNTRY ANALYSIS ----------------
st.subheader("🌍 Country-Level Endangered Species")

if "Country" in df.columns:

    endangered_df = df.copy()

    if "Risk" in df.columns:
        endangered_df = df[df["Risk"] == "Endangered"]

    country_counts = endangered_df["Country"].value_counts().head(10)

    fig3 = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        labels={"x": "Country", "y": "Count"},
        title="Top Countries with Endangered Species"
    )

    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.choropleth(
        locations=country_counts.index,
        locationmode="country names",
        color=country_counts.values,
        title="Global Endangered Species Distribution"
    )

    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------------- INSIGHTS ----------------
st.subheader("🧠 Key Insights")

st.markdown("""
- Growth ratio is strongest predictor  
- Ensemble models outperform linear models  
- Europe & US dominate due to dataset bias  
- Relative change > absolute population  
""")
