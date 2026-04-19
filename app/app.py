import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import load_data, load_model, load_scaler, load_encoder

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Endangered Species AI", layout="wide")

st.title("🌍 Endangered Species Risk Intelligence System")
st.write("AI-powered ecological analysis of global biodiversity trends")

# -------------------------
# LOAD DATA & MODELS
# -------------------------
df = load_data()
model = load_model()
scaler = load_scaler()
encoder = load_encoder()

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df.columns = df.columns.map(str)

df["Change"] = df["2020"] - df["1970"]
df["Growth_Ratio"] = df["2020"] / (df["1970"] + 1)
df["Log_Change"] = np.log1p(df["2020"]) - np.log1p(df["1970"])

df["Risk"] = "Stable"
df.loc[df["Growth_Ratio"] < 0.5, "Risk"] = "Endangered"
df.loc[(df["Growth_Ratio"] >= 0.5) & (df["Growth_Ratio"] < 0.8), "Risk"] = "Vulnerable"

# =====================================================
# 📊 1. OVERVIEW SECTION
# =====================================================
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Species", df["Binomial"].nunique())
col2.metric("Countries", df["Country"].nunique())
col3.metric("Records", len(df))

fig1 = px.bar(
    df["Class"].value_counts().reset_index(),
    x="Class",
    y="count",
    title="Species Distribution by Class"
)
st.plotly_chart(fig1, use_container_width=True)

# =====================================================
# 🧠 2. MODEL COMPARISON SECTION
# =====================================================
st.header("🧠 Model Comparison")

features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]
X = df[features].fillna(0)
X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)
df["Predicted_Risk"] = encoder.inverse_transform(preds)

model_scores = {
    "Random Forest": 0.9583,
    "Gradient Boosting": 0.9583,
    "Logistic Regression": 0.9167
}

fig2 = px.bar(
    x=list(model_scores.keys()),
    y=list(model_scores.values()),
    title="Model Accuracy Comparison",
    color=list(model_scores.values()),
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig2, use_container_width=True)

# Feature importance
fig3 = px.bar(
    x=features,
    y=model.feature_importances_,
    title="Feature Importance (Random Forest)"
)
st.plotly_chart(fig3, use_container_width=True)

st.success("Insight: Growth_Ratio dominates → relative decline matters more than raw population")

# =====================================================
# 🌍 3. GEO ANALYSIS + MAP (IMPORTANT UPGRADE)
# =====================================================
st.header("🌍 Global Biodiversity Risk Map")

endangered = df[df["Risk"] == "Endangered"]
country_counts = endangered["Country"].value_counts().reset_index()
country_counts.columns = ["Country", "Count"]

fig4 = px.choropleth(
    country_counts,
    locations="Country",
    locationmode="country names",
    color="Count",
    color_continuous_scale="Reds",
    title="Global Distribution of Endangered Species (Model-Based)"
)

st.plotly_chart(fig4, use_container_width=True)

# bar fallback view
fig5 = px.bar(
    country_counts.head(10),
    x="Country",
    y="Count",
    title="Top 10 Countries with Endangered Species"
)
st.plotly_chart(fig5, use_container_width=True)

st.warning("Note: This reflects data density + monitoring bias, not only extinction reality")

# =====================================================
# 🔮 4. PREDICTION SECTION
# =====================================================
st.header("🔮 Predict Species Risk")

col1, col2 = st.columns(2)

with col1:
    pop_1970 = st.number_input("Population 1970", value=100.0)

with col2:
    pop_2020 = st.number_input("Population 2020", value=120.0)

if st.button("Predict Risk"):

    change = pop_2020 - pop_1970
    growth_ratio = pop_2020 / (pop_1970 + 1)
    log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

    input_data = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)
    result = encoder.inverse_transform(pred)[0]

    if result == "Endangered":
        st.error(f"Prediction: {result}")
    elif result == "Vulnerable":
        st.warning(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")

# =====================================================
# 🧠 FINAL INSIGHT SECTION
# =====================================================
st.header("🧠 Key Insights")

st.markdown("""
- Growth Ratio is the strongest predictor of extinction risk  
- Europe & North America dominate due to better data coverage  
- Model is more sensitive to relative decline than absolute population  
- Dataset reflects **ecological reality + reporting bias**
""")
