import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils import load_model, create_features, predict_risk

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Endangered Species AI", layout="wide")

st.title("🌍 Endangered Species Risk Intelligence System")

# ----------------------------
# LOAD DATA FROM GOOGLE DRIVE (SAFE)
# ----------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1Ob1WSj2jntkLOKoSMR1MuhQpL0gaItoK"
    return pd.read_csv(url)

df = load_data()

# ----------------------------
# CLEAN DATA
# ----------------------------
df.columns = df.columns.str.strip()

df["1970"] = pd.to_numeric(df["1970"], errors="coerce")
df["2020"] = pd.to_numeric(df["2020"], errors="coerce")
df = df.dropna(subset=["1970", "2020"])

df["Change"] = df["2020"] - df["1970"]
df["Growth_Ratio"] = df["2020"] / (df["1970"] + 1)
df["Log_Change"] = np.log1p(df["2020"]) - np.log1p(df["1970"])

# ----------------------------
# LOAD MODEL (FIXED PATH SAFE)
# ----------------------------
model, scaler, label_encoder = load_model()

# ----------------------------
# NAVIGATION
# ----------------------------
section = st.sidebar.radio(
    "Navigation",
    ["Overview", "Model Insights", "Country Analysis", "Predict Risk"]
)

# ----------------------------
# OVERVIEW
# ----------------------------
if section == "Overview":

    st.subheader("📊 Dataset Overview")

    st.write(df.head())

    st.write("Total Species:", df["Binomial"].nunique())

    st.line_chart(df.groupby("Binomial")["2020"].mean().head(50))

# ----------------------------
# MODEL INSIGHTS
# ----------------------------
elif section == "Model Insights":

    st.subheader("🧠 Feature Importance")

    features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

    fig, ax = plt.subplots()
    ax.bar(features, model.feature_importances_)
    ax.set_title("Random Forest Feature Importance")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")

    corr = df[features].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# ----------------------------
# COUNTRY ANALYSIS
# ----------------------------
elif section == "Country Analysis":

    st.subheader("🌍 Endangered Species by Country")

    df["Risk"] = "Stable"
    df.loc[df["Growth_Ratio"] < 0.5, "Risk"] = "Endangered"
    df.loc[(df["Growth_Ratio"] >= 0.5) & (df["Growth_Ratio"] < 0.8), "Risk"] = "Vulnerable"

    endangered = df[df["Risk"] == "Endangered"]

    country_counts = endangered["Country"].value_counts().head(10)

    fig = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        labels={"x": "Country", "y": "Count"},
        title="Top Countries with Endangered Species"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# PREDICTION
# ----------------------------
elif section == "Predict Risk":

    st.subheader("🔮 Predict Species Risk")

    col1, col2 = st.columns(2)

    pop_1970 = col1.number_input("Population 1970", value=100.0)
    pop_2020 = col2.number_input("Population 2020", value=80.0)

    input_data = create_features(pop_1970, pop_2020)

    if st.button("Predict Risk"):
        result = predict_risk(model, scaler, label_encoder, input_data)
        st.success(f"Predicted Risk: {result}")
