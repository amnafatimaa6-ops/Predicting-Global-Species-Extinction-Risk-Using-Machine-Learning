import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Endangered Species Risk Intelligence System",
    layout="wide"
)

st.title("🌍 Endangered Species Risk Intelligence System")

# ----------------------------
# LOAD DATA FROM GOOGLE DRIVE
# ----------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1Ob1WSj2jntkLOKoSMR1MuhQpL0gaItoK"
    df = pd.read_csv(url)
    return df

df = load_data()

# ----------------------------
# BASIC CLEANING (SAFE GUARD)
# ----------------------------
df = df.drop(columns=["Unnamed: 102"], errors="ignore")
df.columns = df.columns.str.strip()

# Convert year columns safely if needed
year_cols = [col for col in df.columns if str(col).isdigit()]

# ----------------------------
# LOAD MODEL FILES
# ----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# ----------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# ----------------------------
df["1970"] = pd.to_numeric(df["1970"], errors="coerce")
df["2020"] = pd.to_numeric(df["2020"], errors="coerce")

df = df.dropna(subset=["1970", "2020"])

df["Change"] = df["2020"] - df["1970"]
df["Growth_Ratio"] = df["2020"] / (df["1970"] + 1)
df["Log_Change"] = np.log1p(df["2020"]) - np.log1p(df["1970"])

# ----------------------------
# SIDEBAR (SIMPLE - NO TABS)
# ----------------------------
st.sidebar.header("⚙️ Controls")

section = st.sidebar.selectbox(
    "Choose Section",
    ["Overview", "Model Insights", "Country Analysis", "Predict Risk"]
)

# ----------------------------
# OVERVIEW
# ----------------------------
if section == "Overview":
    st.subheader("📊 Dataset Overview")

    st.write(df.head())

    st.write("### Species Count")
    st.write(df["Binomial"].nunique())

    st.write("### Quick Trend (1970 → 2020)")
    trend = df.groupby("Binomial")["2020"].mean()
    st.line_chart(trend.head(50))

# ----------------------------
# MODEL INSIGHTS
# ----------------------------
elif section == "Model Insights":
    st.subheader("🧠 Feature Importance")

    features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(features, importances)
    ax.set_title("Feature Importance (Random Forest)")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")

    corr = df[["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

# ----------------------------
# COUNTRY ANALYSIS
# ----------------------------
elif section == "Country Analysis":
    st.subheader("🌍 Country-Level Endangered Species")

    # simple risk rule (same logic idea you used)
    df["Risk"] = "Stable"
    df.loc[df["Growth_Ratio"] < 0.5, "Risk"] = "Endangered"
    df.loc[(df["Growth_Ratio"] >= 0.5) & (df["Growth_Ratio"] < 0.8), "Risk"] = "Vulnerable"

    endangered = df[df["Risk"] == "Endangered"]

    country_counts = endangered["Country"].value_counts().head(10)

    fig = px.bar(
        country_counts,
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

    with col1:
        pop_1970 = st.number_input("Population in 1970", min_value=0.0, value=100.0)

    with col2:
        pop_2020 = st.number_input("Population in 2020", min_value=0.0, value=80.0)

    change = pop_2020 - pop_1970
    growth_ratio = pop_2020 / (pop_1970 + 1)
    log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

    input_data = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Risk"):
        pred = model.predict(input_scaled)
        label = label_encoder.inverse_transform(pred)

        st.success(f"🌿 Predicted Risk: **{label[0]}**")
