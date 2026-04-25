import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Biodiversity Risk Dashboard", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard")
st.markdown("Machine Learning + Ecological Trend Analysis of Species Population Decline")

# ======================
# LOAD DATA (ONLINE SOURCE)
# ======================
@st.cache_data
def load_data():
    # Replace this with ANY online raw CSV link
    url = "https://your-online-source/pyramids_or_biodiversity_dataset.csv"
    df = pd.read_csv(url)

    # Clean column names
    df.columns = df.columns.str.strip()

    return df


df = load_data()

# ======================
# SIDEBAR FILTERS
# ======================
st.sidebar.header("🔎 Filters")

if "Country" in df.columns:
    country = st.sidebar.multiselect("Country", df["Country"].dropna().unique())

if "Class" in df.columns:
    animal_class = st.sidebar.multiselect("Class", df["Class"].dropna().unique())

# Apply filters safely
filtered_df = df.copy()

if "Country" in df.columns and country:
    filtered_df = filtered_df[filtered_df["Country"].isin(country)]

if "Class" in df.columns and animal_class:
    filtered_df = filtered_df[filtered_df["Class"].isin(animal_class)]

st.subheader("📊 Dataset Overview")
st.write(filtered_df.head())

# ======================
# POPULATION TREND (IF YEARS EXIST)
# ======================
year_cols = [col for col in filtered_df.columns if str(col).isdigit()]

if len(year_cols) > 0:
    st.subheader("📈 Population Trend Over Time")

    df_long = filtered_df.melt(
        id_vars=["Binomial"] if "Binomial" in filtered_df.columns else [],
        value_vars=year_cols,
        var_name="Year",
        value_name="Population"
    )

    df_long["Year"] = pd.to_numeric(df_long["Year"])
    df_long["Population"] = pd.to_numeric(df_long["Population"], errors="coerce")

    trend = df_long.groupby("Year")["Population"].mean().reset_index()

    fig = px.line(trend, x="Year", y="Population", title="Average Population Trend")
    st.plotly_chart(fig, use_container_width=True)

# ======================
# RISK CLASSIFICATION MODEL
# ======================
st.subheader("⚠️ Species Risk Prediction Model")

if "1970" in filtered_df.columns and "2020" in filtered_df.columns:

    ml_df = filtered_df.copy()

    ml_df = ml_df.dropna(subset=["1970", "2020"])

    ml_df["Change"] = ml_df[2020] - ml_df[1970]
    ml_df["Growth_Ratio"] = ml_df[2020] / (ml_df[1970] + 1)
    ml_df["Log_Change"] = np.log1p(ml_df[2020]) - np.log1p(ml_df[1970])

    # Risk label
    def label_risk(x):
        if x > -0.2:
            return 0  # Stable
        elif x > -0.5:
            return 1  # Vulnerable
        else:
            return 2  # Endangered

    ml_df["Risk"] = (ml_df["Growth_Ratio"] - 1).apply(label_risk)

    features = ["1970", "2020", "Change", "Growth_Ratio", "Log_Change"]

    X = ml_df[features]
    y = ml_df["Risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    ml_df["Prediction"] = model.predict(X_scaled)

    st.write("Model trained on current dataset snapshot.")

    fig2 = px.histogram(ml_df, x="Prediction", title="Predicted Risk Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ======================
# COUNTRY INSIGHTS
# ======================
if "Country" in filtered_df.columns:

    st.subheader("🌎 Country-wise Distribution")

    country_counts = filtered_df["Country"].value_counts().head(10)

    fig3 = px.bar(
        country_counts,
        x=country_counts.index,
        y=country_counts.values,
        title="Top Countries in Dataset"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ======================
# FINAL INSIGHT BLOCK
# ======================
st.subheader("🧠 Key Insight")

st.info("""
This dashboard reveals two realities:

1. Nature is changing through population decline trends  
2. But data itself is uneven — some regions are heavily monitored, others underreported  

So what we see is not just ecology…  
it's ecology filtered through human observation systems.
""")
