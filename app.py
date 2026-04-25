import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Biodiversity Intelligence System", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard (LIVE)")
st.markdown("Real-time species occurrence data using GBIF API")

# =========================
# LIVE DATA FUNCTION
# =========================
@st.cache_data(ttl=3600)
def load_live_data(species_name):

    url = f"https://api.gbif.org/v1/occurrence/search?scientificName={species_name}&limit=300"
    response = requests.get(url)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    records = []

    for item in data.get("results", []):
        records.append({
            "species": item.get("species"),
            "country": item.get("country"),
            "year": item.get("year"),
            "lat": item.get("decimalLatitude"),
            "lon": item.get("decimalLongitude")
        })

    return pd.DataFrame(records)

# =========================
# USER INPUT
# =========================
species = st.text_input("🔎 Enter species name", "Panthera leo")

df = load_live_data(species)

# =========================
# SAFETY CHECK
# =========================
if df.empty:
    st.error("No live data found. Try another species name.")
    st.stop()

# =========================
# MAP VISUALIZATION
# =========================
st.subheader(f"📍 Live Sightings Map: {species}")

fig = px.scatter_geo(
    df,
    lat="lat",
    lon="lon",
    hover_name="species",
    title="Global Occurrence Map"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# COUNTRY ANALYSIS
# =========================
st.subheader("🌎 Country Distribution")

country_counts = df["country"].value_counts().dropna().head(10)

fig2 = px.bar(
    country_counts,
    x=country_counts.index,
    y=country_counts.values,
    title="Top Countries (Live Observations)"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# BASIC INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("Total Records Found:", len(df))
st.write("Countries Covered:", df["country"].nunique())
st.write("Years Range:", df["year"].min(), "to", df["year"].max())

st.info("""
This dashboard uses live biodiversity occurrence data from GBIF.
It does NOT rely on stored datasets — everything is fetched in real time.
""")
