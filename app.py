import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Biodiversity Intelligence System", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard (LIVE GBIF)")
st.markdown("Real-time global species occurrence tracking system")

# =========================
# PRESET SPECIES (SIDEBAR)
# =========================
species_list = [
    "Panthera leo",
    "Panthera tigris",
    "Elephas maximus",
    "Canis lupus",
    "Ursus arctos",
    "Giraffa camelopardalis",
    "Bos taurus",
    "Homo sapiens",
    "Equus zebra",
    "Felis catus"
]

st.sidebar.header("🔎 Species Control Panel")

selected_species = st.sidebar.selectbox("Choose a species", species_list)

custom_species = st.sidebar.text_input("Or search manually")

if custom_species:
    selected_species = custom_species

# =========================
# LIVE API FUNCTION
# =========================
@st.cache_data(ttl=3600)
def get_data(species_name):

    url = f"https://api.gbif.org/v1/occurrence/search?scientificName={species_name}&limit=300"
    res = requests.get(url)

    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json()

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

df = get_data(selected_species)

# =========================
# ERROR HANDLING
# =========================
if df.empty:
    st.error("No data found for this species. Try another one.")
    st.stop()

# =========================
# WORLD MAP (FIXED GLOBAL VIEW)
# =========================
st.subheader(f"🌍 Global Distribution: {selected_species}")

map_fig = px.scatter_geo(
    df.dropna(),
    lat="lat",
    lon="lon",
    hover_name="species",
    title="Worldwide Occurrence Map",
    projection="natural earth"
)

map_fig.update_layout(height=600)

st.plotly_chart(map_fig, use_container_width=True)

# =========================
# COUNTRY ANALYSIS
# =========================
st.subheader("🌎 Country Distribution (Top 10)")

country_counts = df["country"].value_counts().dropna().head(10)

bar_fig = px.bar(
    x=country_counts.index,
    y=country_counts.values,
    labels={"x": "Country", "y": "Sightings"},
    title="Top Countries with Observations"
)

st.plotly_chart(bar_fig, use_container_width=True)

# =========================
# INSIGHTS PANEL
# =========================
st.subheader("🧠 Live Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Countries Covered", df["country"].nunique())

with col3:
    st.metric("Latest Year", int(df["year"].max()) if df["year"].notna().any() else "N/A")

st.info("""
This system pulls LIVE biodiversity occurrence records from GBIF.
It dynamically visualizes species distribution without storing any dataset locally.
""")
