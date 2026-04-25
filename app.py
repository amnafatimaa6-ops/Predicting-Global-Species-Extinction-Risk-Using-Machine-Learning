import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Biodiversity Intelligence System", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard (LIVE GBIF)")
st.markdown("Real-time global species occurrence tracking using GBIF API")

# =========================
# SIDEBAR CONTROL PANEL
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
# LIVE DATA FETCH (GBIF API)
# =========================
@st.cache_data(ttl=3600)
def load_data(species_name):
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

df = load_data(selected_species)

# =========================
# ERROR HANDLING
# =========================
if df.empty:
    st.error("No data found for this species. Try another one.")
    st.stop()

df_clean = df.dropna(subset=["lat", "lon"])

# =========================
# 🌍 COLOURED WORLD MAP
# =========================
st.subheader(f"🌍 Global Distribution: {selected_species}")

map_fig = px.scatter_geo(
    df_clean,
    lat="lat",
    lon="lon",
    color="country",
    hover_name="species",
    projection="natural earth",
    title="Global Biodiversity Distribution (Colored by Country)"
)

map_fig.update_traces(marker=dict(size=6, opacity=0.7))

map_fig.update_layout(height=600)

st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 🔥 HEATMAP (HOTSPOT MODE)
# =========================
st.subheader("🔥 Biodiversity Hotspot Heatmap")

heat_fig = px.density_map(
    df_clean,
    lat="lat",
    lon="lon",
    radius=12,
    center=dict(lat=20, lon=0),
    zoom=0,
    map_style="carto-positron",
    title="Species Density Hotspots"
)

st.plotly_chart(heat_fig, use_container_width=True)

# =========================
# 🌎 COUNTRY ANALYSIS
# =========================
st.subheader("🌎 Country Distribution (Top 10)")

country_counts = df_clean["country"].value_counts().head(10)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(country_counts)

with col2:
    fig_pie = px.pie(
        values=country_counts.values,
        names=country_counts.index,
        title="Country Share of Observations"
    )
    st.plotly_chart(fig_pie)

# =========================
# 🧠 LIVE INSIGHTS
# =========================
st.subheader("🧠 Live Ecological Intelligence")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df_clean))

with col2:
    st.metric("Countries Covered", df_clean["country"].nunique())

with col3:
    st.metric("Unique Coordinates", df_clean[["lat", "lon"]].drop_duplicates().shape[0])

with col4:
    score = len(df_clean) / max(df_clean["country"].nunique(), 1)
    st.metric("Data Spread Score", round(score, 2))

# =========================
# 📌 INFO PANEL
# =========================
st.info("""
This dashboard uses LIVE GBIF biodiversity occurrence data.
No dataset is stored locally — all insights are generated in real time.

It visualizes:
- Species distribution
- Geographic hotspots
- Country-level biodiversity spread
- Live ecological metrics
""")
