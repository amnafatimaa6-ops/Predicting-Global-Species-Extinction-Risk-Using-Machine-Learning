import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import pycountry

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Biodiversity Intelligence System", layout="wide")

st.title("🌍 Biodiversity Intelligence Dashboard (LIVE GBIF)")
st.markdown("Real-time ecological intelligence using global species occurrence data")

# =========================
# SIDEBAR SPECIES CONTROL
# =========================
species_list = [
    "Panthera leo",
    "Panthera tigris",
    "Canis lupus",
    "Elephas maximus",
    "Ursus arctos",
    "Giraffa camelopardalis",
    "Homo sapiens",
    "Felis catus",
    "Bos taurus",
    "Equus zebra"
]

st.sidebar.header("🔎 Species Control Panel")

selected_species = st.sidebar.selectbox("Choose a species", species_list)

custom_species = st.sidebar.text_input("Or search manually")

if custom_species:
    selected_species = custom_species

# =========================
# LIVE GBIF DATA FETCH
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

# =========================
# CLEAN GEO (ISO STANDARD)
# =========================
def to_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

df["iso3"] = df["country"].apply(to_iso3)
df = df.dropna(subset=["iso3"])

df_clean = df.dropna(subset=["lat", "lon"])

# =========================
# 🗺️ CHOROPLETH MAP (WORLD BOUNDARIES)
# =========================
st.subheader(f"🗺️ Global Distribution (Choropleth): {selected_species}")

country_counts = df["iso3"].value_counts().reset_index()
country_counts.columns = ["iso3", "count"]

fig = px.choropleth(
    country_counts,
    locations="iso3",
    color="count",
    hover_name="iso3",
    color_continuous_scale="Viridis",
    title="Global Species Distribution by Country"
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 🌍 GEO SCATTER MAP (COLOURED)
# =========================
st.subheader("🌍 Point Distribution Map")

map_fig = px.scatter_geo(
    df_clean,
    lat="lat",
    lon="lon",
    color="country",
    hover_name="species",
    projection="natural earth",
    title="Global Biodiversity Points"
)

map_fig.update_traces(marker=dict(size=6, opacity=0.7))

map_fig.update_layout(height=600)

st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 🔥 HEATMAP (HOTSPOTS)
# =========================
st.subheader("🔥 Biodiversity Hotspot Heatmap")

heat_fig = px.density_map(
    df_clean,
    lat="lat",
    lon="lon",
    radius=12,
    center=dict(lat=20, lon=0),
    zoom=0,
    map_style="carto-darkmatter",
    title="Global Biodiversity Density Heatmap"
)

st.plotly_chart(heat_fig, use_container_width=True)

# =========================
# 📊 COUNTRY CLUSTERS
# =========================
st.subheader("📊 Top Biodiversity Countries")

top_countries = df["country"].value_counts().head(10)

cluster_df = pd.DataFrame({
    "country": top_countries.index,
    "count": top_countries.values
})

fig2 = px.bar(
    cluster_df,
    x="country",
    y="count",
    color="count",
    color_continuous_scale="Blues",
    title="Top 10 Countries by Observations"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# 🧠 ADVANCED INSIGHTS
# =========================
st.subheader("🧠 Ecological Intelligence Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Countries Covered (ISO)", df["iso3"].nunique())

with col3:
    st.metric("Unique Geo Points", df_clean[["lat","lon"]].drop_duplicates().shape[0])

with col4:
    score = len(df) / max(df["iso3"].nunique(), 1)
    st.metric("Data Density Score", round(score, 2))

# =========================
# INFO PANEL
# =========================
st.info("""
This system uses LIVE GBIF biodiversity data.
It performs:
- Real-time species mapping
- Country-level distribution analysis (ISO standard)
- Global hotspot detection
- Ecological density scoring

No dataset is stored locally — everything is fetched dynamically.
""")
