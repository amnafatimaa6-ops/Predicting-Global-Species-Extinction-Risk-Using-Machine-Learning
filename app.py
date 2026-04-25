import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# =========================
# 🌌 NASA-STYLE UI CONFIG
# =========================
st.set_page_config(
    page_title="BIO-INTEL // EARTH MONITORING SYSTEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body {
        background-color: #0B0F19;
        color: #00FFAA;
    }

    .stApp {
        background-color: #0B0F19;
        color: #00FFAA;
    }

    h1, h2, h3 {
        color: #00FFAA;
        font-family: 'Courier New';
    }

    .stMetric {
        background-color: #111827;
        border: 1px solid #00FFAA;
        padding: 10px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌍 BIO-INTEL SYSTEM // LIVE EARTH MONITORING")

st.markdown("🛰️ Real-time biodiversity surveillance powered by GBIF API")

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
    "Homo sapiens",
    "Felis catus"
]

st.sidebar.header("🛰️ CONTROL PANEL")

selected_species = st.sidebar.selectbox("Select Species", species_list)

custom = st.sidebar.text_input("Manual Input")

if custom:
    selected_species = custom

# =========================
# LIVE DATA
# =========================
@st.cache_data(ttl=3600)
def fetch_data(species):
    url = f"https://api.gbif.org/v1/occurrence/search?scientificName={species}&limit=300"
    r = requests.get(url)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()

    records = []

    for i in data.get("results", []):
        records.append({
            "species": i.get("species"),
            "country": i.get("country"),
            "year": i.get("year"),
            "lat": i.get("decimalLatitude"),
            "lon": i.get("decimalLongitude")
        })

    return pd.DataFrame(records)

df = fetch_data(selected_species)

if df.empty:
    st.error("⚠️ SIGNAL LOST — No data available")
    st.stop()

df = df.dropna(subset=["lat", "lon"])

# =========================
# 🌍 GLOBAL MAP (NASA STYLE)
# =========================
st.subheader(f"🌍 GLOBAL TRACKING: {selected_species}")

map_fig = px.scatter_geo(
    df,
    lat="lat",
    lon="lon",
    color="country",
    projection="natural earth",
    title="🛰️ Earth Biodiversity Grid View"
)

map_fig.update_traces(marker=dict(size=5, opacity=0.8))

map_fig.update_layout(
    paper_bgcolor="#0B0F19",
    geo_bgcolor="#0B0F19",
    font_color="#00FFAA",
    height=600
)

st.plotly_chart(map_fig, use_container_width=True)

# =========================
# 🔥 HOTSPOT MAP
# =========================
st.subheader("🔥 ECOLOGICAL HEAT SIGNATURE")

heat_fig = px.density_map(
    df,
    lat="lat",
    lon="lon",
    radius=10,
    center=dict(lat=20, lon=0),
    zoom=0,
    map_style="carto-darkmatter"
)

st.plotly_chart(heat_fig, use_container_width=True)

# =========================
# 🚨 ENDANGERED ALERT SYSTEM
# =========================
st.subheader("🚨 CONSERVATION ALERT SYSTEM")

df["year"] = pd.to_numeric(df["year"], errors="coerce")

trend = df.groupby("year").size().reset_index(name="observations")

if len(trend) > 3:

    slope = (trend["observations"].iloc[-1] - trend["observations"].iloc[0]) / len(trend)

    if slope <= -3:
        alert = "🔴 CRITICAL: EXTINCTION RISK HIGH"
        color = "red"
    elif slope < 0:
        alert = "🟡 WARNING: POPULATION DECLINING"
        color = "orange"
    else:
        alert = "🟢 STABLE ECOLOGICAL STATUS"
        color = "green"

    st.markdown(f"### <span style='color:{color}'>{alert}</span>", unsafe_allow_html=True)

# =========================
# 📊 COUNTRY INTELLIGENCE
# =========================
st.subheader("🌎 TERRITORIAL DISTRIBUTION GRID")

country_counts = df["country"].value_counts().head(10)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(country_counts)

with col2:
    fig = px.pie(values=country_counts.values, names=country_counts.index)
    fig.update_layout(
        paper_bgcolor="#0B0F19",
        font_color="#00FFAA"
    )
    st.plotly_chart(fig)

# =========================
# 🧠 SYSTEM METRICS
# =========================
st.subheader("🧠 SYSTEM STATUS")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RECORDS SCANNED", len(df))

with col2:
    st.metric("COUNTRIES DETECTED", df["country"].nunique())

with col3:
    st.metric("COORDINATE NODES", df[["lat","lon"]].drop_duplicates().shape[0])

# =========================
# 📡 FOOTER
# =========================
st.markdown("""
🛰️ BIO-INTEL SYSTEM ACTIVE  
🌍 Monitoring planetary biodiversity in real-time  
⚡ No static dataset — only live ecological signals
""")
