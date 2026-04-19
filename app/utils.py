import pickle
import pandas as pd
import streamlit as st
import os

@st.cache_resource
def load_model():
    try:
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("model/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return model, scaler, label_encoder

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None


@st.cache_data
def load_data():
    """
    Streamlit-safe dataset loader
    """
    try:
        # OPTION 1: local repo file
        if os.path.exists("data/data.csv"):
            df = pd.read_csv("data/data.csv")
            return df

        # OPTION 2: fallback (replace with your GitHub raw link later)
        url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data/data.csv"
        df = pd.read_csv(url)
        return df

    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None
