import streamlit as st
import joblib

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")

        return model, scaler, le

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None
