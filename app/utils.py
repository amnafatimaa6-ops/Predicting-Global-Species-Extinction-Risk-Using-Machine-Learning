import pickle
import pandas as pd
import streamlit as st

# ---------------- MODEL LOADING ----------------
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


# ---------------- DATA LOADING (DRIVE FIXED) ----------------
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?id=1Ob1WSj2jntkLOKoSMR1MuhQpL0gaItoK"
        df = pd.read_csv(url)
        return df

    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None
