import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model + preprocessing tools
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Biodiversity Risk Predictor", layout="centered")

st.title("🌿 Biodiversity Risk Prediction System")
st.write("Predict extinction risk based on population trends (1970 vs 2020).")

# Input section
st.sidebar.header("Enter Species Data")

pop_1970 = st.sidebar.number_input("Population in 1970", min_value=0.0)
pop_2020 = st.sidebar.number_input("Population in 2020", min_value=0.0)

# Feature engineering (same as training)
change = pop_2020 - pop_1970
growth_ratio = pop_2020 / (pop_1970 + 1)
log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

input_data = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Risk 🚨"):
    pred = model.predict(input_scaled)
    risk = le.inverse_transform(pred)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Risk Level: {risk}")

    if risk == "Endangered":
        st.error("⚠️ High extinction risk detected")
    elif risk == "Vulnerable":
        st.warning("⚠️ Moderate risk detected")
    else:
        st.info("🟢 Species appears stable")
