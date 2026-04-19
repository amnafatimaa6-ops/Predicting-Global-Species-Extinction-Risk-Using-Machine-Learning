import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from utils import load_assets

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Endangered Species AI", layout="wide")

# ---------------- LOAD MODEL ----------------
model, scaler, le = load_assets()

# ---------------- TITLE ----------------
st.title("🌍 Endangered Species Risk Intelligence System")

st.markdown("""
Predict extinction risk using population trends and ecological change signals.
""")

if model is None:
    st.stop()

# ---------------- INPUT ----------------
st.sidebar.header("🔍 Input Features")

pop_1970 = st.sidebar.number_input("Population in 1970", min_value=1.0, value=100.0)
pop_2020 = st.sidebar.number_input("Population in 2020", min_value=1.0, value=80.0)

# ---------------- FEATURE ENGINEERING ----------------
change = pop_2020 - pop_1970
growth_ratio = pop_2020 / (pop_1970 + 1)
log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

X = np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])
X_scaled = scaler.transform(X)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Risk"):
    pred = model.predict(X_scaled)
    label = le.inverse_transform(pred)[0]

    st.success(f"🧠 Predicted Risk: **{label}**")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

features = ["1970", "2020", "Change", "Growth Ratio", "Log Change"]

importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_title("Feature Importance")
st.pyplot(fig)

# ---------------- MODEL INSIGHT ----------------
st.subheader("⚖️ Model Insight")

models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
scores = [0.958, 0.916, 0.958]

fig2 = px.bar(x=models, y=scores, text=scores, title="Model Accuracy Comparison")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER INSIGHT ----------------
st.markdown("""
### 🧠 Insight
- Growth ratio is strongest predictor  
- Model accuracy ~95%  
- Decline rate matters more than raw population  
""")
