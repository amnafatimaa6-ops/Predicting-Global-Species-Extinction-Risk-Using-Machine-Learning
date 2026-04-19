import numpy as np
import pandas as pd
import pickle

# ----------------------------
# LOAD MODEL COMPONENTS
# ----------------------------
def load_model():
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
    return model, scaler, label_encoder


# ----------------------------
# FEATURE ENGINEERING (MATCH NOTEBOOK EXACTLY)
# ----------------------------
def create_features(pop_1970, pop_2020):

    change = pop_2020 - pop_1970
    growth_ratio = pop_2020 / (pop_1970 + 1)
    log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

    return np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])


# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict_risk(model, scaler, label_encoder, input_data):

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)
    label = label_encoder.inverse_transform(pred)

    return label[0]
