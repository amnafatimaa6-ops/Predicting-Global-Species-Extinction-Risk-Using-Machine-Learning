import os
import numpy as np
import pickle

# ----------------------------
# SAFE MODEL LOADING
# ----------------------------
def load_model():

    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "..", "model", "model.pkl")
    scaler_path = os.path.join(base_path, "..", "model", "scaler.pkl")
    encoder_path = os.path.join(base_path, "..", "model", "label_encoder.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


# ----------------------------
# FEATURE ENGINEERING (MATCH TRAINING EXACTLY)
# ----------------------------
def create_features(pop_1970, pop_2020):

    change = pop_2020 - pop_1970
    growth_ratio = pop_2020 / (pop_1970 + 1)
    log_change = np.log1p(pop_2020) - np.log1p(pop_1970)

    return np.array([[pop_1970, pop_2020, change, growth_ratio, log_change]])


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_risk(model, scaler, label_encoder, input_data):

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)
    label = label_encoder.inverse_transform(pred)

    return label[0]
