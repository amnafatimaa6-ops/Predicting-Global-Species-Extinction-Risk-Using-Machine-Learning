import pandas as pd
import joblib
import gdown

# -------------------------
# Load dataset from Google Drive
# -------------------------
def load_data():
    file_id = "1Ob1WSj2jntkLOKoSMR1MuhQpL0gaItoK"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    output = "data.csv"
    gdown.download(url, output, quiet=True)
    
    df = pd.read_csv("data.csv")
    return df


# -------------------------
# Load ML artifacts
# -------------------------
def load_model():
    return joblib.load("model/model.pkl")

def load_scaler():
    return joblib.load("model/scaler.pkl")

def load_encoder():
    return joblib.load("model/label_encoder.pkl")
