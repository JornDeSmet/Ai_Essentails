# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("mushroom_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define expected features (in correct order)
expected_features = [
    'cap-diameter','cap-shape','cap-surface','cap-color','does-bruise-or-bleed',
    'gill-attachment','gill-spacing','gill-color','stem-height','stem-width',
    'stem-root','stem-surface','stem-color','veil-type','veil-color',
    'has-ring','ring-type','spore-print-color','habitat','season'
]

# FastAPI app
app = FastAPI()

# Input schema
class MushroomInput(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "Mushroom classification API is live."}

@app.post("/predict")
def predict(input: MushroomInput):
    raw = input.features

    # Check for missing fields
    missing = [f for f in expected_features if f not in raw]
    if missing:
        return {"error": f"Missing fields: {missing}"}

    # Create DataFrame
    df = pd.DataFrame([raw])

    # Apply encoders to categorical columns
    for col in df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            if df[col][0] not in le.classes_:
                return {"error": f"Unexpected value for '{col}': {df[col][0]}"}
            df[col] = le.transform(df[col])

    # Predict
    prediction = model.predict(df)[0]
    label = "edible" if prediction == 1 else "poisonous"
    return {"prediction": label}
