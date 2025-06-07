# app.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("mushroom_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")  # <== Add this!

expected_features = list(encoders.keys())

class MushroomInput(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: MushroomInput):
    raw_input = data.features

    # Filter to expected and known features only
    filtered = {k: raw_input[k] for k in expected_features if k in raw_input}

    df = pd.DataFrame([filtered])

    # Encode categorical features
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # Reindex to ensure correct feature order (add missing if needed)
    df = df.reindex(columns=feature_order, fill_value=0)

    prediction = model.predict(df)[0]
    return {"prediction": "edible" if prediction == 1 else "poisonous"}
