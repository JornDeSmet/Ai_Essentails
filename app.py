from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("mushroom_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize app
app = FastAPI()

# Define the expected input format
class MushroomInput(BaseModel):
    features: dict  # expects a dictionary of raw feature names and string values

@app.get("/")
def read_root():
    return {"message": "Mushroom classifier is live!"}

@app.post("/predict")
def predict(data: MushroomInput):
    input_dict = data.features

    # Create a DataFrame from the input
    df = pd.DataFrame([input_dict])

    # Apply LabelEncoders
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        else:
            return {"error": f"Unexpected feature: {col}"}

    # Predict
    prediction = model.predict(df)
    result = "edible" if prediction[0] == 1 else "poisonous"

    return {"prediction": result}
