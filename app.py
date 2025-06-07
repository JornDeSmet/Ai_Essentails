# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("mushroom_model.pkl")

# Initialize API
app = FastAPI()

# Define input schema
class MushroomInput(BaseModel):
    features: list  # expects a list of numerical values

@app.get("/")
def read_root():
    return {"message": "Mushroom classifier is live!"}

@app.post("/predict")
def predict(data: MushroomInput):
    prediction = model.predict([data.features])
    result = "edible" if prediction[0] == 1 else "poisonous"
    return {"prediction": result}