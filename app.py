from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("mushroom_model.pkl")

# Define the expected features from training
expected_features = [
    "cap-diameter", "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed",
    "gill-attachment", "gill-spacing", "gill-color", "stem-height", "stem-width",
    "stem-surface", "stem-color", "has-ring", "ring-type", "habitat", "season"
]

# Define the input data schema
class MushroomInput(BaseModel):
    features: dict

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Mushroom classifier is live!"}

@app.post("/predict")
def predict(data: MushroomInput):
    # Filter only the expected features (ignore any extras)
    input_data = {key: data.features[key] for key in expected_features if key in data.features}

    # Ensure all expected features are present
    if set(input_data.keys()) != set(expected_features):
        missing = list(set(expected_features) - set(input_data.keys()))
        return {"error": f"Missing required features: {missing}"}

    # Convert to DataFrame for prediction
    df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(df)[0]
    result = "edible" if prediction == 1 else "poisonous"
    return {"prediction": result}
