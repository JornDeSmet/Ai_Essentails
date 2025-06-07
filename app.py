from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoders
model = joblib.load("mushroom_model.pkl")
encoders = joblib.load("label_encoders.pkl")
expected_features = list(encoders.keys())

class MushroomInput(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: MushroomInput):
    raw_input = data.features

    # Filter to expected features
    filtered_input = {k: raw_input[k] for k in expected_features if k in raw_input}

    # Convert to DataFrame
    df = pd.DataFrame([filtered_input])

    # Encode
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)[0]
    result = "edible" if prediction == 1 else "poisonous"
    return {"prediction": result}
