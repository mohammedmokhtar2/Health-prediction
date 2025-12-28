from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize App
app = FastAPI(title="Vital Signs AI Monitor")

# --- LOAD ARTIFACTS ---
# We load these once when the app starts so it's fast
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    print("✅ Model, Scaler, and Encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Did you copy the .pkl files into the same folder?")

# --- DEFINE INPUT DATA ---
# This ensures the user sends exactly what we need
class VitalSigns(BaseModel):
    heart_rate: float
    blood_pressure: float
    oxygen_saturation: float
    respiratory_rate: float
    temperature: float

@app.get("/")
def home():
    return {"message": "Vital Signs AI is running. Send POST requests to /predict"}

@app.post("/predict")
def predict_condition(vitals: VitalSigns):
    try:
        # 1. Prepare Data
        # We must keep the EXACT same order as training: 
        # [heart_rate, blood_pressure, oxygen_saturation, respiratory_rate, temperature]
        input_data = np.array([[
            vitals.heart_rate,
            vitals.blood_pressure,
            vitals.oxygen_saturation,
            vitals.respiratory_rate,
            vitals.temperature
        ]])

        # 2. Scale Data (CRITICAL STEP)
        # The model expects scaled numbers (Z-scores), not raw values
        scaled_data = scaler.transform(input_data)

        # 3. Predict
        prediction_index = model.predict(scaled_data)

        # 4. Decode Label
        # Converts 0/1/2 back to "Safe", "Warning", "Critical"
        result_label = encoder.inverse_transform(prediction_index)[0]

        return {
            "prediction": result_label,
            "status_code": int(prediction_index[0]), # 0, 1, or 2
            "input_received": vitals
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))