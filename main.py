from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# 1. Initialize the App
app = FastAPI(title="Vital Signs AI Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


artifacts = {}

@app.on_event("startup")
def load_artifacts():
    try:
        
        artifacts["model"] = joblib.load("model.pkl")
        artifacts["scaler"] = joblib.load("scaler.pkl")
        artifacts["encoder"] = joblib.load("encoder.pkl")
        print("✅ Artifacts loaded successfully: model.pkl, scaler.pkl, encoder.pkl")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find model files! {e}")
        print("Make sure model.pkl, scaler.pkl, and encoder.pkl are in the SAME folder as main.py")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")


class VitalSigns(BaseModel):
    heart_rate: float
    blood_pressure: float
    oxygen_saturation: float
    respiratory_rate: float
    temperature: float


@app.get("/")
def home():
    return {"message": "Vital Signs AI is RUNNING. Send a POST request to /predict to use it."}


@app.post("/predict")
def predict_condition(vitals: VitalSigns):
    
    if "model" not in artifacts:
        raise HTTPException(status_code=500, detail="Model files not loaded. Check server logs.")

    try:
        
        input_data = np.array([[
            vitals.heart_rate,
            vitals.blood_pressure,
            vitals.oxygen_saturation,
            vitals.respiratory_rate,
            vitals.temperature
        ]])

        
        scaler = artifacts["scaler"]
        scaled_data = scaler.transform(input_data)

        
        model = artifacts["model"]
        prediction_index = model.predict(scaled_data) # Returns [0], [1], or [2]

        
        encoder = artifacts["encoder"]
        result_label = encoder.inverse_transform(prediction_index)[0]

        
        return {
            "prediction": result_label,          # "Safe", "Warning", or "Critical"
            "status_code": int(prediction_index[0]), # 0, 1, or 2 (useful for hardware logic)
            "input_received": vitals
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))