from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import os

app = FastAPI()
model_name = "tracking-quickstart"
model_version = "1"
model = None
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

class PredictRequest(BaseModel):
    data: list

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model {model_name} version {model_version}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.post("/predict")
def predict(request: PredictRequest):
    global model
    try:
        data = pd.DataFrame(request.data)
        predictions = model.predict(data).tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update-model")
def update_model(new_version: str):
    global model, model_version
    try:
        model_uri = f"models:/{model_name}/{new_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = new_version
        return {"message": f"Model updated to version {new_version}, model : {model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {e}")