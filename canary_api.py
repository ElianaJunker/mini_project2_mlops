from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import random
import pandas as pd

app = FastAPI()

model_name = "tracking-quickstart"
model_version = "1"
p = 0.8

current_model = None
next_model = None
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

class PredictRequest(BaseModel):
    data: list

@app.on_event("startup")
def load_initial_models():
    global current_model, next_model
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        current_model = mlflow.pyfunc.load_model(model_uri)
        next_model = current_model
        print(f"Loaded model {model_name} version {model_version} as both current and next models.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.post("/predict")
def predict(request: PredictRequest):
    global current_model, next_model, p
    try:
        if (random.random() < p):
            data = pd.DataFrame(request.data)
            predictions = current_model.predict(data).tolist()
        else:
            data = pd.DataFrame(request.data)
            predictions = next_model.predict(data).tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update-model")
def update_model(new_version: str):
    global next_model
    try:
        model_uri = f"models:/{model_name}/{new_version}"
        next_model = mlflow.pyfunc.load_model(model_uri)
        return {"message": f"Next model updated to version {new_version}, next model:{next_model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update next model: {e}")

@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model
    try:
        current_model = next_model
        return {"message": "Next model accepted as the current model"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to accept next model: {e}")
