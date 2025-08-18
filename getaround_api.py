import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel,Field
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import os

from getaround_schemas.price_predict import RentalPricePredictInput

description = """
🚗 

This app is made for predicting getaround rental price

# Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.

# Machine Learning

This is a Machine Learning endpoint that predict car rental price given some inputs. Here is the endpoint:

* `/predict` that accepts `RentalPricePredictInput`


Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Simple endpoints to try out!",
    },

    {
        "name": "Machine Learning",
        "description": "Prediction Endpoint."
    }
]

model = None
model_uri = None
best_run_id = None

app = FastAPI(
    title="Getaround api",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

# --- Validate required env vars ---
if not os.getenv("MLFLOW_TRACKING_URI"):
    raise RuntimeError("❌ MLFLOW_TRACKING_URI is not set. Please configure it in Hugging Face Space secrets.")

@app.on_event("startup")
def load_model_at_startup():
    """Load the best MLflow model when the API starts."""
    global model, model_uri, best_run_id    
    print("Starting API, loading model...")
    try:
        experiment_name = 'getaround_rental_price_predictor'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.rmse ASC"],  # or "metrics.neg_root_mean_squared_error DESC"
            max_results=1
        )

        if runs.empty:
            raise ValueError("No runs found for this experiment.")
        
        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print(f'✅ model_uri : {model_uri} loaded successfully.')
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
  


@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    **Simply returns a welcome message !**
    """
    message = "welcome to getaround api, check out documentation of the api at `/docs`"
    return message

@app.post("/predict", tags=["Machine Learning"])
def predict(input_data: RentalPricePredictInput):
    """
    **Prediction of rental price for a given RentalPriceInput**

       check request body schema below (RentalPriceInput)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        input_df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        
