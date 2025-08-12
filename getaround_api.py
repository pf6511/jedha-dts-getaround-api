import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel,Field
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import os

description = """
🚗 

This app is made for predicting getaround rental price

## Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.

## Machine Learning

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


app = FastAPI(
    title="Getaround api",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

# --- Validate required env vars ---
if not os.getenv("MLFLOW_TRACKING_URI"):
    raise RuntimeError("❌ MLFLOW_TRACKING_URI is not set. Please configure it in Hugging Face Space secrets.")

# --- Load best model at startup ---
try:
    experiment_name = 'getaround_rental_price_predictor'
    experiment = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],  # or "metrics.neg_root_mean_squared_error DESC"
        max_results=1
    )

    best_run_id = runs.iloc[0]["run_id"]
    print(f'best_run_id : {best_run_id}')
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f'model_uri : {model_uri} loaded successfully.')
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

class RentalPricePredictInput(BaseModel):
    model_key:str
    mileage: int = Field(ge=0,description="Mileage must be non-negative")
    engine_power: int = Field(ge=0, description="Engine power must be non-negative")
    fuel: Literal["diesel", "petrol", "electro", "hybrid_petrol"]
    paint_color: Literal["green","silver","blue","grey","brown","black","red","beige","white","orange"]
    car_type:Literal["subcompact","estate","hatchback","van","sedan","convertible","suv","coupe"]
    private_parking_available:bool
    has_gps:bool	
    has_air_conditioning:bool	
    automatic_car:bool	
    has_getaround_connect:bool	
    has_speed_regulator:bool
    winter_tires:bool
  

@app.post("/predict", tags=["Machine Learning"])
def predict(input_data: RentalPricePredictInput):
    """
    Prediction of rental price for a given RentalPriceInput
    """
    try:
        input_df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}
        print("Model loaded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        


@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "welcome to getaround api, check out documentation of the api at `/docs`"
    return message
