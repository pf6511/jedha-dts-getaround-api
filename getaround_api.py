import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel,Field
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
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
    openapi_tags=tags_metadata,
    root_path=os.getenv("ROOT_PATH", "")
)

experiment_name = 'getaround_rental_price_predictor'
experiment = mlflow.get_experiment_by_name(experiment_name)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],  # or "metrics.neg_root_mean_squared_error DESC"
    max_results=1
)

best_run_id = runs.iloc[0]["run_id"]
print(f'best_run_id : {best_run_id}')

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
    model_uri = f"runs:/{best_run_id}/model"
    print(f'model_uri : {model_uri}')
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded.")
    except Exception as e:
        print(f"An error occurred when loading model : {e}")
        
    input_df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(input_df)
    return {"prediction": float(prediction[0])}

@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message
