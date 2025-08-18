import requests 
import json

url = "http://localhost:7860/predict"

def test_predict():   
    payload = {
        "model_key": "Honda",
        "mileage": 120000,
        "engine_power": 150,
        "fuel": "diesel",
        "paint_color" : "white",
        "car_type" : "hatchback",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning":True,
        "automatic_car": False,
        "has_getaround_connect": False,
        "has_speed_regulator": True,
        "winter_tires": True
        }

    #response = requests.post(url, data = json.dumps(payload))
    response = requests.post(url,json=payload)
    print(f'response.status_code : {response.status_code}')
    print(f'prediction for input : {payload}')
    print(f'prediction : {response.json()["prediction"]}')

test_predict()