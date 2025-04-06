from fastapi import FastAPI
from body_type_model import BodyTypeModel, BodyMeasurements, PredictionResult, create_fastapi_app

# Create the FastAPI app using the function from body_type_model.py
app = create_fastapi_app()

# The app is now ready with the /predict endpoint