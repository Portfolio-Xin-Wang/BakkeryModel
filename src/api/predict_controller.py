from fastapi import APIRouter, UploadFile, Form
from pydantic import BaseModel
from .predict_function import live_predict
import io
from PIL import Image

predict_router = APIRouter()
class PredictionInputDTO(BaseModel):
    hero: str 

# Receive a DTO: With predicted values and a image file. It should return a online prediction and current predicted value.
# Based on the input values, it should perhaps return secondary predictions.
@predict_router.post("/predict")
async def predict(file: UploadFile, help: str = Form(...)):
    print(f"Received help text: {help}")
    image = await file.read() # Placeholder
    img = Image.open(io.BytesIO(image))  # Example conversion
    print("Image loaded in PIL.")
    # Use prediction function to get results
    end_results = live_predict(img) # Placeholder

    return end_results