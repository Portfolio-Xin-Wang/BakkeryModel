import io

from fastapi import APIRouter, Form, UploadFile
from PIL import Image

from .predict_function import live_predict

predict_router = APIRouter()

@predict_router.post("/predict")
async def predict(file: UploadFile, help: str = Form(...)):
    image = await file.read() # Placeholder
    img = Image.open(io.BytesIO(image))  # Example conversion
    end_results = live_predict(img) # Placeholder

    return end_results