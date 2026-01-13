from fastapi import APIRouter, UploadFile

from .predict_function import live_predict

predict_router = APIRouter()

@predict_router.post("/predict")
async def predict(file: UploadFile):
    image: bytes = await file.read()
    end_results = live_predict(image) # Placeholder
    return end_results