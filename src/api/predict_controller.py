from fastapi import APIRouter, Form, UploadFile

from .predict_function import live_predict

predict_router = APIRouter()

@predict_router.post("/predict")
async def predict(file: UploadFile, help: str = Form(...)):
    image: bytes = await file.read()
    end_results = live_predict(image) # Placeholder
    return end_results