from fastapi import APIRouter

predict_router = APIRouter()

@predict_router.get("/predict")
async def predict():
    return {"message": "This is the predict endpoint."}