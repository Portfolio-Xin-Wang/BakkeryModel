from fastapi import APIRouter, UploadFile

from src.model import BreadClassifier
from torch import load
from src.services import InferencePipeline, LabelManager
from src.const import PATH_MODEL, FOLDER_MODEL

manager = LabelManager(location=FOLDER_MODEL)
# GLOBAL
MODEL = BreadClassifier(input_shape=3, hidden_units=15, output_shape=3)
MODEL.load_state_dict(load(PATH_MODEL, weights_only=True))
# LOCAL
inference_pipeline = InferencePipeline(MODEL, manager)

predict_router = APIRouter()

@predict_router.post("/predict")
async def predict(file: UploadFile):
    image: bytes = await file.read()
    end_results = inference_pipeline.live_predict(image)
    return end_results