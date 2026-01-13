from fastapi import APIRouter, UploadFile

from src.model import BreadClassifier
from torch import load
from src.services import InferencePipeline
from torchvision.datasets import ImageFolder
from torchvision import transforms

LENGHT = 200
WIDTH = 200

# NOTE: Will need to be refactored to decouple ImageFolder with the inference process.
train_transforms = transforms.Compose([
        transforms.Resize((LENGHT, WIDTH)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor()
    ])

TRAIN_SET = ImageFolder("data/training_data", transform=train_transforms)
PATH_MODEL = "ready_models/bakkery_model_v1.pth"
MODEL = BreadClassifier(input_shape=3, hidden_units=15, output_shape=3)
MODEL.load_state_dict(load(PATH_MODEL, weights_only=True))
MAP_LABELS = {v: k for k, v in TRAIN_SET.class_to_idx.items()}
inference_pipeline = InferencePipeline(MODEL, MAP_LABELS)

predict_router = APIRouter()

@predict_router.post("/predict")
async def predict(file: UploadFile):
    image: bytes = await file.read()
    end_results = inference_pipeline.live_predict(image)
    return end_results