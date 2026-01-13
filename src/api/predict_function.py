import io

import torch
from PIL import Image
from torch import Tensor, load, nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.domain import Prediction
from src.model import BreadClassifier

from ..services.training_pipeline import TrainingPipeline

LENGHT = 200
WIDTH = 200

PATH_MODEL = "ready_models/bakkery_model_v1.pth"
MODEL = BreadClassifier(input_shape=3, hidden_units=15, output_shape=3)
MODEL.load_state_dict(load(PATH_MODEL, weights_only=True))

test_transform = transforms.Compose([
        transforms.Resize((LENGHT, WIDTH)),
        transforms.ToTensor()
    ]) 

train_transforms = transforms.Compose([
        transforms.Resize((LENGHT, WIDTH)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor()
    ])
TRAIN_SET = ImageFolder("data/training_data", transform=train_transforms)
TEST_SET = ImageFolder("data/test_data", transform=test_transform)
MAP_LABELS = {v: k for k, v in TRAIN_SET.class_to_idx.items()}

# Mainly used 
def train_model():
    # Placeholder for loss function
    classifier = BreadClassifier(input_shape=3, hidden_units=15, output_shape=len(TRAIN_SET.classes))
    optimizer = optim.SGD(params=classifier.parameters(), lr=0.001, momentum=0, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()   
    pipeline = TrainingPipeline(epochs=4, optimizer=optimizer, loss_fn=loss_fn, bread_model=classifier, train_data=TRAIN_SET, test_data=TEST_SET)
    return pipeline.execute()

# Example function
def bytes_to_tensor(img_bytes)-> Tensor:
    pil_image = Image.open(io.BytesIO(img_bytes)) 
    tensor_image = test_transform(pil_image)
    ready_input = torch.unsqueeze(tensor_image, 0)
    return ready_input

def live_predict(input_data: bytes) -> dict:
    input_image = bytes_to_tensor(input_data)
    MODEL.eval()
    with torch.inference_mode():
        output = MODEL(input_image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the index of the highest probability
    confidence, index = torch.max(probabilities, 0)
    prediction_result = Prediction(
        prediction_nr=index.item(), 
        confidence_percentage=confidence.item() * 100,
        prediction_label=MAP_LABELS.get(index.item(), "Unknown"))
    return prediction_result