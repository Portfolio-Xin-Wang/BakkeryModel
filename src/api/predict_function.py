import io

import torch
from PIL import Image
from torch import load, nn, optim, Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import BreadClassifier
from src.domain import Prediction

from ..services.training_pipeline import TrainingPipeline

LENGHT = 200
WIDTH = 200
MAP_LABELS = {}
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

def _transform():
    return train_transforms, test_transform

# Mainly used 
def train_model():
    train_transforms, test_transform = _transform()
    train_set = ImageFolder("data/training_data", transform=train_transforms)
    test_set = ImageFolder("data/test_data", transform=test_transform)
      # Placeholder for loss function
    classifier = BreadClassifier(input_shape=3, hidden_units=15, output_shape=len(train_set.classes))
    optimizer = optim.SGD(params=classifier.parameters(), lr=0.001, momentum=0, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()   
    pipeline = TrainingPipeline(epochs=4, optimizer=optimizer, loss_fn=loss_fn, bread_model=classifier, train_data=train_set, test_data=test_set)
    return pipeline.execute()

# Example function
def bytes_to_tensor(img_bytes)-> Tensor:
    pil_image = Image.open(io.BytesIO(img_bytes)) 
    tensor_image = test_transform(pil_image)
    ready_input = torch.unsqueeze(tensor_image, 0)
    return ready_input

def live_predict(input_data: bytes) -> dict:
    input_image = bytes_to_tensor(input_data)  # Example conversion
    MODEL.eval()
    with torch.inference_mode():
        output = MODEL(input_image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the index of the highest probability
    confidence, index = torch.max(probabilities, 0)
    prediction_result = Prediction(prediction_nr=index.item(), confidence_percentage=confidence.item())
    return prediction_result