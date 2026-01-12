from torch import nn, optim, load
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import BreadClassifier
from ..services.training_pipeline import TrainingPipeline

LENGHT = 200
WIDTH = 200

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
def live_predict(input_data) -> dict:
    PATH_MODEL = "ready_models/bakkery_model_v1.pth"
    # Placeholder for actual prediction logic
    # In a real implementation, this would involve loading a trained model
    # and using it to make predictions on the input_data.
    imported_model = load(PATH_MODEL, weights_only=True)
    image = test_transform(input_data)
    model = BreadClassifier(input_shape=3, hidden_units=15, output_shape=3)
    model.load_state_dict(imported_model)
    ready_input = torch.unsqueeze(image, 0)
    model.eval()
    with torch.inference_mode():
        output = model(ready_input)
        print(output)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get the index of the highest probability
    confidence, index = torch.max(probabilities, 0)
    prediction_result = {
        "input": index.item(),
        "prediction": "dummy_bread_type", # Some mapping of prediction to business data
        "confidence": confidence.item()
    }
    
    return prediction_result