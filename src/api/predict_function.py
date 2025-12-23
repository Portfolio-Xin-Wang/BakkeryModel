from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import BreadClassifier
from ..services.training_pipeline import TrainingPipeline


def _transform(length: int, width: int):
    test_transform = transforms.Compose([
        transforms.Resize((length, width)),
        transforms.ToTensor()
    ]) 

    train_transforms = transforms.Compose([
        transforms.Resize((length, width)),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAutocontrast(),
        transforms.RandomInvert(),
        transforms.ToTensor()
    ])
    return train_transforms, test_transform

def train_model():
    train_transforms, test_transform = _transform(300, 300)
    train_set = ImageFolder("data/custom", transform=train_transforms)
    test_set = ImageFolder("data/test", transform=test_transform)
      # Placeholder for loss function
    classifier = BreadClassifier(input_shape=270000, hidden_units=10, output_shape=len(train_set.classes))  # Placeholder for model
    optimizer = optim.SGD(params=classifier.parameters(), lr=0.025)  # Placeholder for optimizer
    loss_fn = nn.CrossEntropyLoss()   
    pipeline = TrainingPipeline(epochs=5, optimizer=optimizer, loss_fn=loss_fn, bread_model=classifier, train_data=train_set, test_data=test_set)
    pipeline.execute()

# # Example function
# def live_predict(input_data: dict) -> dict:
#     """
#     Function to make live predictions on input data.

#     Args:
#         input_data (dict): A dictionary containing the input features for prediction.

#     Returns:
#         dict: A dictionary containing the prediction results.
#     """
#     # Placeholder for actual prediction logic
#     # In a real implementation, this would involve loading a trained model
#     # and using it to make predictions on the input_data.
    
#     # For demonstration purposes, we'll return a dummy prediction.
#     prediction_result = {
#         "input": input_data,
#         "prediction": "dummy_bread_type",
#         "confidence": 0.95
#     }
    
#     return prediction_result

train_model()