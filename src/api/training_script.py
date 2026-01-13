from torch import load, nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import BreadClassifier
from src.services import TrainingPipeline

LENGHT = 200
WIDTH = 200 

PATH_MODEL = "ready_models/bakkery_model_v1.pth"

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