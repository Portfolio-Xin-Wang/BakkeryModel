from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import BreadClassifier
from src.services import TrainingPipeline, LabelManager
from src.const import IMG_LENGHT, IMG_WIDTH, TRAIN_DATA_BREAD_CLASSIFIER, TEST_DATA_BREAD_CLASSIFIER

test_transform = transforms.Compose([
        transforms.Resize((IMG_LENGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ]) 

train_transforms = transforms.Compose([
        transforms.Resize((IMG_LENGHT, IMG_WIDTH)),
        transforms.RandomAutocontrast(),
        transforms.ToTensor()
    ])

TRAIN_SET = ImageFolder(TRAIN_DATA_BREAD_CLASSIFIER, transform=train_transforms)
TEST_SET = ImageFolder(TEST_DATA_BREAD_CLASSIFIER, transform=test_transform)
manager = LabelManager(dataloader=TRAIN_SET)

# Mainly used 
def train_model():
    classifier = BreadClassifier(input_shape=3, hidden_units=15, output_shape=len(TRAIN_SET.classes))
    optimizer = optim.SGD(params=classifier.parameters(), lr=0.001, momentum=0, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()   
    pipeline = TrainingPipeline(
        epochs=4, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        label_manager=manager,
        bread_model=classifier, 
        train_data=TRAIN_SET, 
        test_data=TEST_SET)
    return pipeline.execute()