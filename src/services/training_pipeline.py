# Relation with data extraction
import torch
from torch.nn import Module
from torch.optim import Optimizer
from meiosis import LocalFileStorage
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from helper_functions import accuracy_fn

from ..model import BreadClassifier


class TrainingPipeline():
    EPOCHS: int 
    bread_model: BreadClassifier
    loss_fn: Module
    optimizer: Optimizer

    def __init__(self, epochs: int, bread_model: BreadClassifier, loss_fn, optimizer, train_data: Dataset=None, test_data: Dataset=None):
        self.EPOCHS = epochs
        self.bread_model = bread_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
    
    def execute(self, last_data: int = None):
        train, test = self._extract()
        self._train_model(train, test)
        # Returned trained model
        return self.bread_model

    def _extract(self) -> tuple[DataLoader, DataLoader]:
        # Future expansion: Use the recent data to train retrain the model.
        train_data = DataLoader(self.train_data, batch_size=3, shuffle=True, num_workers=0)
        test_data = DataLoader(self.test_data, batch_size=3, num_workers=1, shuffle=False)
        return train_data, test_data

    def _train_model(self, train: DataLoader, test: DataLoader):
        results = []
        # Create training and testing loop
        for epoch in tqdm(range(self.EPOCHS)):
            train_loss = 0
            for batch, (X, y) in enumerate(train):
                self.bread_model.train() 
                y_pred = self.bread_model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch % 400 == 0:
                    print(f"Looked at {batch * len(X)}/{len(self.train_data.dataset)} samples")
            loss_results = self._validate_model(train, test, train_loss)
            results.append(loss_results)
        return results

    def _validate_model(self, train_data, test_data: DataLoader, train_loss: float):
        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_data)
        test_loss, test_acc = 0, 0 
        self.bread_model.eval()
        test_loss, test_acc = self._test_loss_results(test_data)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
        return test_loss, test_acc
    
    # Might need to be moved to a separate class for inference
    def _test_loss_results(self, test_data: DataLoader):
        with torch.inference_mode():
            for X, y in test_data:
                test_pred = self.bread_model(X)
                test_loss += self.loss_fn(test_pred, y) 
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(test_data)
            test_acc /= len(test_data)
        return test_loss, test_acc

    