# Relation with data extraction
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

from helper_functions import accuracy_fn

from ..domain import ModelResult
from ..model import BreadClassifier
from .meta_data_service import MetaDataService
from .label_manager import LabelManager


class TrainingPipeline():
    EPOCHS: int 
    bread_model: BreadClassifier
    loss_fn: Module
    optimizer: Optimizer

    def __init__(self, 
                 epochs: int, 
                 bread_model: BreadClassifier, 
                 label_manager: LabelManager,
                 loss_fn, 
                 optimizer, 
                 train_data: DatasetFolder=None, 
                 test_data: DatasetFolder=None):
        self.EPOCHS = epochs
        self.bread_model = bread_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.label_manager = label_manager
        self.meta_data = MetaDataService()
    
    def execute(self) -> ModelResult:
        self._log_pre_training()
        train, test = self._extract()
        self._train_model(train, test)
        self.label_manager._update_file(self.train_data.class_to_idx)
        # Returned trained model
        return ModelResult(log_entity=self.meta_data.get(), model=self.bread_model)
    
    def _log_pre_training(self):
        optimizer: dict = {}
        self.meta_data.add_pre_training_param(optimizer)

    def _extract(self) -> tuple[DataLoader, DataLoader]:
        # Future expansion: Use the recent data to train retrain the model.
        train_data = DataLoader(self.train_data, batch_size=3, shuffle=True, num_workers=0)
        test_data = DataLoader(self.test_data, batch_size=3, num_workers=1, shuffle=False)
        return train_data, test_data

    def _train_model(self, train: DataLoader, test: DataLoader):
        # Create training and testing loop
        for epoch in tqdm(range(self.EPOCHS)):
            train_loss = 0
            print(f"Epoch: {epoch}\n-------")
            for batch, (X, y) in enumerate(train):
                self.bread_model.train() 
                y_pred = self.bread_model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self._validate_model(train, test, train_loss)
            self.meta_data.up_epoch()
        return self.bread_model

    def _validate_model(self, train_data, test_data: DataLoader, train_loss: float):
        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_data)
        self.bread_model.eval()
        test_loss, test_acc = self._test_loss_results(test_data)
        self.meta_data.add_metric("train_loss", train_loss)
        self.meta_data.add_metric("test_loss", test_loss)
        self.meta_data.add_metric("test_accuracy", test_acc)
        return test_loss, test_acc
    
    # Might need to be moved to a separate class for inference
    def _test_loss_results(self, test_data: DataLoader):
        test_loss, test_acc = 0, 0 
        with torch.inference_mode():
            for X, y in test_data:
                test_pred = self.bread_model(X)
                test_loss += self.loss_fn(test_pred, y) 
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(test_data)
            test_acc /= len(test_data)
        return test_loss, test_acc

    