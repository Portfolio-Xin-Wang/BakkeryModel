
from ..model import BreadClassifier
from .log_entity import LogEntity

class ModelResult:
    training_results: LogEntity
    model: BreadClassifier

    def __init__(self, log_entity, model: BreadClassifier):
        self.training_results = log_entity
        self.model = model

    def get_training_results(self):
        return self.training_results.get_results()
