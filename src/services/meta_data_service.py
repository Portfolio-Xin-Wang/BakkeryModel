from ..domain import LogEntity

class MetaDataService:
    epoch: int = 0

    def __init__(self):
        self.logger = LogEntity()
        self.epoch = 0

    def reset(self):
        self.logger = LogEntity()
        return self
    
    def add_pre_training_param(self, params: dict):
        self.logger.model_training_param = params

    def log_accuracy(self, epoch, accuracy):
        self.logger.add_epoch_result(epoch, accuracy)
        return self
    
    def up_epoch(self):
        self.epoch += 1
    
    def add_metric(self, metric, value):
        result = {"epoch": self.epoch, "metric": metric, "value": value}
        print(f"\nepoch: {self.epoch}: {metric}: {value:.5f} %")
        self.logger.epoch_result.append(result)

    def get(self):
        return self.logger