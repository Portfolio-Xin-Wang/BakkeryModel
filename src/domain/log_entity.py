from pydantic import BaseModel


class LogEntity(BaseModel):
    model_training_param: dict = {}
    epoch_result: list = []

    def add_epoch_result(self, epoch, acc):
        self.epoch_result.append({"epoch": epoch, "accuracy": acc})

    def get_results(self):
        return self.__dict__