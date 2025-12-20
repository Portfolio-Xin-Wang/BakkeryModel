# Relation with data extraction

class TrainingPipeline():

    def __init__(self):
        pass

    def _extract(self):
        pass

    def _train_model(self):
        pass

    def _evaluate(self):
        pass

    def execute(self):
        print("Executing training pipeline")
        self._extract()
        self._train_model()
        self._evaluate()
        print("Training pipeline execution completed")