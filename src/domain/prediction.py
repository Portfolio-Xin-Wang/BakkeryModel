from pydantic import BaseModel


class Prediction(BaseModel):
    prediction_nr: int 
    confidence_percentage: float 
    prediction_label: str
    
    def get_label(self, mapper):
        return mapper[self.prediction_nr]
