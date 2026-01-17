import io

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.domain import Prediction
from src.model import BreadClassifier
from .label_manager import LabelManager

class InferencePipeline:
    def __init__(self, model: BreadClassifier, mapper: LabelManager):
        self.model = model
        self.test_transform = transforms.ToTensor()
        self.mapper = mapper

    def _bytes_to_tensor(self, img_bytes: bytes) -> Tensor:
        pil_image = Image.open(io.BytesIO(img_bytes))
        tensor_image = self.test_transform(pil_image)
        ready_input = torch.unsqueeze(tensor_image, 0)
        return ready_input

    def live_predict(self, input_data: bytes) -> dict:
        input_image = self._bytes_to_tensor(input_data)
        confidence, index = self.model.single_predict(input_image)
        prediction_result = Prediction(
            prediction_nr=index.item(),
            confidence_percentage=confidence.item() * 100,
            prediction_label=self.mapper.get_label(index.item())
        )
        return prediction_result