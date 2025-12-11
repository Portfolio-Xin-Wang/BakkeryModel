from torch.utils.data import Dataset
from torch import Tensor
import torch
import os
import numpy as np
import io 
from pandas import DataFrame
from PIL import Image

class BreadData(Dataset):
    labels: list 
    img: list 

    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Tuple(Tensor, int):
        # Get first image entity.

        # Map label of image to prediction_idx: Frikandelbroodje -> 1 bijvoorbeeld.

        # Perform transformation if applicable

        # Return tuple.
        return (0,0)
    
    def __len__(self):
        return len(self.img)