from torch.utils.data import Dataset
from torch import Tensor
import torch
import os
import numpy as np
import io 
from pandas import DataFrame
from PIL import Image

class BreadData(Dataset):
    labels: DataFrame
    original_directory: str

    def __init__(self, labels: DataFrame, original_directory: str, transform=None):
        self.labels = labels
        self.transform = transform
        self.original_directory = original_directory

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        # Replace with actual image_name column.
        img_name = os.path.join(self.original_directory, self.labels.iloc[idx, 1])
        # Should be fixed with working image reader like PIL or OpenCV
        image = Image.open(img_name)
        labels = self.labels.iloc[idx, 0]
        labels = np.array([labels], dtype=float).reshape(-1, 1)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample