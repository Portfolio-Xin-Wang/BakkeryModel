from torch.utils.data import Dataset
from torch import Tensor
import torch
import os
import numpy as np
import io 
from pandas import DataFrame
from skimage import io, transform

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
        print(img_name)
        image = io.imread(img_name)
        landmarks = self.labels.loc[idx, 2:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample