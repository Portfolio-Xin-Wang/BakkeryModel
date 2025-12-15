import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class ModelPipeline(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)