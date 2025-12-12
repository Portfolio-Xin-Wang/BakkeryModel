from meiosis import ImageFrame, PILEntity
from torch import Tensor
from torch.utils.data import Dataset

from .map_bread import data


class BreadData(Dataset):
    """
    This is a PyTorch DataSet class.
    """
    labels: list[str]
    meta_data: list[str]
    images: list[PILEntity]

    def __init__(self, image_frame: ImageFrame, meta_data: list[str]):
        self.images = image_frame.images_collection
        self.labels = [e.meta_data.name for e in self.images]
        self.meta_data = meta_data
        super().__init__()

    def __getitem__(self, index:int) -> Tuple(Tensor, int):
        # Get first image entity.
        image = self.images[index]
        img = image.image_to_numpy()
        label = self.meta_data[index]
        # Map label of image to prediction_idx: Frikandelbroodje -> 1 bijvoorbeeld.
        class_idx = 1
        # Perform transformation if applicable

        # Return tuple.
        return (img, class_idx)
    
    def __len__(self):
        return len(self.images)