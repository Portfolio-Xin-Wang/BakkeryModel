# from meiosis import ImageFrame, PILEntity
from torch import Tensor
from torch.utils.data import Dataset

from ..domain.map_bread import DATA_LABEL


class BreadData(Dataset):
    """
    This is a PyTorch DataSet class.
    """
    # images: list[PILEntity]

    # def __init__(self, image_frame: ImageFrame, transform=None):
    #     self.images = image_frame.images_collection
    #     self.transform = transform
    #     super().__init__()

    # def __getitem__(self, index:int) -> Tuple(Tensor, int):
    #     # Get first image entity.
    #     image = self.images[index]
    #     # Replaced with something more maintainable
    #     label_entity = image.meta_data._applied_transformation.get("label_name", "other_code")
    #     # Map label of image to prediction_idx: Frikandelbroodje -> 1 for example.
    #     LABEL_ENTITY = DATA_LABEL.get(label_entity)
    #     class_idx = LABEL_ENTITY.get("label_id")
    #     # Perform transformation if applicable
    #     if self.transform is not None:
    #         return self.transform(image.image), class_idx
    #     return image.image, class_idx
    
    # def __len__(self):
    #     return len(self.images)