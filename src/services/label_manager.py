from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset

class LabelManager:
    map_cache: dict 

    def __init__(self, location="/label_generic", dataloader: VisionDataset=None):
        self.location = location 
        self.dataloader = dataloader
        self.map_cache = {}
        self._pre_load()
    
    def _pre_load(self):
        # Performs all of the pre-tasks before completing initialisation of this class
        pass 

    def _update_file(self, mapping):
        """
        Store or change the file containing the label information.
        Mostly stored in a {location}.parquet file.
        """
        pass

    def get_label(self, label_id: int) -> str:
        """
        Returns a label based on the predicted label_id provided.
        If no mapping is present, return exception.
        If label id is not present in id, return exception.
        Otherwise return label name
        """
        pass


    def update(self, mapping: dict = None) -> bool:
        # Update cache
        # Call update file to change
        pass 