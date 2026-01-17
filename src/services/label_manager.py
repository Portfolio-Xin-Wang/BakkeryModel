from torchvision.datasets import DatasetFolder
import json
from abc import ABC, abstractmethod

class ILabelManager(ABC):
    
    @abstractmethod
    def _pre_load(self):
        pass

    @abstractmethod
    def _update_file(self, mapping):
        pass

    @abstractmethod
    def get_label(self, label_id: int) -> str:
        pass


class LabelManager(ILabelManager):
    """
    Handles the retrieval of labels.

    Alternatively this could be stored within the model itself, 
    but changes to the labels like spelling mistakes requires the training of the entire model
    """
    map_cache: dict 

    def __init__(self, location="/ready_model", dataloader: DatasetFolder=None):
        self.location = location 
        self.dataloader = dataloader
        self.map_cache = {}
        # Verbeteringen vereist.
        if (dataloader is not None):
            self._pre_load()
    
    def _pre_load(self):
        mappings = { id: label for (label, id) in self.dataloader.class_to_idx.items() }
        self._update_file(mappings)

    def _update_file(self, mapping):
        """
        Store or change the file containing the label information.
        Mostly stored in a {location}/model_labels.json file.
        """
        with open(f"{self.location}/model_label.json", "w") as f:
            maps = json.dumps(mapping, indent=4)
            f.write(maps)

    def get_label(self, label_id: int) -> str:
        """
        Returns a label based on the predicted label_id provided.
        If no mapping is present, return exception.
        If label id is not present in id, return exception.
        Otherwise return label name
        """
        if self.map_cache == {}:
            with open(f"{self.location}/model_label.json", "r") as f:
                 self.map_cache = json.load(f)
        return self.map_cache.get(label_id)
    
    def update(self, map: dict[str, int]):
        mappings = { id: label for (label, id) in map.items() }
        self._update_file(mappings)