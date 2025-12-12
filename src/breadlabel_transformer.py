from meiosis import MapTransformer
from .const import LABEL
from .map_bread import data, natural_labels

class BreadLabeler(MapTransformer):

    def transform(self, image_store):
        image_store = super().transform(image_store)
        for entity in image_store.get_all():
            label = self._return_correct_label(entity.return_image_name())
            entity.meta_data.add_transformation(LABEL, label)
            entity.meta_data.location = label
        return image_store
    
    def _return_correct_label(self, name: str) -> str:
        """
        Assigns label based on the image name and the present labels inside natural labels.
        '3;copy&228;rot&frikandel_brood_2.jpg' should return frikandel_brood
        """
        for label_name, label_code in natural_labels:
            if label_name in name:
                return label_code
        return "other_code"
            
