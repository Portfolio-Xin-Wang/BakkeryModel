from torch import nn

class BreadClassifier(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def foward(self, image):
        pass