from torch import nn, Tensor

class BreadClassifier(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=input_shape, 
                      out_features=output_shape)
        )
    
    def forward(self, x: Tensor):
        x = self.classifier(x)
        return x