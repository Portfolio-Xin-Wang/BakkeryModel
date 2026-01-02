from torch import nn, Tensor

class BreadClassifier(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.adaptive = nn.AdaptiveAvgPool2d(output_size=(300, 300))
        # Convolutional Layer
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # Activation function
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Output layer
        self.output = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=84375, # Gebasseerd op 200x200 input images
                      out_features=output_shape)
        )

    def _forward(self, x: Tensor):
        x = self.adaptive(x)
        x = self.block_1(x)
        # print(f"Shape after block 1: {x.shape}")
        x = self.block_2(x)
        # print(f"Shape after block 2: {x.shape}")
        x = self.output(x)
        return x
    
    def forward(self, x: Tensor):
        return self._forward(x)