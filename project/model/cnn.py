import torch
import torch.nn as nn

class CNNClassif(nn.Module):
    def __init__(self, input_size_linear, num_channels1=16, num_channels2=32, num_classes=10):
        super(CNNClassif, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(1, num_channels1, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.linear_layer = nn.Linear(input_size_linear, num_classes)

    def forward(self, x):
        y = self.cnn_layer1(x)
        y = self.cnn_layer2(y)
        y = y.reshape(y.shape[0], -1)
        out = self.linear_layer(y)

        return out
