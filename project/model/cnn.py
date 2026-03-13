import torch
import torch.nn as nn

# il faut surement changer les entrees car elles sont pas bon pour notre dataset
# The CNN module takes as inputs the number of output channels in each layer (and the number of classes), 
# but we also have to explicitely provide the input size of the Linear layer, since it depends on many other parameters 
# (image size, but also kernel, padding, stride, number of channels in the convolutions...). 
# Instead of computing it with a general (and heavy) formula, it's easier to just get it from an example (as above), and then pass it as an input parameter.

class CNNClassif(nn.Module):
    
    def __init__(self, input_size_linear, nb_hidden_layers, num_channels1=16, num_channels2=32, num_classes=10):
        
        super(CNNClassif, self).__init__()
        self.nb_hidden_layers
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(3, num_channels1, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        out_nb_channels = num_channels1 
        self.hidden_layers = []
        
        for i in range(1, self.nb_hidden_layers):
            layers = nn.Sequential(nn.Conv2d(out_nb_channels, out_nb_channels * 2, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
            self.hidden_layers.append(layers)
            out_nb_channels = out_nb_channels * 2
        
        self.linear_layer = nn.Linear(input_size_linear, num_classes)

    def forward(self, x):
        y = self.cnn_layer1(x)
        for i in range(1, self.nb_hidden_layers):
            y = self.hidden_layers[i](y)    
        y = y.reshape(y.shape[0], -1)
        out = self.linear_layer(y)
        return out
