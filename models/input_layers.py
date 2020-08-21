from collections import OrderedDict
from typing import List
import torch.nn as nn


class LinearInputLayer(nn.Module):
    def __init__(self, input_size: int, hidden_layers_size: List = [128]):
        super(LinearInputLayer, self).__init__()

        layers = OrderedDict()

        layers['input'] = nn.Linear(input_size, hidden_layers_size[0])
        layers['relu'] = nn.ReLU()

        for i in range(1, len(hidden_layers_size)):
            layers['linear' + str(i)] = nn.Linear(hidden_layers_size[i - 1],
                                                  hidden_layers_size[i])
            layers['relu' + str(i)] = nn.ReLU()

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)
