import torch 
import torch.nn as nn 
from torch.nn.init import kaiming_uniform_

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=1)
        self.maxp1 = nn.MaxPool2d(3, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_block1 = nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1),
            nn.ReLU(), 
            nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1), 
        )

        self.residual_block2 = nn.Sequential(
            nn.ReLU(), 
            nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1),
            nn.ReLU(), 
            nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1),  
        )

    def forward(self, obs):
        src = self.maxp1(self.conv1(obs))
        src2 = src 
        src2 = self.residual_block1(src2)
        src = src + self.dropout1(src2)
        src2 = src 
        src2 = self.residual_block2(src2)
        src = src + self.dropout2(src2)
        return src 


class ResNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i, (input_channels, output_channels) in enumerate([(3, 16), (16, 32), (32, 32)]): 
            self.conv_layers.append(ResNetBlock(input_channels, output_channels, dropout))
        self.output_activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                kaiming_uniform_(p, nonlinearity='relu')
        
    def forward(self, obs):
        for layer in self.conv_layers:
            obs = layer(obs)
        return self.output_activation(obs)