import torch 
import torch.nn as nn 

class GTrXL(nn.Module):
    def __init__(
        self, 
        n_layers:int, 
        n_heads:int, 
        d_model:int, 
        d_head:int, 
        d_inner:int,
        dropout:float=0.0,
        **kwargs,
    ):
        super(GTrXL, self).__init__(**kwargs):

    def forward(self, inputs, **kwargs):
        return NotImplementedError