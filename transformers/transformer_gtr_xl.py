import torch 
import torch.nn as nn 

from transformers.positional_encoding_layer import RelativePositionalEncoding
from transformers.attention_layer import GTrXLBlock

class GTrXL(nn.Module):
    def __init__(
        self, 
        num_layers:int, 
        num_heads:int, 
        dim_model:int, 
        dim_head:int, 
        dim_mlp:int,    
        dropout:float=0.0,
    ):
        super(GTrXL, self).__init__()
        self.embedding = nn.Linear(dim_model, dim_model)
        self.positional_encoding_layer = RelativePositionalEncoding(dim_model)
        self.dropout = nn.Dropout(dropout)

        self.GTrXLs = [
            GTrXLBlock(
                num_heads=num_heads,
                dim_model=dim_model,
                dim_mlp=dim_mlp,
                use_scale=True,
                dropout=dropout, 
            )
            for k in range(num_layers)
        ]

        self.output_layer = nn.Sequential(
            nn.Linear(dim_model, dim_mlp, bias=False), nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(dim_mlp, dim_model, bias=False),
        )


    def forward(self, inputs):
        # Leave out embedding & positional encoding for now 
        # Leave dropout as well 
        # Word embedding format 
        # Positional encoding returns []
        x = inputs.view(inputs.size(0), 1, inputs.size(1))
        for layer in self.GTrXLs:
            x = layer(x)
        return self.output_layer(x)