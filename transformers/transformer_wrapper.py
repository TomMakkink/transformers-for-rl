import torch 
import torch.nn as nn
from transformers.transformer_gtr_xl import GTrXL
from transformers.transformer import TransformerModel
from transformers.transformer_xl import TransformerXL

class Transformer(nn.Module):
    """
    Wrapper for the various Transformer classes. 
    """
    def __init__(
        self,
        dim_model:int,
        dim_head:int,
        transformer_type:str="None",
        num_layers:int=2,
        num_heads:int=1, 
        dim_mlp:int=32, 
        dropout:float=0.1,  
    ):
        super(Transformer, self).__init__()
        if transformer_type.lower() == "gtrxl":
            print("Using GTrXL Transformer...")
            self.transformer = GTrXL(
                num_layers=num_layers, 
                dim_model=dim_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dim_mlp=dim_mlp,
                dropout=dropout,
            )
        # elif transformer_type.lower() == "xl":
        #     self.transformer =
        else: self.transformer = None

    def forward(self, inputs):
        if self.transformer is None: return inputs
        # print(f"wrapper outputs: {x}")
        return self.transformer(inputs)