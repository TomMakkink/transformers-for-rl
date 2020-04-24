import torch 
import torch.nn as nn
from transformers.transformer_gtr_xl import GTrXL
from transformers.transformer import TransformerModel
from transformers.transformer_xl import TransformerXL
from transformers.re_zero import ReZero

Tensor = torch.Tensor

class Transformer(nn.Module):
    """
    Wrapper for the various Transformer classes. 
    """
    def __init__(
        self,
        d_model:int,
        output_dim:int,
        transformer_type:str="None",
        num_layers:int=2,
        num_heads:int=1, 
        dim_mlp:int=32, 
        dropout:float=0.0,  
        mem_len:int=0,
    ):
        """
        Args: 
            d_model: number of expected features in the input. 
            output_dim: output dimension of the model. 
            transformer_type: type of transformer used: ["vanilla", "xl", "gtrxl", "None"]. Default: "None". 
            num_layers: number of layers in the transformer model. Default: 2.
            num_heads: number of attention heads. Default: 1. 
            dim_mlp: inner dimension of the positionwise multi-layer perceptron. Default: 32. 
            dropout: dropout. Default: 0.0. 
            mem_len: length of memory. Only relevant to "xl" and "gtrxl". Default: 0. 
        """
        super(Transformer, self).__init__()
        self.transformer_type = transformer_type
        if transformer_type.lower() == "gtrxl":
            print("Using GTrXL Transformer...")
            self.transformer = GTrXL( 
                d_model=d_model,
                output_dim=output_dim, 
                num_layers=num_layers,
                num_heads=num_heads,
                dim_mlp=dim_mlp,
                dropout=dropout,
            )
        elif transformer_type.lower() == "xl":
            print("Using Transformer-XL...")
            self.mem = tuple()
            self.transformer = TransformerXL(
                d_model=d_model,
                output_dim=output_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_mlp=dim_mlp,
                dropout=dropout,
                mem_len=mem_len, 
            )
        elif transformer_type.lower() == "rezero":
            print("Using ReZero...")
            self.transformer = ReZero(
                d_model=d_model, 
                output_dim=output_dim,
                num_heads=num_heads, 
                num_layers=num_layers,
                dim_mlp=dim_mlp, 
                dropout=dropout,
            )
        elif transformer_type.lower() == "vanilla":
            print("Using Transformer...")
            self.transformer = TransformerModel(
                d_model=d_model, 
                output_dim=output_dim,
                num_heads=num_heads, 
                num_layers=num_layers,
                dim_mlp=dim_mlp, 
                dropout=dropout,
            )
        else: 
            print("Vanilla Policy Gradient...")
            self.transformer = None

    def forward(self, inputs:Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features]

        Returns: 
            Transformer output, of shape: [source_seq_len, batch_size, output_dim]
        """
        if self.transformer is None: return inputs
        if self.transformer_type == "xl":
            ret = self.transformer(inputs, *self.mem)
            output, self.mem = ret[0], ret[1:]
        else:
            output = self.transformer(inputs)[0]      # Needed because of the positional encoding 
        return output