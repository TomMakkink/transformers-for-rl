import torch
import torch.nn as nn
import numpy as np
from transformers.transformer_gtr_xl import GTrXL
from transformers.transformer import TransformerModel
from transformers.transformer_xl import TransformerXL
from transformers.re_zero import ReZero
from transformers.linformer import Linformer
from configs.ppo_config import ppo_config

Tensor = torch.Tensor


class Transformer(nn.Module):
    """
    Wrapper for the various Transformer classes. 
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        transformer_type: str = "None",
        num_layers: int = 2,
        num_heads: int = 1,
        dim_mlp: int = 32,
        dropout: float = 0.0,
        mem_len: int = 0,
        max_seq_len: int = 1,
    ):
        """
        Args: 
            d_model: number of expected features in the input. 
            output_dim: output dimension of the model. 
            transformer_type: type of transformer used: ["vanilla", "xl", "gtrxl", "rezero", "linformer", "None"]. Default: "None". 
            num_layers: number of layers in the transformer model. Default: 2.
            num_heads: number of attention heads. Default: 1. 
            dim_mlp: inner dimension of the positionwise multi-layer perceptron. Default: 32. 
            dropout: dropout. Default: 0.0. 
            mem_len: length of memory. Only relevant to "xl" and "gtrxl". Default: 0. 
        """
        super(Transformer, self).__init__()
        self.transformer_type = transformer_type.lower()
        if self.transformer_type == "gtrxl":
            print("Using GTrXL Transformer...")
            self.mem = tuple()
            self.transformer = GTrXL(
                d_model=d_model,
                output_dim=output_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_mlp=dim_mlp,
                dropout=dropout,
                mem_len=mem_len,
            )
        elif self.transformer_type == "xl":
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
        elif self.transformer_type == "rezero":
            print("Using ReZero...")
            self.transformer = ReZero(
                d_model=d_model,
                output_dim=output_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_mlp=dim_mlp,
                dropout=dropout,
            )
        elif self.transformer_type == "vanilla":
            print("Using Transformer...")
            self.transformer = TransformerModel(
                d_model=d_model,
                output_dim=output_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_mlp=dim_mlp,
                dropout=dropout,
            )
        elif self.transformer_type == "linformer":
            print("Using Linformer...")
            self.transformer = Linformer(
                d_model=d_model,
                output_dim=output_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_mlp=dim_mlp,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        else:
            print("No Transformer Selected...")
            self.transformer = None

    def reset_mem(self):
        if self.transformer_type == "xl" or self.transformer_type == "gtrxl":
            self.mem = self.transformer.init_mem()

    def forward(self, inputs: Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [seq_len, batch_size, features]

        Returns: 
            Transformer output, of shape: [seq_len, batch_size, output_dim]
        """
        if self.transformer is None:
            return inputs
        if self.transformer_type == "xl" or self.transformer_type == "gtrxl":
            output, self.mem = self.transformer(inputs, self.mem)
        else:
            output = self.transformer(inputs)
        return output
