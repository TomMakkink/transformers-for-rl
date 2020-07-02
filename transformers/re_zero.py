import math
import torch 
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import RZTXEncoderLayer

Tensor = torch.Tensor 

class ReZero(nn.Module):
    """
    ReZero transformer architecture based on the "ReZero is All You Need" paper: https://arxiv.org/pdf/2003.04887.pdf. 

    The architecture consists of stacked self-attention and point-wise, fully connected layers with 
    residual weights for faster convergece. The implementation is adapted from the official implementation 
    which can be found here: https://github.com/majumderb/rezero. 
    """
    def __init__(
        self, 
        d_model:int,
        output_dim:int,
        num_layers:int,  
        num_heads:int=2, 
        dim_mlp:int=2048, 
        dropout:float=0.0,
    ):
        """
        Args: 
            d_model: number of expected features in the input. 
            output_dim: output dimension of the model. 
            num_layers: number of submodules in the transformer. 
            num_heads: number of attention heads. Default: 2. 
            dim_mlp: inner dimension of the multilayer perceptron. Default: 2048. 
            dropout: dropout. Default: 0.0. 
        """
        super(ReZero, self).__init__() 
        self.d_model = d_model
        # self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        encoder_layer = RZTXEncoderLayer(d_model, num_heads, dim_mlp, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(), 
        )
        
        self._reset_parameters()


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, inputs:Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features]

        Returns: 
            Transformer output, of shape: [source_seq_len, batch_size, output_dim]
        """
        # x = self.pos_encoder(inputs * math.sqrt(self.d_model))
        x = self.transformer_encoder(inputs)
        return self.out_layer(x)