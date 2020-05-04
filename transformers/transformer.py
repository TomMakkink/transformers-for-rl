import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers.attention_layer import MultiHeadAttention, TransformerBlock
from transformers.positional_encoding_layer import PositionalEncoding

Tensor = torch.Tensor

class TransformerModel(nn.Module):
    """
    Transformer Architecture from the "Attention is All you Need" paper: https://arxiv.org/abs/1706.03762. 

    The architecture consists of stacked self-attention and point-wise, fully connected layers. 
    This transformer architecture achieved excellent results on NLP tasks such as machine translation and 
    language modelling. In addition, the Transformer architecture is more parallelisable than Recurrent Neural Networks. 
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
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        self.Transformers = [
            TransformerBlock(
                num_heads=num_heads,
                d_model=d_model,
                dim_mlp=dim_mlp,
                dropout=dropout, 
            )
            for k in range(num_layers)
        ]

        self.out_layer = nn.Linear(d_model, output_dim, bias=False)
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
            Transformer output, of shape: [output_dim]
        """
        x = self.pos_encoder(inputs * math.sqrt(self.d_model))
        for layer in self.Transformers:
            x = layer(x)
        return self.out_layer(x)[0]

    
