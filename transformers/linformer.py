import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers.attention_layer import MultiheadLinearAttention, PositionWiseMLP
from transformers.positional_encoding_layer import PositionalEncoding

Tensor = torch.Tensor


class LinformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dim_mlp: int,
        dropout: int,
        max_seq_len: int,
    ):
        super(LinformerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiheadLinearAttention(
            d_model, num_heads, dropout, max_seq_len, k=4,
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout=dropout)

    def forward(self, inputs):
        # Attention
        x = inputs
        y = self.attention(inputs)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output


class Linformer(nn.Module):
    """
    Linformer Architecture from the "Linformer: Self-Attention with Linear Complexity" paper: https://arxiv.org/pdf/2006.04768.pdf. 
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        dim_mlp: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 1,
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
        super(Linformer, self).__init__()
        self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.LinformerLayers = nn.ModuleList(
            [
                LinformerBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    dim_mlp=dim_mlp,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for k in range(num_layers)
            ]
        )

        self.out_layer = nn.Linear(d_model, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inputs: Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [seq_len, batch_size, features]

        Returns: 
            Transformer output, of shape: [seq_len, batch_size, output_dim]
        """
        x = self.pos_encoder(inputs * math.sqrt(self.d_model))

        for layer in self.LinformerLayers:
            x = layer(x)
        return self.out_layer(x)  # [0]
