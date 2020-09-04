import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers.attention_layer import MultiHeadAttention, PositionWiseMLP
from transformers.positional_encoding_layer import PositionalEncoding

Tensor = torch.Tensor


class TransformerBlock(nn.Module):
    def __init__(
        self, num_heads: int, d_model: int, dim_mlp: int, dropout: int,
    ):
        super(TransformerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(self, inputs):
        max_len = inputs.shape[0]
        # mask = _generate_square_subsequent_mask(max_len)
        mask = None

        # Attention
        x = inputs
        y = self.attention(inputs, inputs, inputs, attn_mask=mask)[0]
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output


class TransformerModel(nn.Module):
    """
    Transformer Architecture from the "Attention is All you Need" paper: https://arxiv.org/abs/1706.03762. 

    The architecture consists of stacked self-attention and point-wise, fully connected layers. 
    This transformer architecture achieved excellent results on NLP tasks such as machine translation and 
    language modelling. In addition, the Transformer architecture is more parallelisable than Recurrent Neural Networks. 
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 2,
        dim_mlp: int = 2048,
        dropout: float = 0.0,
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

        self.Transformers = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    dim_mlp=dim_mlp,
                    dropout=dropout,
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
        for layer in self.Transformers:
            x = layer(x)
        return self.out_layer(x)  # [0]

