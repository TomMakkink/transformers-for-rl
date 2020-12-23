import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MultiheadLinearAttention,
    RelativeMultiHeadAttention,
    PositionWiseMLP,
)

Tensor = torch.Tensor


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_mlp: int, dropout: int):
        super(TransformerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(self, inputs):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, inputs, inputs, attn_mask=None)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output, attn_output_weights.detach()


class ReZeroBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_mlp: int, dropout: int):
        super(ReZeroBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, inputs):
        # Self attention layer
        x = inputs
        y, attn_output_weights = self.attention(inputs, inputs, inputs, attn_mask=None)
        y = y * self.resweight
        y = x + self.dropout1(y)

        # Pointiwse FF Layer
        x = y
        y = self.pos_wise_mlp(y)
        y = y * self.resweight
        y = x + self.dropout2(y)
        return y, attn_output_weights.detach()


class LinformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_mlp: int,
        max_seq_len: int,
        k: int,
        dropout: int,
    ):
        super(LinformerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiheadLinearAttention(
            d_model,
            num_heads,
            dropout,
            max_seq_len,
            k,
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout=dropout)

    def forward(self, inputs):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output, attn_output_weights.detach()


class TransformerXLBlock(nn.Module):
    def __init__(
        self, d_model: int, dim_mlp: int, num_heads: int, mem_len: int, dropout: int
    ):
        super(TransformerXLBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
        hidden: Tensor = None,
    ):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, r, u, v, attn_mask, mem)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output, attn_output_weights.detach()

    def reset(self):
        pass


class GTrXLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_mlp: int,
        num_heads: int,
        mem_len: int,
        dropout: int,
        device,
    ):
        super(GTrXLBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )

        self.hidden_1 = torch.zeros(1, 1, d_model, device=device)
        self.hidden_2 = torch.zeros(1, 1, d_model, device=device)
        self.gated_layer_1 = nn.GRU(d_model, d_model, num_layers=1)
        self.gated_layer_2 = nn.GRU(d_model, d_model, num_layers=1)

        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.d_model = d_model
        self.device = device

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask=None,
        mem: Tensor = None,
    ):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, r, u, v, attn_mask, mem)
        y = self.layer_norm_1(y)
        y, self.hidden_1 = self.gated_layer_1(x + y, self.hidden_1)

        # Position-wise MLP
        x = y
        y = self.layer_norm_2(y)
        y = self.pos_wise_mlp(y)
        y, self.hidden_2 = self.gated_layer_2(x + y, self.hidden_2)
        return y, attn_output_weights.detach()

    def reset(self):
        self.hidden_1 = torch.zeros(1, 1, self.d_model, device=self.device)
        self.hidden_2 = torch.zeros(1, 1, self.d_model, device=self.device)


class UniversalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_mlp: int, dropout: int):
        super(UniversalTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(self, inputs: Tensor):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, inputs, inputs, attn_mask=None)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output, attn_output_weights.detach()


class ReZeroXLBlock(nn.Module):
    def __init__(
        self, d_model: int, dim_mlp: int, num_heads: int, mem_len: int, dropout: int
    ):
        super(ReZeroXLBlock, self).__init__()
        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
        hidden: Tensor = None,
    ):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, r, u, v, attn_mask, mem)
        y = y * self.resweight
        y = x + self.dropout1(y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        y = y * self.resweight
        y = x + self.dropout2(y)
        return y, attn_output_weights.detach()

    def reset(self):
        pass


class ReZeroGTrXLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_mlp: int,
        num_heads: int,
        mem_len: int,
        dropout: int,
        device,
    ):
        super(ReZeroGTrXLBlock, self).__init__()
        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )

        self.hidden_1 = torch.zeros(1, 1, d_model).to(device)
        self.hidden_2 = torch.zeros(1, 1, d_model).to(device)
        self.gated_layer_1 = nn.GRU(d_model, d_model, num_layers=1)
        self.gated_layer_2 = nn.GRU(d_model, d_model, num_layers=1)

        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.d_model = d_model
        self.device = device
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask=None,
        mem: Tensor = None,
    ):
        # Attention
        x = inputs
        y, attn_output_weights = self.attention(inputs, r, u, v, attn_mask, mem)
        y = y * self.resweight
        y, self.hidden_1 = self.gated_layer_1(x + y, self.hidden_2)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        y = y * self.resweight
        output, self.hidden_2 = self.gated_layer_2(x + y, self.hidden_2)
        return output, attn_output_weights.detach()

    def reset(self):
        self.hidden_1 = torch.zeros(1, 1, self.d_model).to(self.device)
        self.hidden_2 = torch.zeros(1, 1, self.d_model).to(self.device)


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: int):
        super(MHA, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)

    def forward(self, inputs):
        y, attn_output_weights = self.attention(inputs, inputs, inputs, attn_mask=None)
        return y, attn_output_weights.detach()


class LMHA(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, max_seq_len: int, k: int, dropout: int
    ):
        super(LMHA, self).__init__()
        self.attention = MultiheadLinearAttention(
            d_model,
            num_heads,
            dropout,
            max_seq_len,
            k,
        )

    def forward(self, inputs):
        return self.attention(inputs)


class RMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mem_len: int, dropout: int):
        super(RMHA, self).__init__()
        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
    ):
        return self.attention(inputs, r, u, v, attn_mask, mem)

    def reset(self):
        pass


class GMHA(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, mem_len: int, dropout: int, device
    ):
        super(GMHA, self).__init__()
        self.attention = RelativeMultiHeadAttention(
            num_heads,
            d_model,
            dropout,
            mem_len=mem_len,
        )
        self.gru_hidden = torch.zeros(1, 1, d_model).to(device)
        self.gated_layer = nn.GRU(d_model, d_model, num_layers=1)
        self.d_model = d_model
        self.device = device

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
    ):
        y, attn_output_weights = self.attention(inputs, r, u, v, attn_mask, mem)
        y, self.gru_hidden = self.gated_layer(y, self.gru_hidden)
        return y, attn_output_weights

    def reset(self):
        self.gru_hidden = torch.zeros(1, 1, self.d_model).to(self.device)
