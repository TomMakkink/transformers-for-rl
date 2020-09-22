import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MultiHeadAttention,
    MultiheadLinearAttention,
    RelativeMultiHeadAttention,
    PositionWiseMLP,
)
from configs.transformer_config import transformer_config
from configs.experiment_config import experiment_config


Tensor = torch.Tensor


class TransformerBlockBase(nn.Module):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(TransformerBlockBase, self).__init__()
        pass

    def forward(self, x):
        pass


class TransformerBlock(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(TransformerBlock, self).__init__(d_model, dim_mlp, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, transformer_config["num_heads"], dropout
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(self, inputs):
        # Attention
        x = inputs
        y = self.attention(inputs, inputs, inputs, attn_mask=None)[0]
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output


class ReZeroBlock(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(ReZeroBlock, self).__init__(d_model, dim_mlp, dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, transformer_config["num_heads"], dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention layer
        src2 = src
        src2 = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src2 = src2[0]  # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src


class LinformerBlock(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(LinformerBlock, self).__init__(d_model, dim_mlp, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiheadLinearAttention(
            d_model,
            transformer_config["num_heads"],
            dropout,
            transformer_config["max_seq_len"],
            transformer_config["k"],
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


class TransformerXLBlock(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(TransformerXLBlock, self).__init__(d_model, dim_mlp, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(
            transformer_config["num_heads"],
            d_model,
            dropout,
            mem_len=transformer_config["mem_len"],
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
        y = self.attention(inputs, r, u, v, attn_mask, mem)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
        return output

    def reset(self):
        pass


class GTrXLBlock(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(GTrXLBlock, self).__init__(d_model, dim_mlp, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.attention = RelativeMultiHeadAttention(
            transformer_config["num_heads"],
            d_model,
            dropout,
            mem_len=transformer_config["mem_len"],
        )

        self.hidden_1 = torch.zeros(1, 1, d_model).to(experiment_config["device"])
        self.hidden_2 = torch.zeros(1, 1, d_model).to(experiment_config["device"])
        self.gated_layer_1 = nn.GRU(d_model, d_model, num_layers=1)
        self.gated_layer_2 = nn.GRU(d_model, d_model, num_layers=1)

        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.d_model = d_model

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
        y = self.attention(inputs, r, u, v, attn_mask, mem)
        y = self.layer_norm_1(y)
        y, self.hidden_1 = self.gated_layer_1(x + y, self.hidden_2)
        # y = self.gated_layer_1([x, y])

        # Position-wise MLP
        x = y
        y = self.layer_norm_2(y)
        y = self.pos_wise_mlp(y)
        output, self.hidden_2 = self.gated_layer_2(x + y, self.hidden_2)
        return output

    def reset(self):
        self.hidden_1 = torch.zeros(1, 1, self.d_model).to(experiment_config["device"])
        self.hidden_2 = torch.zeros(1, 1, self.d_model).to(experiment_config["device"])


class MHA(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(MHA, self).__init__(d_model, dim_mlp, dropout)
        self.attention = nn.MultiheadAttention(
            d_model, transformer_config["num_heads"], dropout
        )

    def forward(self, inputs):
        y = self.attention(inputs, inputs, inputs, attn_mask=None)[0]
        return y


class LMHA(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(LMHA, self).__init__(d_model, dim_mlp, dropout)
        self.attention = MultiheadLinearAttention(
            d_model,
            transformer_config["num_heads"],
            dropout,
            transformer_config["max_seq_len"],
            transformer_config["k"],
        )

    def forward(self, inputs):
        return self.attention(inputs)


class RMHA(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(RMHA, self).__init__(d_model, dim_mlp, dropout)
        self.attention = RelativeMultiHeadAttention(
            transformer_config["num_heads"],
            d_model,
            dropout,
            mem_len=transformer_config["mem_len"],
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


class GMHA(TransformerBlockBase):
    def __init__(self, d_model: int, dim_mlp: int, dropout: int):
        super(GMHA, self).__init__(d_model, dim_mlp, dropout)
        self.attention = RelativeMultiHeadAttention(
            transformer_config["num_heads"],
            d_model,
            dropout,
            mem_len=transformer_config["mem_len"],
        )
        self.gru_hidden = torch.zeros(1, 1, d_model).to(experiment_config["device"])
        self.gated_layer = nn.GRU(d_model, d_model, num_layers=1)
        self.d_model = d_model

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
    ):
        y = self.attention(inputs, r, u, v, attn_mask, mem)
        y, self.gru_hidden = self.gated_layer(y, self.gru_hidden)
        return y

    def reset(self):
        self.gru_hidden = torch.zeros(1, 1, self.d_model).to(
            experiment_config["device"]
        )


def get_transformer_submodule(transformer: str):
    return {
        "vanilla": TransformerBlock,
        "rezero": ReZeroBlock,
        "linformer": LinformerBlock,
        "xl": TransformerXLBlock,
        "gtrxl": GTrXLBlock,
        "mha": MHA,
        "lmha": LMHA,
        "rmha": RMHA,
        "gmha": GMHA,
    }.get(transformer, TransformerBlock)
