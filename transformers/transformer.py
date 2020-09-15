import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers.attention_layer import (
    MultiHeadAttention,
    MultiheadLinearAttention,
    RelativeMultiHeadAttention,
    PositionWiseMLP,
    GatedRecurrentUnit,
)
from transformers.positional_encoding_layer import PositionalEncoding
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config

Tensor = torch.Tensor


class TransformerModel(nn.Module):
    """
    Transformer baseclass. 
    """

    def __init__(self, d_model: int, output_dim: int, submodule):
        """
        Args: 
            d_model: number of expected features in the input. 
            output_dim: output dimension of the model. 
            dropout: dropout. Default: 0.0. 
        """
        super(TransformerModel, self).__init__()
        assert issubclass(
            submodule, TransformerBlockBase
        ), "Invalid Transformer submodule. "
        dropout = transformer_config["dropout"]
        self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.TransformerLayers = nn.ModuleList(
            [
                submodule(
                    d_model=d_model,
                    dim_mlp=transformer_config["dim_mlp"],
                    dropout=dropout,
                )
                for k in range(transformer_config["num_layers"])
            ]
        )

        self.out_layer = nn.Linear(d_model, output_dim)
        self._init_network()

    def _init_network(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x: Tensor):
        """
        Args:
            x: input tensor, of shape: [seq_len, batch_size, features]

        Returns:
            Transformer output, of shape: [seq_len, batch_size, output_dim]
        """
        x = self.pos_encoder(x * math.sqrt(self.d_model))
        for layer in self.TransformerLayers:
            x = layer(x)
        return self.out_layer(x)  # [0]


class MemoryTransformerModel(nn.Module):
    """
    Transformer base model that uses memory. 
    """

    def __init__(self, d_model: int, output_dim: int, submodule):
        """
        Args: 
            d_model: number of expected features in the input.  
            output_dim = output dimension of the model.  
        """
        super(MemoryTransformerModel, self).__init__()
        num_heads = transformer_config["num_heads"]
        dim_head = d_model // num_heads
        dropout = transformer_config["dropout"]
        self.drop = nn.Dropout(dropout)
        self.mem_len = transformer_config["mem_len"]
        self.num_layers = transformer_config["num_layers"]
        self.positional_encoding_layer = PositionalEncoding(
            encoding_type="relative", d_model=d_model
        )
        self.u = nn.Parameter(torch.zeros(num_heads, dim_head))
        self.v = nn.Parameter(torch.zeros(num_heads, dim_head))

        self.MemoryTransformerLayers = nn.ModuleList(
            [
                submodule(
                    d_model=d_model,
                    dim_mlp=transformer_config["dim_mlp"],
                    dropout=dropout,
                )
                for k in range(self.num_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, output_dim, bias=False)

    def init_mem(self):
        if self.mem_len > 0:
            mem = []
            param = next(self.parameters())
            for i in range(self.num_layers + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mem.append(empty)
            return mem
        else:
            return None

    def _update_mem(self, hids, mem, qlen, mlen):
        if mem is None:
            return None
        assert len(hids) == len(mem), "len(hids) != len(mem)"
        with torch.no_grad():
            new_mem = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mem[i], hids[i]], dim=0)
                # cat = torch.cat([mem[i], hids[i][-1]], dim=0)
                new_mem.append(cat[beg_idx:end_idx].detach())
        return new_mem

    def forward(self, inputs: Tensor, mem: Tensor = None):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features] 
            mem: memory from previous sequence. 

        Returns: 
            Transformer output, of shape: [target_seq_len, batch_size, output_dim]
        """
        if not mem:
            mem = self.init_mem()
        qlen, bsz, _ = inputs.size()

        mlen = mem[0].size(0) if mem is not None else 0
        klen = mlen + qlen

        attn_mask = None

        hids = []
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, dtype=inputs.dtype, device=inputs.device
        )
        pos_emb = self.positional_encoding_layer(pos_seq)
        print(f"Pos emd shape: {pos_emb.shape}")

        core_out = self.drop(inputs)
        pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.MemoryTransformerLayers):
            hids.append(core_out)
            mem_i = None if mem is None else mem[i]
            core_out = layer(
                core_out, pos_emb, self.u, self.v, attn_mask=None, mem=mem_i,
            )
            hids.append(core_out)

        core_out = self.drop(core_out)
        core_out = self.output_layer(core_out)
        new_mem = self._update_mem(hids, mem, mlen, qlen)

        return core_out, new_mem

    def reset(self):
        self.init_mem()
        for layer in self.MemoryTransformerLayers:
            if type(TransformerBlockBase):
                layer.reset()


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

        self.hidden_1 = torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
            experiment_config["device"]
        )
        self.hidden_2 = torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
            experiment_config["device"]
        )
        self.gated_layer_1 = nn.GRU(
            d_model, lstm_config["hidden_dim"], num_layers=lstm_config["num_layers"]
        )
        self.gated_layer_2 = nn.GRU(
            d_model, lstm_config["hidden_dim"], num_layers=lstm_config["num_layers"]
        )

        # self.gated_layer_1 = GatedRecurrentUnit(d_model)
        # self.gated_layer_2 = GatedRecurrentUnit(d_model)

        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)
        self.relu = nn.ReLU(inplace=True)

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
        y = self.relu(y)
        output, self.hidden_2 = self.gated_layer_2(x + y, self.hidden_2)
        return output

    def reset(self):
        self.hidden_1 = torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
            experiment_config["device"]
        )
        self.hidden_2 = torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
            experiment_config["device"]
        )


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
        self.hidden = torch.zeros(1, 1, d_model).to(experiment_config["device"])
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
        y, self.hidden = self.gated_layer(y, self.hidden)
        return y

    def reset(self):
        self.hidden = torch.zeros(1, 1, self.d_model).to(experiment_config["device"])


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
