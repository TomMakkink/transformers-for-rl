import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import PositionalEncoding

Tensor = torch.Tensor


class TransformerModel(nn.Module):
    """
    Transformer baseclass.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        max_sequence_length: int,
        submodule,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Args:
            d_model: number of expected features in the input.
            output_dim: output dimension of the model.
            dropout: dropout. Default: 0.0.
        """
        super(TransformerModel, self).__init__()
        assert isinstance(submodule, nn.Module), "Invalid Transformer submodule. "
        self.dropout = dropout
        self.pos_encoder = PositionalEncoding(
            encoding_type="absolute", d_model=d_model, max_len=max_sequence_length
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.submodules = nn.ModuleList([submodule for k in range(num_layers)])

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
        attn_output_weights = []
        for layer in self.submodules:
            x, attn_output_weight = layer(x)
            attn_output_weights.append(attn_output_weight)
        attn_output_weights = torch.stack(attn_output_weights)
        return self.out_layer(x), attn_output_weights


class MemoryTransformerModel(nn.Module):
    """
    Transformer base model that uses memory.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        submodule,
        num_layers: int,
        num_heads: int,
        mem_len: int,
        dropout: float,
    ):
        """
        Args:
            d_model: number of expected features in the input.
            output_dim = output dimension of the model.
        """
        super(MemoryTransformerModel, self).__init__()
        assert isinstance(submodule, nn.Module), "Invalid Transformer submodule. "
        num_heads = num_heads
        dim_head = d_model // num_heads
        dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.mem_len = mem_len
        self.num_layers = num_layers
        self.positional_encoding_layer = PositionalEncoding(
            encoding_type="relative", d_model=d_model
        )
        self.u = nn.Parameter(torch.zeros(num_heads, dim_head))
        self.v = nn.Parameter(torch.zeros(num_heads, dim_head))

        self.submodules = nn.ModuleList([submodule for k in range(self.num_layers)])

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

        core_out = self.drop(inputs)
        pos_emb = self.drop(pos_emb)

        attn_output_weights = []
        hids.append(core_out)
        for i, layer in enumerate(self.submodules):
            mem_i = None if mem is None else mem[i]
            core_out, attn_output_weight = layer(
                core_out,
                pos_emb,
                self.u,
                self.v,
                attn_mask=None,
                mem=mem_i,
            )
            hids.append(core_out)
            attn_output_weights.append(attn_output_weight)
        attn_output_weights = torch.stack(attn_output_weights)
        core_out = self.drop(core_out)
        core_out = self.output_layer(core_out)
        new_mem = self._update_mem(hids, mem, mlen, qlen)

        return core_out, attn_output_weights, new_mem

    def reset(self):
        self.init_mem()
        for layer in self.submodules:
            layer.reset()
