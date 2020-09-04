import torch.nn as nn
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import (
    RelativeMultiHeadAttention,
    MultiheadLinearAttention,
)
from configs.experiment_config import experiment_config
import torch

Tensor = torch.Tensor


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.0):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)

    def forward(self, x):
        x = self.pos_encoder(inputs * math.sqrt(self.d_model))
        return self.attention(x)


class LMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.0):
        super(LMHA, self).__init__()
        self.pos_encoder = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.attention = MultiheadLinearAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=experiment_config["max_steps_per_episode"],
        )

    def forward(self, x):
        x = self.pos_encoder(inputs)
        return self.attention(x)


class RMHA(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 1, mem_len: int = 0, dropout: float = 0.0
    ):
        super(RMHA, self).__init__()
        dim_head = d_model // num_heads
        self.attention = RelativeMultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.mem_len = mem_len
        self.positional_encoding_layer = PositionalEncoding(
            encoding_type="relative", d_model=d_model
        )
        self.drop = nn.Dropout(dropout)
        self.u = nn.Parameter(torch.Tensor(num_heads, dim_head))
        self.v = nn.Parameter(torch.Tensor(num_heads, dim_head))

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
                new_mem.append(cat[beg_idx:end_idx].detach())
        return new_mem

    def forward(self, x):
        output, self.mem = self._forward(inputs, self.mem)
        return output

    def _forward(self, inputs: Tensor, mem: Tensor = None):
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

        pos_seq = torch.arange(
            klen - 1, -1, -1.0, dtype=inputs.dtype, device=inputs.device
        )
        pos_emb = self.positional_encoding_layer(pos_seq)

        core_out = self.drop(inputs)
        pos_emb = self.drop(pos_emb)

        hids = []
        # hids.append(core_out)
        core_out = self.attention(core_out, pos_emb, self.u, self.v, mem=mem)
        hids.append(core_out)

        # for i, layer in enumerate(self.TransformerXLs):
        #     mem_i = None if mem is None else mem[i]
        #     core_out = layer(
        #         core_out, pos_emb, self.u, self.v, attn_mask=attn_mask, mem=mem_i
        #     )
        #     hids.append(core_out)

        core_out = self.dropout(core_out)
        core_out = self.output_layer(core_out)
        new_mem = self._update_mem(hids, mem, mlen, qlen)
        return core_out, new_mem

