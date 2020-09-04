import torch
import torch.nn as nn
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import (
    GatedRecurrentUnit,
    RelativeMultiHeadAttention,
    PositionWiseMLP,
)

Tensor = torch.Tensor


class GTrXLBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dim_mlp: int,
        use_scale: bool = True,
        dropout: float = 0.0,
        mem_len: int = None,
    ):
        super(GTrXLBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.attention = RelativeMultiHeadAttention(
            num_heads, d_model, dropout, mem_len=mem_len
        )

        self.gated_layer_1 = GatedRecurrentUnit(d_model)
        self.gated_layer_2 = GatedRecurrentUnit(d_model)

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
        y = self.gated_layer_1([x, y])

        # Position-wise MLP
        x = y
        y = self.layer_norm_2(y)
        y = self.pos_wise_mlp(y)
        y = self.relu(y)
        output = self.gated_layer_2([x, y])
        return output


class GTrXL(nn.Module):
    """
    GTrXL Transformer model from the "Stabilizing Transformers for Reinforcement Learning" paper: 
    https://arxiv.org/abs/1910.06764. 
    
    The GTrXL modifies the Transformer-XL architecture by chaning 
    the position of the layer normalisation layer, and add additional Gated Recurrent Unit layer. 
    These modifications were demonstrated to improve the stability and learning speed of the original
    Transformer and Transformer XL in a variety of partially observable RL domains. 
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float = 0.0,
        mem_len: int = None,
    ):
        """
        Args: 
            d_model: number of expected features in the input. 
            output_dim: output dimension of the model.  
            num_layers: number of submodules in the transformer. 
            num_heads: number of attention heads.  
            dim_mlp: inner dimension of multilayer perceptron. 
            dropout: dropout. Default: 0.0. 
            mem_len: length of memory. Default: None. 
        """
        super(GTrXL, self).__init__()
        dim_head = d_model // num_heads
        self.mem_len = mem_len
        self.num_layers = num_layers
        self.positional_encoding_layer = PositionalEncoding(
            encoding_type="relative", d_model=d_model
        )
        self.u = nn.Parameter(torch.zeros(num_heads, dim_head))
        self.v = nn.Parameter(torch.zeros(num_heads, dim_head))
        self.dropout = nn.Dropout(dropout)

        self.GTrXLs = nn.ModuleList(
            [
                GTrXLBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    dim_mlp=dim_mlp,
                    dropout=dropout,
                    mem_len=mem_len,
                )
                for k in range(num_layers)
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
                new_mem.append(cat[beg_idx:end_idx].detach())
        return new_mem

    def forward(self, inputs: Tensor, mem: Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features]
            mem: memory from previous sequence. 

        Returns: 
            GTrXL output of shape [source_seq_len, batch_size, output_dim]
        """
        if not mem:
            mem = self.init_mem()
        qlen, bsz, features = inputs.size()

        mlen = mem[0].size(0) if mem is not None else 0
        klen = mlen + qlen

        # Masking
        attn_mask = torch.triu(inputs.new_ones(qlen, klen), diagonal=1 + mlen).bool()[
            :, :, None
        ]

        hids = []
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, dtype=inputs.dtype, device=inputs.device
        )
        pos_emb = self.positional_encoding_layer(pos_seq)

        core_out = self.dropout(inputs)
        pos_emb = self.dropout(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.GTrXLs):
            mem_i = None if mem is None else mem[i]
            core_out = layer(
                inputs=core_out,
                r=pos_emb,
                u=self.u,
                v=self.v,
                attn_mask=attn_mask,
                mem=mem_i,
            )
            hids.append(core_out)

        # core_out = self.dropout(core_out)
        core_out = self.output_layer(core_out)
        new_mem = self._update_mem(hids, mem, mlen, qlen)
        return core_out, new_mem

        # core_out = self.dropout(core_out)

        # new_mem = self._update_mem(hids, mem, mlen, qlen)
        # pred_hid = core_out[-qlen:]
        # loss = self.output_layer(pred_hid)

        # if new_mem is None:
        #     return [loss]
        # else:
        #     return [loss] + new_mem
