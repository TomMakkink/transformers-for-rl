import math 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers.positional_encoding_layer import RelativePositionalEncoding
from transformers.attention_layer import RelativeMultiHeadAttention

Tensor = torch.Tensor
# TODO: Need to standardies variable names
# TODO: Clean up code 

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src):
        core_out = self.CoreNet(src)
        output = self.layer_norm(src + core_out)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_head, d_inner, dropout, mem_len=None, **kwargs):
        super(DecoderLayer, self).__init__()
        self.attention = RelativeMultiHeadAttention(n_heads, d_head, dropout, mem_len=mem_len, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, x:Tensor, r:Tensor, u:Tensor, v:Tensor, mems:Tensor=None):
        output = self.attention(x, r, u, v, mems=mems)
        output = self.pos_ff(output)
        return output


class TransformerXL(nn.Module):
    """
        Transformer-XL Architecture from the "Transformer-XL: Attentive Language Models 
        Beyond a Fixed-Length Context" paper: https://arxiv.org/pdf/1901.02860.pdf.

        The Transformer-XL extends the original Transformer by adding a segment-level encoding recurrence mechanism 
        and a novel relative positional encoding scheme.

        This implementation is strongly based on the original Transformer-XL transformer implementation, 
        which can be found here: https://github.com/kimiyoung/transformer-xl.
    """
    def __init__(
        self, 
        n_layers:int, 
        n_heads:int, 
        d_model:int, 
        d_head:int, 
        d_inner:int,
        d_embed:int=None,
        dropout:float=0.0,
        dropoutattn:float=0.0,
        mem_len:int=None,
        tgt_len:int=None,
    ):
        """
        Args: 

        """
        super(TransformerXL, self).__init__()

        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head 
        self.d_embed = d_model if d_embed is None else d_embed
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)
        self.mem_len, self.tgt_len = mem_len, tgt_len

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                DecoderLayer(n_heads, d_model, d_head, d_inner, dropout, mem_len)
            )

        self.pos_emb = RelativePositionalEncoding(d_model)
        self.u = nn.Parameter(torch.Tensor(n_heads, d_head))
        self.v = nn.Parameter(torch.Tensor(n_heads, d_head))

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layers+1):
                empty = torch.empty(0)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None: return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems


    def forward(self, data, *mems):
        if not mems: mems = self.init_mems()
        tgt_len = data.size(0)
        qlen, bsz = data.size()

        # Word-embedding format
        data = data.view(data.size(0), 1, data.size(1))
        
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen 
        hids = []

        pos_seq = torch.arange(klen-1, -1, -1.0)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(data)
        pos_emb = self.drop(pos_emb)
        # Exclude positional embedding for simple cartpole problem 
        pos_emb = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.u, self.v, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)
        pred_hid = core_out[-tgt_len:]

        # loss = F.softmax(pred_hid, -1)
        loss = pred_hid
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems




