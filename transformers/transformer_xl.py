"""Transformer-XL Architecture from the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Paper. 

The Transformer-XL extends the original Transformer by adding a segment-level encoding recurrence mechanism 
and a novel relative positional encoding scheme. The Transformer-XL was demonstrated to learn longer-term dependencies 
than RNN's and Transformers, achieved better performance on both short and long sequences, and is far faster than 
vanilla transformers at evaluation. 

The original paper can be found here: https://arxiv.org/pdf/1901.02860.pdf. 
The original opensource implementation can be found here: 
https://github.com/kimiyoung/transformer-xl
This implementation was strongly based on the fastai Transformer-XL implementation, which can be found here: 
https://github.com/fastai/fastai/blob/master/fastai/text/models/transformer.py#L175
"""
import math 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers.positional_encoding_layer import RelativePositionalEncoding
from transformers.attention_layer import RelativeMultiHeadAttention

Tensor = torch.Tensor
# TODO: Need to standardies variable names

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


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, x:Tensor, r, y, v, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, mems=mems)
        output = self.pos_ff(output)

        return output


class TransformerXL(nn.Module):
    "TransformerXL Model"
    def __init__(self, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=False, scale:bool=True,
                 mask:bool=True, mem_len:int=0):
        super(TransformerXL, self).__init__()
        # self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.drop_emb = nn.Dropout(0.0)
        self.u = nn.Parameter(torch.Tensor(n_heads, d_head)) 
        self.v = nn.Parameter(torch.Tensor(n_heads, d_head)) 
        self.mem_len,self.n_layers,self.d_model,self.mask = mem_len,n_layers,d_model,mask
        self.init = False
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale) for k in range(n_layers)])

    def reset(self):
        "Reset the internal memory."
        self.hidden = [next(self.parameters()).data.new(0) for i in range(self.n_layers+1)]

    def _update_mems(self, hids):
        if not getattr(self, 'hidden', False): return None
        assert len(hids) == len(self.hidden), 'len(hids) != len(self.hidden)'
        with torch.no_grad():
            for i in range(len(hids)):
                cat = torch.cat([self.hidden[i], hids[i]], dim=1)
                self.hidden[i] = cat[:,-self.mem_len:].detach()

    def select_hidden(self, idxs): self.hidden = [h[idxs] for h in self.hidden]

    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init:
            self.reset()
            self.init = True
        x = x.view(x.size(0), 1, x.size(1))
        bs,x_len, _ = x.size()
        inp = x
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        # mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None] if self.mask else None 
        mask = None
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        # return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]
        return core_out






