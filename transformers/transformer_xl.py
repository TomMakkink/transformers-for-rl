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


class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."
    def __init__(self, d:int): 
        super(PositionalEncoding, self).__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))

    def forward(self, pos:torch.Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc


def feed_forward(d_model:int, d_ff:int, ff_p:float=0.0):
    return nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(ff_p),
            nn.LayerNorm(d_model),
        )


class MultiHeadAttention(nn.Module):
    "MutiHeadAttention"
    def __init__(self, n_heads:int, d_model:int, d_head:int, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True):
        super(MultiHeadAttention, self).__init__()
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.attention = nn.Linear(d_model, 3 * n_heads * d_head, bias=bias)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(attn_p),nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None, **kwargs):
        return self.ln(x + self.drop_res(self.out(self._apply_attention(x, mask=mask, **kwargs))))

    def _apply_attention(self, x:torch.Tensor, mask:torch.Tensor=None):
        bs,x_len = x.size(0),x.size(1)
        wq,wk,wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        attn_score = torch.matmul(wq, wk)
        if self.scale: attn_score.div_(self.d_head ** 0.5)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bs, x_len, -1)


def _line_shift(x:torch.Tensor, mask:bool=False):
    "Shift the line i of `x` by p-i elements to the left, is `mask` puts 0s on the diagonal."
    bs,nh,n,p = x.size()
    x_pad = torch.cat([x.new_zeros(bs,nh,n,1), x], dim=3)
    x_shift = x_pad.view(bs,nh,p + 1,n)[:,:,1:].view_as(x)
    if mask: x_shift.mul_(torch.tril(x.new_ones(n,p), p-n)[None,None,])
    return x_shift


class MultiHeadRelativeAttention(MultiHeadAttention):
    "MultiHeadAttention with relative positional encoding."
    def __init__(self, n_heads:int, d_model:int, d_head:int, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True):
        super(MultiHeadRelativeAttention, self).__init__(n_heads, d_model, d_head)
        self.r_attn = nn.Linear(d_model, n_heads * d_head, bias=bias)
        
    def _apply_attention(self, x:torch.Tensor, r:torch.Tensor=None, u:torch.Tensor=None, v:torch.Tensor=None, 
                        mask:torch.Tensor=None, mem:torch.Tensor=None):
        # Notations from the paper: x input, r vector of relative distance between two elements, u et v learnable
        # parameters of the model common between all layers, mask to avoid cheating and mem the previous hidden states.
        bs,x_len,seq_len = x.size(0),x.size(1),r.size(0)
        context = x if mem is None else torch.cat([mem, x], dim=1)
        attn = self.attention(context)
        wq,wk,wv = torch.chunk(attn, 3, dim=-1)
        wq = wq[:,-x_len:]
        # No embedding dimension, so have changed view function. 
        # wq,wk,wv = map(lambda x:x.view(bs, x_len, self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = map(lambda x:x.view(bs, self.n_heads, self.d_head), (wq,wk,wv))
        # wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        wk = wk.permute(0, 2, 1)
        wkr = self.r_attn(r)
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        wkr = wkr.permute(1,2,0)
        ### compute attention score (AC is (a) + (c) and BD is (b) + (d) in the paper)
        AC = torch.matmul(wq+u,wk)
        # BD = _line_shift(torch.matmul(wq+v, wkr))
        BD = torch.matmul(wq+v, wkr)
        if self.scale: attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        # if mask is not None:
        #     attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_prob = attn_prob.permute(0, 2, 1)
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.contiguous().view(bs, x_len, -1)


class DecoderLayer(nn.Module):
    "Basic block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True):
        super(DecoderLayer, self).__init__()
        self.mhra = MultiHeadRelativeAttention(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None, **kwargs): 
        return self.ff(self.mhra(x, mask=mask, **kwargs))


class TransformerXL(nn.Module):
    "TransformerXL Model"
    def __init__(self, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=False, scale:bool=True,
                 mask:bool=True, mem_len:int=0):
        super(TransformerXL, self).__init__()
        # self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.drop_emb = nn.Dropout(0.0)
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) 
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) 
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
        bs,x_len = x.size()
        # inp = self.drop_emb(self.encoder(x)) #.mul_(self.d_model ** 0.5)
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
        # core_out = inp[:,-x_len:]
        # if self.mem_len > 0 : self._update_mems(hids)
        # return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]
        return x


def init_transformer(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 0., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:     nn.init.constant_(m.bias, 0.)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 1., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:     nn.init.constant_(m.bias, 0.)
    elif classname.find('TransformerXL') != -1:
        if hasattr(m, 'u'): nn.init.normal_(m.u, 0., 0.02)
        if hasattr(m, 'v'): nn.init.normal_(m.v, 0., 0.02)
