import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

Tensor = torch.Tensor

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer as described in "Attention is All You Need": https://arxiv.org/abs/1706.03762.

    The Multi-Head Attention (MHA) submodule computes H soft-attention layers for every 
    element in parallel. MHA operates by computing learned, linear projections of queries Q, 
    keys K, and values V, which are combined and use to compute soft attention.

    This implementation is strongly influenced by the official pytorch implementation.  
    """

    def __init__(
        self, 
        embed_dim:int, 
        n_heads:int,
        dropout:float=0.0,
        bias:bool=True,
        **kwargs,  
    ):
        """
        Args:
            embed_dim: embedding dimension. 
            n_heads: number of attention heads. 
            dropout: dropout layer on attention output weigths. Default: 0.0. 
            bias: add bias as module parameter. Default: True. 
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters() 

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query:Tensor, key:Tensor, value:Tensor):
        """
        Args: 
            query, key, value: Map a query and a set of key-value pairs to an output, 
                with shapes: 
                    query: [target_seq_len, batch_size, dim], 
                    key:   [source_seq_len, batch_size, dim],
                    value: [source_seq_len, batch_size, dim]
        Returns: 
            Attention output with shape [target_seq_len, batch_size, dim]
        """
        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()
        head_dim = self.embed_dim // self.n_heads
        assert head_dim * self.n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"
        scaling = 1 / (head_dim ** 0.5)
        
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            raise RuntimeError(f"Query, key, value of unequal size.")

        q = q * scaling 

        q = q.contiguous().view(tgt_len, bsz * self.n_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.n_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.n_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.n_heads, tgt_len, src_len]

        attn_output_weights = attn_output_weights.view(bsz * self.n_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.n_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output



class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding as described in the 
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context": 
    https://arxiv.org/pdf/1901.02860.pdf.  

    This variation of MHA includes a form of ‘memory’ to solve the context fragmentation
    problem, whereby the current input is fixed and cached during training so that it can be
    reused in the next input. In addition, two learnable parameters 'u' and 'v' are introduced.
    """
    def __init__(
        self, 
        n_heads:int,
        embed_dim:int, 
        dropout:float=0.0,
        bias:bool=False,
        mem_len=None, 
        **kwargs,  
    ):
        """
        Args:
            embed_dim: embedding dimension. 
            n_heads: number of attention heads. 
            dropout: dropout layer on attention output weigths. Default: 0.0. 
            bias: add bias as module parameter. Default: False. 
            mem_len: length of memory. 
        """
        super(RelativeMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"
        
        self.dropout = nn.Dropout(dropout)
        self.qkv_net = nn.Linear(self.embed_dim, 3 * self.n_heads * self.head_dim, bias=False)
        self.r_net = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.scale = 1 / (self.head_dim ** 0.5)

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]))
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]
        return x

    def forward(self, x:Tensor, r:Tensor, u:Tensor, v:Tensor, mems:Tensor=None):
        """
        Args: 
            x: Input tensor, shape [target_seq_len, batch_size, dim]
            r: relative distance between two elements, shape [source_seq_len, ?]
            u, v: learnable parameters of model common between layers 
            mem: fixed cache of previous hidden states, of shape: [target_seq_len, batch_size, dim]

        Returns: 
            Attention output with shape [target_seq_len, batch_size, dim]
        """
        qlen, rlen, bsz = x.size(0), r.size(0), x.size(1)

        context = x if mems is None else torch.cat([mems, x], 0)
        w_heads = self.qkv_net(context)
        rk = self.r_net(r)
        wq, wk, wv = torch.chunk(w_heads, 3, dim=-1)
        wq = wq[-qlen:]
        klen = wk.size(0)

        # Shape: [qlen, batch_size, num_head, head_dim]
        # wq,wk,wv = map(lambda x:x.view(tgt_len, bsz, self.n_heads, self.head_dim), (wq,wk,wv))
        wq = wq.view(qlen, bsz, self.n_heads, self.head_dim)
        wk = wk.view(klen, bsz, self.n_heads, self.head_dim)
        wv = wv.view(klen, bsz, self.n_heads, self.head_dim)

        # Shape: [tgt_len, num_head, head_dim]
        rk = rk.view(rlen, self.n_heads, self.head_dim)

        # Compute attention score 
        # Length of k, q, and v must be the same 
        # Shape: [tgt_len, tgt_len, batch_size, num_head]
        AC = torch.einsum('ibnd,jbnd->ijbn', (wq+u, wk))
        # Shape: [tgt_len, tgt_len, batch_size, num_head]
        BD = torch.einsum('ibnd,jnd->ijbn', (wq+v, rk))
        BD = self._rel_shift(BD)

        # [tgt_len x tgt_len x batch_size x num_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # [tgt_len x tgt_len x batch_size x num_head]
        attn_prob = F.softmax(attn_score, dim=1)
        # attn_score = self.dropout()    optionally could introduce another drop out layer here

        # Compute attention vector 
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, wv))

        # [tgt_len, batch_size, num_head * head_dim]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_heads * self.head_dim)

        attn_out = self.out_proj(attn_vec)
        attn_out = self.dropout(attn_out)

        # Residual connection + layer normalization 
        output = self.layer_norm(x + attn_out)

        return output
        


        

        


            
            

