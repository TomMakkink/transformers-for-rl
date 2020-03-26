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
            num_heads:int,
            dropout:float=0.0,
            bias:bool=True,
            **kwargs,  
    ):
        """
        Args:
            embed_dim: embedding dimension. 
            num_heads: number of attention heads. 
            dropout: dropout layer on attention output weigths. Default: 0.0. 
            bias: add bias as module parameter. Default: True. 
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

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

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args: 
            query, key, value: Map a query and a set of key-value pairs to an output, 
                with shapes: 
                    query: [target_seq_len, batch_size, dim], 
                    key:   [source_seq_len, batch_size, dim],
                    value: [source_seq_len, batch_size, dim]
            - key_padding_mask: if provided, padded elements in the key will be ignored
              by the attention.
            - attn_mask:
                2D mask that prevents attention in certain positions, 
                with shape [target_seq_len, source_seq_len]. 

        Returns: 
            Attention output with shape [target_seq_len, batch_size, dim]
        """
        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()
        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5
        
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            raise RuntimeError(f"Query, key, value of unequal size.")

        q = q * scaling 

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)


        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output

        # return F.multi_head_attention_forward(
        #     query, key, value, self.embed_dim, self.self.num_heads,                      
        #     self.in_proj_weight, self.in_proj_bias, dropout_p=self.dropout,                     
        #     out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, 
        #     add_zero_attn=None, bias_k=None, bias_v=None,                
        #     key_padding_mask=key_padding_mask, attn_mask=attn_mask)

