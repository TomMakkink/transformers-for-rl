import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


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
                    query: (target_seq_len, batch_size, dim), 
                    key:   (source_seq_len, batch_size, dim),
                    value: (source_seq_len, batch_size, dim)
            - key_padding_mask: if provided, padded elements in the key will be ignored
              by the attention.
            - attn_mask:
                2D mask that prevents attention in certain positions, 
                with shape (target_seq_len, source_seq_len). 

        Returns: 
            Attention output with shape (target_seq_len, batch_size, dim)
        """
        return F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,                      
            self.in_proj_weight, self.in_proj_bias, dropout_p=self.dropout,                     
            out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, 
            add_zero_attn=None, bias_k=None, bias_v=None,                
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)

