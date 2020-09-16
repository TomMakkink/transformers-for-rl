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
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        """
        Args:
            d_model: number of expected features in the input. 
            num_heads: number of attention heads. 
            dropout: dropout layer on attention output weigths. Default: 0.0. 
            bias: add bias as module parameter. Default: False. 
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim * num_heads == self.d_model
        ), "d_model must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
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
        tgt_len, bsz, d_model = query.size()
        assert key.size() == value.size()
        head_dim = self.d_model // self.num_heads
        assert (
            head_dim * self.num_heads == self.d_model
        ), "d_model must be divisible by num_heads"
        scaling = 1 / (head_dim ** 0.5)

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
                3, dim=-1
            )
        else:
            raise RuntimeError(f"Query, key, value of unequal size.")

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        attn_output_weights = attn_output_weights.view(
            bsz * self.num_heads, tgt_len, src_len
        )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_model)
        )
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
        num_heads: int,
        d_model: int,
        dropout: float = 0.0,
        dropout_attn: float = 0.0,
        bias: bool = False,
        mem_len=None,
        **kwargs,
    ):
        """
        Args:
            d_model: number of expected features in the input.  
            num_heads: number of attention heads. 
            dropout: dropout layer on attention output weigths. Default: 0.0. 
            bias: add bias as module parameter. Default: False. 
            mem_len: length of memory. 
        """
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim * num_heads == self.d_model
        ), "d_model must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.qkv_net = nn.Linear(
            self.d_model, 3 * self.num_heads * self.head_dim, bias=False
        )
        self.r_net = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)

        self.scale = 1 / (self.head_dim ** 0.5)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(
        self,
        x: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask=None,
        mem: Tensor = None,
    ):
        """
        Args: 
            x: Input tensor, shape [target_seq_len, batch_size, dim]
            r: relative distance between two elements.
            u, v: learnable parameters of model common between layers 
            attn_mask: prevent model from looking forward using attention mask
            mem: fixed cache of previous hidden states, of shape: [target_seq_len, batch_size, dim]

        Returns: 
            Attention output with shape [target_seq_len, batch_size, dim]
        """
        qlen, rlen, bsz = x.size(0), r.size(0), x.size(1)

        context = x if mem is None else torch.cat([mem, x], 0)
        w_heads = self.qkv_net(context)

        rk = self.r_net(r)
        wq, wk, wv = torch.chunk(w_heads, 3, dim=-1)
        wq = wq[-qlen:]
        klen = wk.size(0)

        # Shape: [qlen, batch_size, num_head, head_dim]
        wq = wq.view(qlen, bsz, self.num_heads, self.head_dim)
        wk = wk.view(klen, bsz, self.num_heads, self.head_dim)
        wv = wv.view(klen, bsz, self.num_heads, self.head_dim)

        # Shape: [tgt_len, num_head, head_dim]
        rk = rk.view(rlen, self.num_heads, self.head_dim)

        # Compute attention score
        AC = torch.einsum("ibnd,jbnd->ijbn", (wq + u, wk))
        BD = torch.einsum("ibnd,jnd->ijbn", (wq + v, rk))
        BD = self._rel_shift(BD)

        # [qlen, klen, batch_size, num_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # [qlen, klen, batch_size, num_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        # Compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, wv))

        # [tgt_len, batch_size, num_head * head_dim]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.num_heads * self.head_dim
        )

        attn_out = self.out_proj(attn_vec)
        attn_out = self.dropout(attn_out)

        # Residual connection + layer normalization
        output = self.layer_norm(x + attn_out)

        return output


class MultiheadLinearAttention(nn.Module):
    """
    Taken from: https://github.com/facebookresearch/pytext/blob/master/pytext/models/representations/transformer/multihead_linear_attention.py
    This is a TorchScriptable implementation of MultiheadLinearAttention:
    https://arxiv.org/pdf/2006.04768.pdf. from fairseq for the purposes of
    creating a productionized Linformer model. It distills just
    the elements which are required to implement the RoBERTa use cases of
    MultiheadLinearAttention, and within that is restructured and rewritten to be able
    to be compiled by TorchScript for production use cases.
    The default constructor values match those required to import the public
    RoBERTa weights. Unless you are pretraining your own model, there's no need to
    change them.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        k: int = 4,
        # compress_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.kput_projection = nn.Linear(d_model, d_model, bias=bias)
        self.vput_projection = nn.Linear(d_model, d_model, bias=bias)
        self.qput_projection = nn.Linear(d_model, d_model, bias=bias)

        self.output_projection = nn.Linear(d_model, d_model)
        if max_seq_len > 2:
            self.compress_k = nn.Linear(max_seq_len, k)

    def forward(self, query, key_padding_mask=None):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x source_length, where padding elements are indicated by 1s.
        """
        target_length, batch_size, d_model = query.size()

        # mask_batch_size, source_length = key_padding_mask.size()

        assert d_model == self.d_model
        # assert (
        #     batch_size == mask_batch_size
        # ), "query and key_padding_mask batch sizes differed"

        q = self.qput_projection(query)
        scaling = 1 / (self.head_dim ** 0.5)
        q *= scaling

        k_input = query.permute(1, 2, 0).contiguous()  # B * C * T
        k_input = (
            F.linear(k_input, self.compress_k.weight[:, 0:target_length])
            .permute(2, 0, 1)
            .contiguous()
        )
        k = self.kput_projection(k_input)

        v_input = query.permute(1, 2, 0).contiguous()  # B * C * T
        v_input = (
            F.linear(v_input, self.compress_k.weight[:, 0:target_length])
            .permute(2, 0, 1)
            .contiguous()
        )
        v = self.vput_projection(v_input)

        batch_heads = batch_size * self.num_heads

        q = q.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_heads, self.head_dim).transpose(0, 1)

        source_length = k.size(1)  # T_k
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.shape) == [batch_heads, target_length, source_length]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.shape) == [batch_heads, target_length, self.head_dim]
        attn = (
            attn.transpose(0, 1).contiguous().view(target_length, batch_size, d_model)
        )
        attn = self.output_projection(attn)

        return attn


class PositionWiseMLP(nn.Module):
    def __init__(self, d_model: Tensor, dim_mlp: int, dropout: float = 0.1):
        super(PositionWiseMLP, self).__init__()
        self.pos_wise_mlp = nn.Sequential(
            nn.Linear(d_model, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src):
        return self.pos_wise_mlp(src)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class GatedRecurrentUnit(nn.Module):
    def __init__(self, d_model):
        super(GatedRecurrentUnit, self).__init__()
        self.linear_wr = nn.Linear(d_model, d_model, bias=False)
        self.linear_ur = nn.Linear(d_model, d_model, bias=False)
        self.linear_wz = nn.Linear(d_model, d_model)
        self.linear_uz = nn.Linear(d_model, d_model, bias=False)
        self.linear_wg = nn.Linear(d_model, d_model, bias=False)
        self.linear_ug = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        # Manually set to allow learning of markov process during initial training. -2 used in paper.
        with torch.no_grad():
            self.linear_wz.bias.fill_(-2)

    def forward(self, inputs: Tensor):
        x, y = inputs
        assert x.shape == y.shape, "Inputs to GRU layer should be the same size"

        z = torch.sigmoid(self.linear_wz(y) + self.linear_uz(x))
        r = torch.sigmoid(self.linear_wr(y) + self.linear_ur(x))
        h_hat = torch.tanh(self.linear_wg(y) + self.linear_ug(r * x))

        return (1.0 - z) * x + z * h_hat


# class GatedRecurrentUnit(nn.Module):
#     def __init__(self, d_model:int):
#         super(GatedRecurrentUnit, self).__init__()
#         self.wr = nn.Parameter(torch.ones(d_model, d_model))
#         self.ur = nn.Parameter(torch.ones(d_model, d_model))
#         self.wz = nn.Parameter(torch.ones(d_model, d_model))
#         self.uz = nn.Parameter(torch.ones(d_model, d_model))
#         self.bz = nn.Parameter(torch.ones(d_model,))
#         self.wg = nn.Parameter(torch.ones(d_model, d_model))
#         self.ug = nn.Parameter(torch.ones(d_model, d_model))

#     def forward(self, inputs:Tensor):
#         x, y = inputs
#         batch_size, seq_len, d_model = x.size()
#         assert x.shape == y.shape, "Two inputs should be of the same size"

#         r = torch.sigmoid(torch.matmul(y, self.wr) + torch.matmul(x, self.ur))
#         z = torch.sigmoid(torch.matmul(y, self.wz) + torch.matmul(x, self.uz) - self.bz)
#         h = torch.tanh(torch.matmul(y, self.wg) + torch.matmul(r * x, self.ug))
#         return (1.0 - z) * x + z * h


class RelMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        pre_lnorm=False,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->ijbn", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

