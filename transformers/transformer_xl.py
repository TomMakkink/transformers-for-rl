import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import RelativeMultiHeadAttention, PositionWiseMLP

Tensor = torch.Tensor


class TransformerXLBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dim_head: int,
        dim_mlp: int,
        dropout: float = 0.0,
        mem_len: int = None,
        tgt_len: int = None,
    ):
        super(TransformerXLBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(
            num_heads, d_model, dropout, mem_len=mem_len
        )
        self.pos_wise_mlp = PositionWiseMLP(d_model, dim_mlp, dropout)

    def forward(
        self,
        inputs: Tensor,
        r: Tensor,
        u: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        mem: Tensor = None,
    ):
        # Attention
        x = inputs
        y = self.attention(inputs, r, u, v, attn_mask, mem)
        y = self.layer_norm_1(x + y)

        # Position-wise MLP
        x = y
        y = self.pos_wise_mlp(y)
        output = self.layer_norm_2(x + y)
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
        d_model: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float = 0.0,
        mem_len: int = None,
        tgt_len: int = None,
    ):
        """
        Args: 
            d_model: number of expected features in the input.  
            output_dim = output dimension of the model. 
            num_layers: number of 'submodule' layers in the transformer. 
            num_heads: number of attention heads.  
            dim_mlp: inner dimension of multilayer perceptron. 
            dropout: dropout. Default: None. 
            mem_len: length of memory. Default: None. 
            tgt_len: length of target sequence. Default: None. 
        """
        super(TransformerXL, self).__init__()
        dim_head = d_model // num_heads
        self.drop = nn.Dropout(dropout)
        self.mem_len = mem_len
        self.tgt_len = tgt_len
        self.num_layers = num_layers
        self.positional_encoding_layer = PositionalEncoding(
            encoding_type="relative", d_model=d_model
        )
        self.u = nn.Parameter(torch.zeros(num_heads, dim_head))
        self.v = nn.Parameter(torch.zeros(num_heads, dim_head))

        self.TransformerXLs = nn.ModuleList(
            [
                TransformerXLBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    dim_head=dim_head,
                    dim_mlp=dim_mlp,
                    dropout=dropout,
                    mem_len=mem_len,
                    tgt_len=tgt_len,
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
                # print(hids[i][-1].shape)
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

        # Masking
        # attn_mask = torch.triu(inputs.new_ones(qlen, klen), diagonal=1 + mlen).bool()[
        #     :, :, None
        # ]
        attn_mask = None

        hids = []
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, dtype=inputs.dtype, device=inputs.device
        )
        pos_emb = self.positional_encoding_layer(pos_seq)

        core_out = self.drop(inputs)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.TransformerXLs):
            mem_i = None if mem is None else mem[i]
            core_out = layer(
                core_out, pos_emb, self.u, self.v, attn_mask=None, mem=mem_i
            )
            hids.append(core_out)

        # core_out = self.dropout(core_out)
        core_out = self.output_layer(core_out)
        new_mem = self._update_mem(hids, mem, mlen, qlen)
        # print(f"New mem: {new_mem}")
        return core_out, new_mem

        # TODO: Check the out layer here isn't destorying the memory mechanism somehow?
        # pred_hid = core_out[-qlen:]
        # loss = self.output_layer(pred_hid)

        # if new_mem is None:
        #     return [loss]
        # else:
        #     return [loss] + new_mem

