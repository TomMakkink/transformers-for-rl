import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import RelativeMultiHeadAttention, TransformerXLBlock

Tensor = torch.Tensor 

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
        d_model:int, 
        output_dim:int,
        num_layers:int, 
        num_heads:int, 
        dim_mlp:int,
        dropout:float=0.0,
        mem_len:int=None,
        tgt_len:int=None,
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
        self.positional_encoding_layer = PositionalEncoding(d_model=d_model)
        self.u = nn.Parameter(torch.Tensor(num_heads, dim_head))
        self.v = nn.Parameter(torch.Tensor(num_heads, dim_head))

        self.TransformerXLs = nn.ModuleList([
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
        ])

        self.output_layer = nn.Linear(d_model, output_dim, bias=False)

    def init_mem(self):
        if self.mem_len > 0:
            mem = []
            param = next(self.parameters())
            for i in range(self.num_layers+1):
                empty = torch.empty(0)
                mem.append(empty)
            return mem
        else:
            return None

    def _update_mem(self, hids, mem, qlen, mlen):
        if mem is None: return None
        assert len(hids) == len(mem), 'len(hids) != len(mem)'
        with torch.no_grad():
            new_mem = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mem[i], hids[i]], dim=0)
                new_mem.append(cat[beg_idx:end_idx].detach())
        return new_mem


    def forward(self, inputs:Tensor, *mem:Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features] 
            mem: memory from previous sequence. 

        Returns: 
            Transformer output, of shape: [source_seq_len, batch_size, output_dim]
        """
        if not mem: mem = self.init_mem()
        qlen, bsz, features = inputs.size()
        
        mlen = mem[0].size(0) if mem is not None else 0
        klen = mlen + qlen 
        hids = []

        pos_seq = torch.arange(klen-1, -1, -1.0)
        pos_emb = self.positional_encoding_layer(pos_seq)

        core_out = self.drop(inputs)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.TransformerXLs):
            mem_i = None if mem is None else mem[i]
            core_out = layer(core_out, pos_emb, self.u, self.v, mem=mem_i)
            hids.append(core_out)

        core_out = self.drop(core_out)

        new_mem = self._update_mem(hids, mem, mlen, qlen)
        pred_hid = core_out[-qlen:]
        loss = self.output_layer(pred_hid)

        if new_mem is None:
            return [loss]
        else:
            return [loss] + new_mem




