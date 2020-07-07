import torch 
import torch.nn as nn 
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import GTrXLBlock

Tensor = torch.Tensor

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
        d_model:int, 
        output_dim:int,
        num_layers:int, 
        num_heads:int, 
        dim_mlp:int,     
        dropout:float=0.0,
        mem_len:int=None,
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
        self.positional_encoding_layer = PositionalEncoding(encoding_type="relative", d_model=d_model)
        self.u = nn.Parameter(torch.Tensor(num_heads, dim_head))
        self.v = nn.Parameter(torch.Tensor(num_heads, dim_head))
        self.dropout = nn.Dropout(dropout)

        self.GTrXLs = nn.ModuleList([
            GTrXLBlock(
                num_heads=num_heads,
                d_model=d_model,
                dim_mlp=dim_mlp,
                dropout=dropout, 
                mem_len=mem_len,
            )
            for k in range(num_layers)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, output_dim, bias=False),
            nn.ReLU(),
        )

    
    def init_mem(self):
        if self.mem_len > 0:
            mem = []
            param = next(self.parameters())
            for i in range(self.num_layers+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
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
            GTrXL output of shape [source_seq_len, batch_size, output_dim]
        """
        if not mem: mem = self.init_mem()
        qlen, bsz, features = inputs.size()
        
        mlen = mem[0].size(0) if mem is not None else 0
        klen = mlen + qlen 
        hids = []

        pos_seq = torch.arange(klen-1, -1, -1.0, dtype=inputs.dtype, device=inputs.device)
        pos_emb = self.positional_encoding_layer(pos_seq)

        core_out = self.dropout(inputs)
        pos_emb = self.dropout(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.GTrXLs):
            mem_i = None if mem is None else mem[i]
            core_out = layer(core_out, pos_emb, self.u, self.v, mem=mem_i)
            hids.append(core_out)

        core_out = self.dropout(core_out)

        new_mem = self._update_mem(hids, mem, mlen, qlen)
        pred_hid = core_out[-qlen:]
        loss = self.output_layer(pred_hid)

        if new_mem is None:
            return [loss]
        else:
            return [loss] + new_mem