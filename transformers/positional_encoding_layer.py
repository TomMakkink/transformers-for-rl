import math
import torch 
import torch.nn as nn

Tensor = torch.Tensor

# TODO: Refactor this so you only have one PositionalEncoding class, and you chose the encoding scheme: 
# e.g. relative or absolute 


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding as used in the "Attention is All you Need" 
    paper: https://arxiv.org/abs/1706.03762. 
    Provides the model with information regarding the absolute position of inputs 
    in the input sequence. 
    """
    def __init__(self, d_model:Tensor, dropout:float=0.1, max_len:int=5000):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding as used in the "Transformer-XL: Attentive Language Models 
    Beyond a Fixed-Length Context" paper: https://arxiv.org/pdf/1901.02860.pdf. 
    Provides the model with information regarding the relative position of inputs 
    in the input sequence. 
    """
    def __init__(self, d_model:int, dropout:float=0.1): 
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        freq = 1 / (10000 ** (torch.arange(0., d_model, 2.)/d_model))
        self.register_buffer('freq', freq)

    def forward(self, pos:torch.torch.Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        enc = enc[:, None, :]
        return self.dropout(enc)