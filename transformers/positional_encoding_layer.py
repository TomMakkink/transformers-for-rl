import math
import torch 
import torch.nn as nn

Tensor = torch.Tensor

class PositionalEncoding(nn.Module):
    """
    Wrapper for the positional-encoding layer. Possibilites include: absolute and relative. 
    """
    def __init__(
        self, 
        encoding_type:str, 
        dim_model:int, 
        max_len:int=5000, 
        dropout:float=0.1
    ):
        super(PositionalEncoding, self).__init__()
        if encoding_type.lower() == "absolute":
            self.encoder = AbsolutePositionalEncoding(dim_model, dropout, max_len)
        elif encoding_type.lower() == "relative":
            self.encoder = RelativePositionalEncoding(dim_model, dropout)
        else:
            raise ValueError("Possible encodings are: 'relative' and 'absolute'")

    def forward(self, x:Tensor):
        return self.encoder(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding as used in the "Attention is All you Need" 
    paper: https://arxiv.org/abs/1706.03762. 
    Provides the model with information regarding the absolute position of inputs 
    in the input sequence. 
    """
    def __init__(self, dim_model:int, dropout:float=0.1, max_len:int=5000):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
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
    def __init__(self, dim_model:int, dropout:float=0.1): 
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        freq = 1 / (10000 ** (torch.arange(0., dim_model, 2.)/dim_model))
        self.register_buffer('freq', freq)

    def forward(self, pos:Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        enc = enc[:, None, :]
        return self.dropout(enc)