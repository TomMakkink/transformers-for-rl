import math
import torch
import torch.nn as nn

Tensor = torch.Tensor


class PositionalEncoding(nn.Module):
    """
    Wrapper for the positional-encoding layer, which provides the model with information
    regarding the position of inputs in the input sequence.
    Type of encoding: absolute and relative.
    """

    def __init__(
        self,
        encoding_type: str,
        d_model: int,
        max_len: int = 5000,
    ):
        """
        Args:
            encoding_type: type of encoding: ["absolute", "relative"].
            d_model: number of expected features in the input.
            max_len: max context length.
        """
        super(PositionalEncoding, self).__init__()
        if encoding_type.lower() == "absolute":
            self.encoder = AbsolutePositionalEncoding(d_model, max_len)
        elif encoding_type.lower() == "relative":
            self.encoder = RelativePositionalEncoding(d_model)
        else:
            raise ValueError("Possible encodings are: 'relative' and 'absolute'")

    def forward(self, x: Tensor):
        return self.encoder(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding as used in the "Attention is All you Need"
    paper: https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(AbsolutePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)]
        return x


class RelativePositionalEncoding(nn.Module):
    def __init__(self, demb):
        super(RelativePositionalEncoding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]
