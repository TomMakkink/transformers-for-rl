import math
import torch
import torch.nn as nn
import numpy as np

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
        max_len: int,
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
        elif encoding_type.lower() == "coordinate":
            self.encoder = CoordinateEncoding(d_model, max_len)
        elif encoding_type.lower() == "relative_coordinate":
            self.encoder = RelativeCoordinateEncoding(d_model)
        else:
            raise ValueError("Possible encodings are: 'relative' and 'absolute'")

    def forward(self, x: Tensor):
        return self.encoder(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding as used in the "Attention is All you Need"
    paper: https://arxiv.org/abs/1706.03762.
    """

    def __init__(self, d_model: int, max_len: int):
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


class CoordinateEncoding(nn.Module):
    """
    Adapted from: https://github.com/tensorflow/tensor2tensor/blob/21dba2c1bdcc7ab582a2bfd8c0885c217963bb4f/tensor2tensor/layers/common_attention.py#L460
    """

    def __init__(self, d_model: int, max_len: int):
        super(CoordinateEncoding, self).__init__()
        self.pos_encoder = AbsolutePositionalEncoding(d_model, max_len)

    def forward(self, x: Tensor):
        x = self.pos_encoder(x)
        x = add_timing_signal(x)
        return x


class RelativeCoordinateEncoding(nn.Module):
    def __init__(self, d_model: int):
        super(RelativeCoordinateEncoding, self).__init__()
        self.pos_encoder = RelativePositionalEncoding(d_model)

    def forward(self, x, pos_seq, min_timescale=1.0, max_timescale=1.0e4, bsz=None):
        pos_emb = self.pos_encoder(pos_seq, bsz)

        length = x.shape[1]
        channels = x.shape[2]
        time_emb = _gen_timing_signal(length, channels, min_timescale, max_timescale)
        return pos_emb, time_emb


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        float(num_timescales) - 1
    )
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(
        signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0]
    )
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).float()


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Add timing signals as outlined in
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    """
    length = x.shape[1]
    channels = x.shape[2]
    signal = _gen_timing_signal(length, channels, min_timescale, max_timescale)
    return x + signal
