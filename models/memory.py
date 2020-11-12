import torch.nn as nn
from models.custom_lstm import CustomLSTM
from transformers.transformer_models import TransformerModel, MemoryTransformerModel
from transformers.transformer_submodules import (
    TransformerBlock,
    ReZeroBlock,
    LinformerBlock,
    TransformerXLBlock,
    GTrXLBlock,
)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(LSTM, self).__init__()
        self.memory = CustomLSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.hidden = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        x, self.hidden, viz_data = self.memory(x, self.hidden)
        return x

    def reset(self):
        self.hidden = None


class Canonical(nn.Module):
    """
    Canonical Transformer model from the 'Attention is All you Need' paper.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_mlp,
        max_sequence_length,
        dropout,
        name,
    ):
        super(Canonical, self).__init__()
        canonical_submodule = TransformerBlock(
            d_model=input_dim, num_heads=num_heads, dim_mlp=dim_mlp, dropout=dropout
        )
        self.name = name
        self.memory = TransformerModel(
            d_model=input_dim,
            output_dim=output_dim,
            max_sequence_length=max_sequence_length,
            submodule=canonical_submodule,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, viz_data = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        pass

    def get_name(self):
        return self.name


class ReZero(nn.Module):
    """
    ReZero Transformer model from the 'ReZero is all you Need' paper.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_mlp,
        max_sequence_length,
        dropout,
        name,
    ):
        super(ReZero, self).__init__()
        self.name = name
        rezero_submodule = ReZeroBlock(
            d_model=input_dim, num_heads=num_heads, dim_mlp=dim_mlp, dropout=dropout
        )
        self.memory = TransformerModel(
            d_model=input_dim,
            output_dim=output_dim,
            max_sequence_length=max_sequence_length,
            submodule=rezero_submodule,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, viz_data = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        pass

    def get_name(self):
        return self.name


class Linformer(nn.Module):
    """
    Linformer.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_mlp,
        dropout,
        max_sequence_length,
        k,
        name,
    ):
        super(Linformer, self).__init__()
        self.name = name
        linformer_submodule = LinformerBlock(
            d_model=input_dim,
            num_heads=num_heads,
            dim_mlp=dim_mlp,
            max_seq_len=max_sequence_length,
            k=k,
            dropout=dropout,
        )

        self.memory = TransformerModel(
            d_model=input_dim,
            output_dim=output_dim,
            max_sequence_length=max_sequence_length,
            submodule=linformer_submodule,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, viz_data = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        pass

    def get_name(self):
        return self.name


class TransformerXL(nn.Module):
    """
    Transformer-XL model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_mlp,
        mem_len,
        dropout,
        name,
    ):
        super(TransformerXL, self).__init__()
        self.name = name
        xl_submodule = TransformerXLBlock(
            d_model=input_dim,
            dim_mlp=dim_mlp,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
        )
        self.mem = None

        self.memory = MemoryTransformerModel(
            d_model=input_dim,
            output_dim=output_dim,
            submodule=xl_submodule,
            num_layers=num_layers,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, viz_data, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.mem = None
        self.memory.reset()

    def get_name(self):
        return self.name


class GTrXL(nn.Module):
    """
    GTrXL model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        num_heads,
        dim_mlp,
        mem_len,
        dropout,
        device,
        name,
    ):
        super(GTrXL, self).__init__()
        self.name = name
        gtrxl_submodule = GTrXLBlock(
            d_model=input_dim,
            dim_mlp=dim_mlp,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
            device=device,
        )
        self.mem = None
        self.memory = MemoryTransformerModel(
            d_model=input_dim,
            output_dim=output_dim,
            submodule=gtrxl_submodule,
            num_layers=num_layers,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, viz_data, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.mem = None
        self.memory.reset()

    def get_name(self):
        return self.name


# class Memory(nn.Module):
#     """
#     Memory wrapper that is either an LSTM or a Transformer.
#     """
#
#     def __init__(self, memory_type, input_dim, output_dim):
#         super(Memory, self).__init__()
#         self.memory = None
#         self.memory_type = None
#
#         if memory_type is not None:
#             self.memory_type = memory_type.lower()
#             print(f"Using {self.memory_type}...")
#
#             self.visualisation_data = [[]]
#
#             if self.memory_type == "lstm":
#                 self.memory = CustomLSTM(input_size=input_dim,
#                                          hidden_size=lstm_config["hidden_dim"])
#                 self.hidden = None
#
#             elif self.memory_type in ["vanilla", "rezero", "linformer", "mha", "lmha"]:
#                 submodule = get_transformer_submodule(self.memory_type)
#                 self.memory = TransformerModel(input_dim, output_dim, submodule)
#
#             elif self.memory_type in ["gtrxl", "xl", "rmha", "gmha"]:
#                 submodule = get_transformer_submodule(self.memory_type)
#                 self.mem = None
#                 self.memory = MemoryTransformerModel(input_dim, output_dim, submodule)
#
#     def forward(self, x):
#         """
#         x: shape [batch_size, seq_len, feature_dim]
#         """
#         if (type(self.memory) is nn.LSTM) or (type(self.memory) is CustomLSTM):
#             x, self.hidden, viz_data = self.memory(x, self.hidden)
#
#         # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
#         x = x.transpose(0, 1)
#         if type(self.memory) == MemoryTransformerModel:
#             x, viz_data, self.mem = self.memory(x, self.mem)
#         elif type(self.memory) == TransformerModel:
#             x, viz_data = self.memory(x)
#         x = x.transpose(0, 1)
#
#         self.visualisation_data[-1].append(viz_data)
#         return x
#
#     def reset(self):
#         if self.memory_type == "lstm":
#             self.hidden = None
#         elif type(self.memory) == MemoryTransformerModel:
#             self.mem = None
#             self.memory.reset()
#         if self.memory_type is not None:
#             self.visualisation_data.append([])
