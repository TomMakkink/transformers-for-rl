import torch.nn as nn
from models.custom_lstm import CustomLSTM
from transformers.transformer_models import (
    TransformerModel,
    MemoryTransformerModel,
    AdaptiveComputationalTime,
)
from transformers.transformer_submodules import (
    TransformerBlock,
    ReZeroBlock,
    LinformerBlock,
    TransformerXLBlock,
    GTrXLBlock,
    UniversalTransformerBlock,
    ReZeroXLBlock,
    ReZeroGTrXLBlock,
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
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)

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
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)

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
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)

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
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)
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
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)
        self.mem = None
        self.memory.reset()

    def get_name(self):
        return self.name


class UniversalTransformer(nn.Module):
    """
    Universal Transformer model: https://arxiv.org/abs/1807.03819
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        dim_mlp,
        max_sequence_length,
        max_act_timesteps,
        halting_threshold,
        dropout,
        name,
    ):
        super(UniversalTransformer, self).__init__()
        ut_submodule = UniversalTransformerBlock(
            d_model=input_dim, num_heads=num_heads, dim_mlp=dim_mlp, dropout=dropout
        )
        self.name = name
        self.memory = AdaptiveComputationalTime(
            d_model=input_dim,
            output_dim=output_dim,
            submodule=ut_submodule,
            max_sequence_length=max_sequence_length,
            max_act_timesteps=max_act_timesteps,
            halting_threshold=halting_threshold,
        )
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight, meta_info = self.memory(x)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)


class ReZeroXL(nn.Module):
    """
    Transformer-XL model with residual weighting from the ReZero paper.
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
        super(ReZeroXL, self).__init__()
        self.name = name
        rezero_xl_submodule = ReZeroXLBlock(
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
            submodule=rezero_xl_submodule,
            num_layers=num_layers,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
        )
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)
        self.mem = None
        self.memory.reset()

    def get_name(self):
        return self.name


class ReZeroGTrXL(nn.Module):
    """
    GTrXL model with rezero submodule.
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
        super(ReZeroGTrXL, self).__init__()
        self.name = name
        rezero_gtrxl_submodule = ReZeroGTrXLBlock(
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
            submodule=rezero_gtrxl_submodule,
            num_layers=num_layers,
            num_heads=num_heads,
            mem_len=mem_len,
            dropout=dropout,
        )
        self.viz_data = []
        self.attn_output_weight = None

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        x, self.attn_output_weight, self.mem = self.memory(x, self.mem)
        x = x.transpose(0, 1)
        return x

    def reset(self):
        self.viz_data.append(self.attn_output_weight)
        self.mem = None
        self.memory.reset()

    def get_name(self):
        return self.name
