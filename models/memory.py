import torch
import torch.nn as nn
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config
from models.custom_lstm import CustomLSTM
from transformers.transformer_models import TransformerModel, MemoryTransformerModel
from transformers.transformer_submodules import get_transformer_submodule


class Memory(nn.Module):
    """
    Memory wrapper that is either an LSTM or a Transformer. 
    """

    def __init__(self, memory_type, input_dim, output_dim):
        super(Memory, self).__init__()
        self.memory = None
        self.memory_type = None

        if memory_type is not None:
            self.memory_type = memory_type.lower()
            print(f"Using {self.memory_type}...")

            self.visualisation_data = [[]]

            if self.memory_type == "lstm":
                self.memory = CustomLSTM(input_size=input_dim,
                                         hidden_size=lstm_config["hidden_dim"])
                self.hidden = None

            elif self.memory_type in ["vanilla", "rezero", "linformer", "mha", "lmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.memory = TransformerModel(input_dim, output_dim, submodule)

            elif self.memory_type in ["gtrxl", "xl", "rmha", "gmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.mem = None
                self.memory = MemoryTransformerModel(input_dim, output_dim, submodule)

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, feature_dim]
        """
        if (type(self.memory) is nn.LSTM) or (type(self.memory) is CustomLSTM):
            x, self.hidden, viz_data = self.memory(x, self.hidden)
            # x, self.hidden = self.memory(x, self.hidden)
            # viz_data = self.hidden

        # Transformers expect input of shape: [seq_len, batch_size, feature_dim]
        x = x.transpose(0, 1)
        if type(self.memory) == MemoryTransformerModel:
            x, viz_data, self.mem = self.memory(x, self.mem)
        elif type(self.memory) == TransformerModel:
            x, viz_data = self.memory(x)
        x = x.transpose(0, 1)

        self.visualisation_data[-1].append(viz_data)
        return x

    def reset(self):
        if self.memory_type == "lstm":
            self.hidden = None

        elif type(self.memory) == MemoryTransformerModel:
            self.mem = None
            self.memory.reset()
        if self.memory_type is not None:
            self.visualisation_data.append([])
