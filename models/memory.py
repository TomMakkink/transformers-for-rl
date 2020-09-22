import torch
import torch.nn as nn
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config

from transformers.transformer import (
    TransformerModel,
    MemoryTransformerModel,
    get_transformer_submodule,
)


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
            if self.memory_type == "lstm":
                self.memory = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=lstm_config["hidden_dim"],
                    num_layers=lstm_config["num_layers"],
                )
                self.hidden = (
                    torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                        experiment_config["device"]
                    ),
                    torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                        experiment_config["device"]
                    ),
                )
            elif self.memory_type == "stable":
                self.memory = MemTransformerLM(
                    n_token=None,
                    n_layer=transformer_config["num_layers"],
                    n_head=transformer_config["num_heads"],
                    d_head=input_dim // transformer_config["num_heads"],
                    d_model=input_dim,
                    d_inner=transformer_config["dim_mlp"],
                    dropout=0.1,
                    dropatt=0.0,
                    mem_len=transformer_config["mem_len"],
                    use_stable_version=True,
                    use_gate=True,
                )
                self.memory.apply(weights_init)
                self.memory.init_gru_bias()
                self.mem = None

            elif self.memory_type in ["vanilla", "rezero", "linformer", "mha", "lmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.memory = TransformerModel(input_dim, output_dim, submodule)
            elif self.memory_type in ["gtrxl", "xl", "rmha", "gmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.mem = None
                self.memory = MemoryTransformerModel(input_dim, output_dim, submodule)

    def forward(self, x):
        """
        x: shape [seq_len, batch_size, feature_dim]
        """
        if type(self.memory) is nn.LSTM:
            batch_size = x.shape[1]
            if batch_size > 1:
                print("Batch Size above 1")
                x, self.hidden = self.memory(x)
            else:
                x, self.hidden = self.memory(x, self.hidden)
        elif type(self.memory) == MemoryTransformerModel:
            x, self.mem = self.memory(x, self.mem)
        elif type(self.memory) == TransformerModel:
            x = self.memory(x)
        return x

    def reset(self):
        if self.memory_type == "lstm":
            self.hidden = (
                torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                    experiment_config["device"]
                ),
                torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                    experiment_config["device"]
                ),
            )
        elif type(self.memory) == MemoryTransformerModel:
            self.mem = None
            self.memory.reset()
