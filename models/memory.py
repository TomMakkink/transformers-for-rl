import torch
import torch.nn as nn
from transformers.transformer_wrapper import Transformer
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config

from transformers.components import MHA, RMHA, LMHA


class Memory(nn.Module):
    """
    Memory wrapper that is either an LSTM or a Transformer. 
    """

    def __init__(self, memory_type, input_dim, output_dim):
        super(Memory, self).__init__()
        self.memory_type = memory_type.lower()
        if self.memory_type is None:
            self.memory = None
        elif self.memory_type == "lstm":
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
        elif self.memory_type in ["vanilla", "rezero", "linformer", "xl", "gtrxl"]:
            self.memory = Transformer(
                d_model=input_dim, output_dim=output_dim, **transformer_config
            )
        elif self.memory_type == "mha":
            self.memory = MHA(input_dim, transformer_config["num_heads"])
        elif self.memory_type == "lmha":
            self.memory = LMHA(input_dim, transformer_config["num_heads"])
        elif self.memory_type == "rmha":
            self.memory = RMHA(
                input_dim,
                transformer_config["num_heads"],
                mem_len=transformer_config["mem_len"],
            )
        else:
            self.memory = None

    def forward(self, x):
        """
        x: shape [seq_len, batch_size, feature_dim] 
        """
        if type(self.memory) is nn.LSTM:
            batch_size = x.shape[1]
            if batch_size > 1:
                x, self.hidden = self.memory(x)
            else:
                x, self.hidden = self.memory(x, self.hidden)

        elif type(self.memory) is Transformer:
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
        elif self.memory_type in ["xl", "gtrxl"]:
            self.memory.reset_mem()

