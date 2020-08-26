import torch
import torch.nn as nn
from transformers.transformer_wrapper import Transformer
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config


class Memory(nn.Module):
    """
    Memory wrapper that is either an LSTM or a Transformer. 
    """

    def __init__(self, memory_type, input_dim, output_dim):
        super(Memory, self).__init__()
        if memory_type.lower() == "lstm":
            self.memory_network = nn.LSTM(
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
        elif memory_type.lower() in ["vanilla", "rezero", "linformer", "xl", "gtrxl"]:
            self.memory_network = Transformer(
                d_model=input_dim, output_dim=output_dim, **transformer_config
            )
        else:
            self.memory_network = None

    def forward(self, x):
        """
        x: shape [batch_size, feature_dim] 
        """
        if type(self.memory_network) is nn.LSTM:
            x, self.hidden = self.memory_network(x, self.hidden)
        elif type(self.memory_network) is Transformer:
            x = x.unsqueeze(1)
            x = self.memory_network(x)
            x = x.squeeze(1)
        return x
