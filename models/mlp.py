from typing import List

import torch.nn as nn

from models.common import mlp
from models.memory import Memory


class MLP(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size: List[int], memory_type=None,
    ):
        super(MLP, self).__init__()
        self.fc_network = nn.Sequential(
            nn.Linear(state_size, hidden_size[0]), nn.ReLU()
        )
        self.memory_network = Memory(
            memory_type, input_dim=hidden_size[0], output_dim=hidden_size[0]
        )
        hidden_size_ = [*hidden_size]
        hidden_size_.append(action_size)
        self.network = mlp(hidden_size_, nn.ReLU)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, features)

        Returns:
            Network outputs last sequence of shape (batch_size, features)
        """
        assert len(x.shape) == 3
        x = self.fc_network(x)
        if self.memory_network.memory:
            x = x.transpose(0, 1)
            x = self.memory_network(x)
            x = x.transpose(0, 1)
        x = self.network(x)
        x = x[:, -1, :]  # Only use most recent sequence
        return x

    def reset(self):
        self.memory_network.reset()
