from typing import List

import torch.nn as nn

from models.common import mlp
from models.memory import Memory


class MLP(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size: List[int], memory_type=None,
    ):
        super(MLP, self).__init__()
        hidden_size_ = hidden_size.copy()
        hidden_size_.insert(0, state_size)
        self.fc_network = mlp(hidden_size_, nn.ReLU, nn.ReLU)
        self.memory_network = Memory(memory_type, hidden_size_[-1], hidden_size_[-1])
        self.network = nn.Linear(hidden_size_[-1], action_size)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, features)

        Returns:
            Network outputs last sequence of shape (batch_size, features)
        """
        # assert len(x.shape) == 3
        x = self.fc_network(x)
        # if self.memory_network.memory:
        #     x = x.transpose(0, 1)
        #     x = self.memory_network(x)
        #     x = x.transpose(0, 1)
        x = self.network(x)
        x = x[:, -1, :]  # Only use most recent sequence
        return x

    def reset(self):
        self.memory_network.reset()
