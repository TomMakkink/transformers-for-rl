import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
from models.memory import Memory
from typing import List


class ActorCriticMLP(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size: List[int] = [256, 128],
        memory_type="None",
    ):
        super(ActorCriticMLP, self).__init__()
        self.fc_network = nn.Sequential(
            nn.Linear(state_size, hidden_size[0]), nn.ReLU()
        )
        self.memory_network = Memory(
            memory_type, input_dim=hidden_size[0], output_dim=hidden_size[0]
        )

        self.fc_policy = nn.Linear(hidden_size[0], action_size)
        self.fc_value_function = nn.Linear(hidden_size[0], 1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, features)
        Returns:
            Network outputs last sequence of shape (batch_size, features)
        """
        x = self.fc_network(x)
        if self.memory_network.memory:
            # Memory recieves input of shape (seq_len, batch_size, features)
            x = x.transpose(0, 1)
            x = self.memory_network(x)
            x = x.transpose(0, 1)
        x = x[:, -1, :]  # Only use most recent sequence
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value

    def reset(self):
        self.memory_network.reset()

