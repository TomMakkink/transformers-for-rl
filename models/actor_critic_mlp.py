import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from models.common import mlp


class ActorCriticMLP(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        memory,
    ):
        super(ActorCriticMLP, self).__init__()
        assert len(hidden_size) > 0, "Hidden Layer sizes cannot be empty list"
        hidden_size_ = hidden_size.copy()
        hidden_size_.insert(0, state_size)
        self.fc_network = mlp(hidden_size_, nn.ReLU, nn.ReLU)
        self.memory_network = memory
        self.fc_policy = nn.Linear(hidden_size_[-1], action_size)
        self.fc_value_function = nn.Linear(hidden_size_[-1], 1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, features)

        Returns:
            Network outputs last sequence of shape (batch_size, features)
        """
        x = self.fc_network(x)
        if self.memory_network is not None:
            skip_conn_input = x
            x = self.memory_network(x)
            x = F.relu(x) + skip_conn_input
        x = x[:, -1, :]  # Only use most recent sequence
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value

    def reset(self):
        if self.memory_network is not None:
            self.memory_network.reset()
