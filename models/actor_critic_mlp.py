import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
from models.memory import Memory
from models.common import mlp


class ActorCriticMLP(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size, memory_type="None",
    ):
        super(ActorCriticMLP, self).__init__()
        assert len(hidden_size) > 0, "Hidden Layer sizes cannot be empty list"
        self.fc_network = nn.Sequential(
            nn.Linear(state_size, hidden_size[0]), nn.ReLU()
        )
        self.memory_network = Memory(memory_type, hidden_size[0], hidden_size[0])

        policy_size = hidden_size.copy()
        policy_size.append(action_size)
        value_size = hidden_size.copy()
        value_size.append(1)

        self.fc_policy = mlp(policy_size, nn.ReLU)
        self.fc_value_function = mlp(value_size, nn.ReLU)

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
