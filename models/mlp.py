import torch.nn as nn
from models.memory import Memory


class MLP(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size=128, memory_type="None",
    ):
        super(MLP, self).__init__()
        self.fc_network = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU())
        self.memory_network = Memory(memory_type, hidden_size, hidden_size)
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        """
        Args: 
            x: input tensor of shape (seq_len, batch_size, features)

        Returns:
            Network outputs last sequence of shape (batch_size, features)
        """
        assert len(x.shape) == 3
        x = self.fc_network(x)
        if self.memory_network.memory:
            x = self.memory_network(x)
        x = x[-1]
        x = self.network(x)
        return x

    def reset(self):
        self.memory_network.reset()

