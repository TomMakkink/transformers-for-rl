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
        x = self.fc_network(x)
        x = self.memory_network(x) if self.memory_network else x
        return self.network(x)

    def reset(self, x):
        self.memory_network.reset()

