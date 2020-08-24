import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import random


class MLP(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(MLP, self).__init__()

        state_size = observation_space.shape[1]
        self.action_size = action_space.n

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

    def forward(self, x):
        return self.network(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
            # action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action

