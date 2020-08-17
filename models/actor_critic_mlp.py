import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch


class ActorCriticMLP(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(ActorCriticMLP, self).__init__()

        state_size = observation_space.shape[1]
        action_size = action_space.n

        self.fc_shared = nn.Linear(state_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, action_size)
        self.fc_value_function = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value