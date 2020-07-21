import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPA2C(nn.Module):
    def __init__(self, observation_space, action_space, h_size=128):
        super(MLPA2C, self).__init__()

        s_size = observation_space.shape[1]
        a_size = action_space.n

        self.fc_shared = nn.Linear(s_size, h_size)

        self.fc_policy = nn.Linear(h_size, a_size)

        self.fc_value_function = nn.Linear(h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
       
        logits = self.fc_policy(x)

        value = self.fc_value_function(x)

        dist = Categorical(logits=logits)

        return dist, value
