import torch.nn as nn
from torch.distributions import Categorical
from .input_layers import LinearInputLayer


class ActorCriticMLP(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticMLP, self).__init__()

        state_size = observation_space.shape[1]
        action_size = action_space.n

        hidden_layers_size = [128]
        self.fc_shared = LinearInputLayer(state_size, hidden_layers_size)
        self.fc_policy = nn.Linear(hidden_layers_size[-1], action_size)
        self.fc_value_function = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        x = self.fc_shared(x)
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers_size=[128]):
        super(Actor, self).__init__()

        state_size = observation_space.shape[1]
        action_size = action_space.n

        self.input_layer = LinearInputLayer(state_size, hidden_layers_size)
        self.fc_policy = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        x = self.input_layer(x)
        logits = self.fc_policy(x)
        dist = Categorical(logits=logits)
        return dist


class Critic(nn.Module):
    def __init__(self, observation_space, hidden_layers_size=[128]):
        super(Critic, self).__init__()

        state_size = observation_space.shape[1]

        self.input_layer = LinearInputLayer(state_size, hidden_layers_size)
        self.fc_value_function = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        x = self.input_layer(x)
        value = self.fc_value_function(x)
        return value
