import torch.nn as nn
from torch.distributions import Categorical
<<<<<<< HEAD
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
=======
import torch.nn.functional as F
import torch
from models.memory import Memory


class ActorCriticMLP(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_size=128, memory_type="None",
    ):
        super(ActorCriticMLP, self).__init__()
        self.fc_network = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU())
        self.memory_network = Memory(memory_type, hidden_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, action_size)
        self.fc_value_function = nn.Linear(hidden_size, 1)

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
>>>>>>> 66408be54c5c559a731078fee1acad15f67046f4
        logits = self.fc_policy(x)
        value = self.fc_value_function(x)
        dist = Categorical(logits=logits)
        return dist, value

<<<<<<< HEAD

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
=======
    def reset(self):
        self.memory_network.reset()
>>>>>>> 66408be54c5c559a731078fee1acad15f67046f4
