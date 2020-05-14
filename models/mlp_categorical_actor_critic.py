import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical


class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, act_dim),
            nn.ReLU(),
        )

    def forward(self, obs, action=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        action_dist = self._distribution(obs)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(action_dist, action)
        return action_dist, logp_a

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, action_dist, action):
        return action_dist.log_prob(action)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
        

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        self.actor = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            action_dist = self.actor._distribution(obs)
            action = action_dist.sample()
            logp = self.actor._log_prob_from_distribution(action_dist, action)
            value = self.critic(obs)
        return action.numpy(), value.numpy(), logp.numpy()

    def act(self, obs):
        return self.step(obs)[0]