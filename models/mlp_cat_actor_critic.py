import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical
import gym

class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
    ):
        super().__init__()
        # print(f"Obs space: {observation_space}") --> Box(1, 6)
        # print(f"Obs shape: {observation_space.shape}") --> Obs shape: (1, 6)
        obs_dim = observation_space.shape[1]
        act_dim = action_space.n
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, act_dim), 
            nn.ReLU(), 
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        ) 
        self._reset_parameters()

    def _reset_parameters(self, gain=1):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=gain)

    def forward(self, obs, action):
        obs = obs.squeeze()
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(action)
        value = torch.squeeze(self.critic(obs), -1)
        ent = action_dist.entropy().mean().item()
        return action, value, logp, ent
    
    def select_action(self, obs):
        with torch.no_grad():
            obs = obs.squeeze()
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            value = torch.squeeze(self.critic(obs), -1)
        return action.cpu().numpy(), value.cpu().numpy(), logp.cpu().numpy()