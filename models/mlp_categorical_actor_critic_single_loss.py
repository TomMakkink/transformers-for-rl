import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, action_space.n), 
            nn.ReLU(), 
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, obs, action):
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(action)
        value = torch.squeeze(self.critic(obs), -1)
        ent = action_dist.entropy().mean().item()
        return action, value, logp, ent
    
    def select_action(self, obs):
        with torch.no_grad():
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            value = torch.squeeze(self.critic(obs), -1)
        return action.numpy(), value.numpy(), logp.numpy()