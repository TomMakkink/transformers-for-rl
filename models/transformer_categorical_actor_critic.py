import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical
from transformers.transformer_wrapper import Transformer


class TransformerActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Observation_space shape: [seq_len, features]
        obs_dim = observation_space.shape[1]
        self.transformer = Transformer(obs_dim, 8, "rezero", 2, 2, 32, 0.1, 0)
        self.actor = nn.Sequential(
            nn.Linear(64, action_space.n), 
            nn.ReLU(), 
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, obs, action):
        """
        Args: 
            obs: [batch_size, seg_len, dim], 
            action: [batch_size]
        """
        obs = obs.permute(1, 0, 2)
        obs = self.transformer(obs)
        seq_len, batch_size, output_dim = obs.shape
        obs = obs.reshape(batch_size, seq_len * output_dim)
        logits = self.actor(obs)
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(action)
        value = torch.squeeze(self.critic(obs), -1)
        ent = action_dist.entropy().mean().item()
        return action, value, logp, ent
    
    def select_action(self, obs):
        """
        Args: 
            obs: [seq_len, features]
        """
        obs = obs.unsqueeze(1)
        with torch.no_grad():
            obs = self.transformer(obs)
            obs = obs.view(-1)
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            value = torch.squeeze(self.critic(obs), -1)
        return action.numpy(), value.numpy(), logp.numpy()