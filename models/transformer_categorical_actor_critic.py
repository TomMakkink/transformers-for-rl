import torch 
import torch.nn as nn
import numpy as np 
from torch.distributions import Categorical
from transformers.transformer_wrapper import Transformer
from configs.transformer_config import transformer_config
import gym

class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
    ):
        super().__init__()
        obs_dim = observation_space.shape[1]
        act_dim = action_space.n
        # self.obs_linear = nn.Sequential(
        #     nn.Linear(obs_dim, 32), 
        #     nn.ReLU(), 
        #     nn.Linear(32, 64))
        self.transformer = Transformer(**transformer_config)
        self.actor = nn.Sequential(nn.Linear(32, act_dim), nn.ReLU())
        self.critic = nn.Linear(32, 1) 

    def forward(self, obs, action):
        """
        Args: 
            obs: [seq_len, features],
            action: [seq_len]               
        """
        # obs = self.obs_linear(obs)
        # [seq_len, batch_size, features]
        obs = obs.unsqueeze(1)
        # print(f"Shape of obs: {obs.shape}")
        obs = self.transformer(obs)[-1]
        # print(f"Obs output shape: {obs.shape}")
        # obs = obs.permute(1, 0, 2)
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
        with torch.no_grad():
            # obs = self.obs_linear(obs)
            # Transformer input shape: [seq_len, batch_size, features]
            obs = obs.unsqueeze(1)
            obs = self.transformer(obs)[-1]             # Take last element of sequence
            # Linear layer: [batch_size, *, features])
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            value = torch.squeeze(self.critic(obs), -1)
        # Only return last action/value/logp
        return action.cpu().numpy(), value.cpu().numpy(), logp.cpu().numpy()