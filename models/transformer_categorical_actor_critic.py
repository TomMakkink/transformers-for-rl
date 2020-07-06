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
        self.transformer = Transformer(**transformer_config)
        self.actor = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, act_dim), 
            nn.ReLU(), 
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        ) 

    def forward(self, obs, action):
        """
        Args: 
            obs: [batch_size, seg_len, dim],   # TODO: Update 
            action: [batch_size]               
        """
        # [seq_len, dim] -> [batch_size]
        print(f"Obs shape: {obs.shape}")
        obs = self.transformer(obs)
        
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
            obs = self.transformer(obs)
            logits = self.actor(obs)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            value = torch.squeeze(self.critic(obs), -1)
            # Only return last action/value/logp
        return action.cpu().numpy()[-1], value.cpu().numpy()[-1], logp.cpu().numpy()[-1]