import torch 
import torch.nn as nn 
import numpy as np
from configs.transformer_config import transformer_config
from transformers.transformer_wrapper import Transformer
from models.resnet import ResNet

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.alpha_head = nn.Sequential(
            nn.Linear(100, 1),
            nn.Softplus(), 
        )
        self.beta_head = nn.Sequential(
            nn.Linear(100, 1),
            nn.Softplus(), 
        )

    def forward(self, state):
        alpha = self.alpha_head(state) + 1
        beta = self.beta_head(state) + 1
        return alpha, beta


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.critic_net = nn.Linear(100, 1)

    def forward(self, state):
        return self.critic_net(state)   


class TransformerActorCritic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.resnet = ResNet()
        self.transformer = Transformer(**transformer_config)
        self.fully_connected = nn.Sequential(
            nn.Linear(800, 256), 
            nn.ReLU(), 
            nn.Linear(256, 100),
            nn.ReLU(), 
        )
        self.actor = Actor(obs_shape, action_shape)
        self.critic = Critic(obs_shape)

    def forward(self, state):
        """
        Args: 
            state: Input observations. Shape: [batch_size, seq_len, channels, height, width]

        Returns: 
            alpha, beta, value
        """
        batch_size, seq_len, channels, height, width = state.shape
        state = state.reshape(batch_size * seq_len, channels, height, width)
        state = self.resnet(state)
        state = state.reshape(batch_size, seq_len, 800)
        state = self.transformer(state)
        state = self.fully_connected(state)
        alpha, beta = self.actor(state)
        value = self.critic(state)
        return alpha, beta, value


    # def act(self, obs):
    #     return self.step(obs)[0]