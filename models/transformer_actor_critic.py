import torch 
import torch.nn as nn 
from torch.distributions import Normal
import numpy as np
from configs.transformer_config import transformer_config
from transformers.transformer_wrapper import Transformer

# class ResNet(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, stride=1)
#         # self.conv_2 = nn.Conv2d()

#         self.max_pool = nn.MaxPool2d(3, stride=2),
#         # self.relu_1 = nn.ReLU()

#     def forward(self, obs):

       


class TransformerGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(
            # Transformer(d_model=obs_dim, output_dim=126, **transformer_config),
            ResNet(3, 3), 
            nn.Linear(126, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Linear(64, act_dim),
            nn.Tanh(), 
        )

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs) 
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        print("Hope this works")
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class TransformerCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(), 
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Tanh(), 
        )

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class TransformerActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = torch.prod(torch.tensor(observation_space.shape))

        # policy builder depends on action space
        self.pi = TransformerGaussianActor(obs_dim, action_space.shape[0])

        # build value function
        self.v  = TransformerCritic(obs_dim)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().detach().numpy(), v.cpu().detach().numpy(), logp_a.cpu().detach().numpy()

    def act(self, obs):
        return self.step(obs)[0]