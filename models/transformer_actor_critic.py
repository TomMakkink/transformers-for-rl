import torch 
import torch.nn as nn 
from torch.distributions import Normal
import numpy as np
from configs.transformer_config import transformer_config
from transformers.transformer_wrapper import Transformer
from models.resnet import ResNet

class TransformerGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.resnet = ResNet()
        self.transformer = Transformer(d_model=2592, output_dim=512, **transformer_config)
        self.mu_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True), 
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, act_dim), 
            nn.Tanh(), 
        )

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions. Sequence_len = number of frames. 
        actor = self._distribution(obs) 
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(actor, act)
        return actor, logp_a

    def _distribution(self, obs):
        batch_size, sequence_len, channels, height, width = obs.shape 
        obs = obs.reshape(batch_size * sequence_len, channels, height, width)
        conv_out = self.resnet(obs)
        conv_out = conv_out.reshape(sequence_len, batch_size, 2592)
        trans_out = self.transformer(conv_out)[0]
        mu = self.mu_net(trans_out)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, actor, act):
        # Need to double check this 
        return actor.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class TransformerCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.resnet = ResNet()
        self.transformer = Transformer(d_model=2592, output_dim=512, **transformer_config)
        self.critic_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 1), 
        )

    def forward(self, obs):
        batch_size, sequence_len, channels, height, width = obs.shape 
        obs = obs.reshape(batch_size * sequence_len, channels, height, width)
        conv_out = self.resnet(obs)
        conv_out = conv_out.reshape(sequence_len, batch_size, 2592)
        trans_out = self.transformer(conv_out)[0]
        value = self.critic_net(trans_out)
        return value           # Critical to ensure v has right shape.


class TransformerActorCritic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()

        # policy builder depends on action space
        self.actor = TransformerGaussianActor(obs_shape, action_shape)

        # build value function
        self.critic  = TransformerCritic(obs_shape)

    def step(self, obs):
        with torch.no_grad():
            actor = self.actor._distribution(obs)
            a = actor.sample()
            logp_a = self.actor._log_prob_from_distribution(actor, a)
            critic = self.critic(obs)
        return a.cpu().detach().numpy(), critic.cpu().detach().numpy(), logp_a.cpu().detach().numpy()

    def act(self, obs):
        return self.step(obs)[0]