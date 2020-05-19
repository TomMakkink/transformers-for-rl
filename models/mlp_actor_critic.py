import torch 
import torch.nn as nn 
import gym
from torch.distributions import Beta

class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
    ):
        """
        Args: 
        """
        super(MLPActorCritic, self).__init__() 
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.shared_network = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softplus(), 
        )
        self.beta_head = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softplus(), 
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, obs, action):
        x = self.shared_network(obs)

        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        action_dist = Beta(alpha, beta)
        logp = action_dist.log_prob(action)

        value = torch.squeeze(self.critic(x), -1)
        ent = action_dist.entropy().mean().item()
        return action, value, logp, ent
    
    def select_action(self, obs):
        with torch.no_grad():
            x = self.shared_network(obs)

            alpha = self.alpha_head(x) + 1
            beta = self.beta_head(x) + 1
            action_dist = Beta(alpha, beta)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            
            value = torch.squeeze(self.critic(x), -1)
        return action.numpy(), value.numpy(), logp.numpy()
        

