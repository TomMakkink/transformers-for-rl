from agents.agent import Agent
from models.actor_critic_mlp import ActorCriticMLP
from configs.a2c_config import a2c_config
from configs.experiment_config import experiment_config
import numpy as np
import torch
import torch.optim as optim


class A2C(Agent):
    def __init__(self, state_size, action_size, memory):
        super(A2C, self).__init__(state_size, action_size, memory)
        self.device = experiment_config["device"]
        self.net = ActorCriticMLP(state_size, action_size, memory_type=memory).to(
            self.device
        )
        self.optimiser = optim.Adam(self.net.parameters(), lr=a2c_config["lr"])
        self.log_probs = []
        self.values = []
        self.rewards = []

    def _compute_returns(self):
        R = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + a2c_config["gamma"] * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns -= returns.mean()
        if returns.std() > 0.0:
            returns /= returns.std()
        return returns

    def optimize_network(self):
        returns = self._compute_returns()
        returns = torch.from_numpy(returns).float().to(self.device)

        values = torch.cat(self.values)
        log_probs = torch.cat(self.log_probs)

        delta = returns - values
        policy_loss = -torch.sum(log_probs * delta.detach())
        value_function_loss = 0.5 * torch.sum(delta ** 2)
        loss = policy_loss + value_function_loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def reset(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.net.reset()

    def act(self, state):
        dist, value = self.net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob)
        self.values.append(value)

        return action.item()

    def collect_experience(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

