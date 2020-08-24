from agents.agent import Agent
from configs.a2c_config import a2c_config
import numpy as np
import torch
import torch.optim as optim


class A2C(Agent):
    def __init__(self, model, env, device):
        super(A2C, self).__init__(model, env, device)
        self.net = model(env.observation_space, env.action_space).to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=a2c_config["lr"])
        self.log_probs = []
        self.values = []

    def _compute_returns(self, rewards):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + a2c_config["gamma"] * R
            returns.insert(0, R)
        returns = np.array(returns)
        returns -= returns.mean()
        if returns.std() > 0.0:
            returns /= returns.std()
        return returns

    def optimise_network(self, rewards):
        returns = self._compute_returns(rewards)
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

        self.values = []
        self.log_probs = []
        return loss

    def act(self, state):
        dist, value = self.net(state)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob.unsqueeze(0))
        self.values.append(value)

        return action.item()

